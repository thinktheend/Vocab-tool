import os
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")

# Tuned client options: avoid long internal retries (faster failures) and cap total wait.
_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url=OPENAI_BASE_URL or None,
    organization=OPENAI_ORG_ID or None,
    timeout=60.0,         # Hard client timeout per request (seconds)
    max_retries=0,        # Don't silently retry for 60+ sec; fail fast so frontend can re-try if needed
)

SYSTEM_PROMPT = (
    "You are an expert assistant for the Fast Conversational Spanish (FCS) program. "
    "Return ONLY the raw HTML (a full, valid, self-contained document). "
    "You MUST strictly honor all counts/limits/section names implied by the user's prompt. "
    "No table may have an empty <tbody>. "
    "If any required section would be empty or mis-sized, you MUST internally regenerate BEFORE replying, "
    "and return only the final compliant HTML. "
    "Do NOT add greetings, explanations, or code fences like ```html."
)

class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length else b"{}"

            try:
                data = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in request body.")

            prompt = (data.get("prompt") or "").strip()
            if not prompt:
                raise ValueError("Missing 'prompt' in request body.")

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Server configuration error: OPENAI_API_KEY is not set in Vercel env vars.")

            # (client is module-level; it will read the env key on init; double-check here)
            if not _client.api_key:
                raise ValueError("Server configuration error: OpenAI client missing API key.")

            # Bigger token budget = less chance of truncation; you said output size > token cost.
            completion = _client.chat.completions.create(
                model="gpt-4o",
                temperature=0.9,
                max_tokens=15000,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )

            ai_content = (completion.choices[0].message.content or "").strip()

            # Strip code fences if any slipped through
            if "```" in ai_content:
                parts = ai_content.split("```")
                for chunk in parts:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    lines = chunk.splitlines()
                    if lines and len(lines[0]) <= 10 and lines[0].lower() in {"html", "xml", "markdown"}:
                        chunk = "\n".join(lines[1:]).strip()
                    ai_content = chunk
                    break
                ai_content = ai_content.strip()

            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"content": ai_content}).encode("utf-8"))

        except Exception as e:
            print(f"AN ERROR OCCURRED: {e}")
            self.send_response(500)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "An internal server error occurred.",
                "details": str(e),
            }).encode("utf-8"))
