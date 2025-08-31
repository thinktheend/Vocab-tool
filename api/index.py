import os
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

# Optional: allow custom endpoints (e.g., Azure/OpenAI-compatible providers)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # e.g. "https://api.openai.com/v1"
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")

class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    # (Optional) lightweight health check
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
                raise ValueError(
                    "Server configuration error: OPENAI_API_KEY is not set in Vercel env vars."
                )

            # Build the client. Do NOT pass any 'proxies=' kwarg; httpx is pinned in requirements.
            client = OpenAI(
                api_key=api_key,
                base_url=OPENAI_BASE_URL or None,
                organization=OPENAI_ORG_ID or None,
            )

            # You can swap the model to gpt-4o-mini for speed/cost if desired.
            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.9,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert assistant for the Fast Conversational Spanish (FCS) program. "
                            "Return ONLY the raw HTML (a full, valid, self-contained document). "
                            "BEFORE YOU REPLY: verify that every required section in the user's template is populated. "
                            "No table may have an empty <tbody>. "
                            "If any section would be empty, you MUST generate appropriate content to fill it. "
                            "Do NOT add greetings, explanations, or code fences like ```html."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            ai_content = (completion.choices[0].message.content or "").strip()

            # Strip code fences if the model added them anyway.
            if "```" in ai_content:
                # Take the first fenced block if present
                parts = ai_content.split("```")
                # Try to find the first non-empty fenced content
                for chunk in parts:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    # Remove an optional language hint on the first line (e.g., "html")
                    lines = chunk.splitlines()
                    if lines and len(lines[0]) <= 10 and lines[0].lower() in {"html", "xml", "markdown"}:
                        chunk = "\n".join(lines[1:]).strip()
                    ai_content = chunk
                    break
                ai_content = ai_content.strip()

            # Build response
            response_payload = {"content": ai_content}
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode("utf-8"))

        except Exception as e:
            # Log server-side for Vercel live logs
            print(f"AN ERROR OCCURRED: {e}")
            self.send_response(500)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            error_payload = {
                "error": "An internal server error occurred.",
                "details": str(e),
            }
            self.wfile.write(json.dumps(error_payload).encode("utf-8"))
