import os
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # optional
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")      # optional

class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-type", "application/json; charset=utf-8")
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

            client = OpenAI(
                api_key=api_key,
                base_url=OPENAI_BASE_URL or None,
                organization=OPENAI_ORG_ID or None,
            )

            # Faster single-pass: push strict self-checking into the prompt and avoid retries here.
            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.9,
                max_tokens=15000,  # allow bigger/longer outputs; user is fine with more tokens
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert FCS assistant. Return ONLY full raw HTML (a complete, valid document). "
                            "STRICTLY follow the embedded contract in the user's HTML prompt: "
                            "• Obey all UI selections/quantities exactly (levels, sections, counts, turns, sentences/turn). "
                            "• Nouns must include subcategories as header rows in the same table. "
                            "• Descriptive rows must reference nouns/verbs introduced in this output. "
                            "• If 1 conversation & 1 speaker, produce a monologue (single-speaker) obeying turns/sentences-per-turn. "
                            "• Level differences must align with CEFR notions included in the prompt. "
                            "• No empty <tbody>; self-check all constraints BEFORE responding. "
                            "Do NOT add explanations or code fences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            ai_content = (completion.choices[0].message.content or "").strip()

            # Strip accidental fences if present
            if "```" in ai_content:
                parts = ai_content.split("```")
                for chunk in parts:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    lines = chunk.splitlines()
                    if lines and len(lines[0]) <= 10 and lines[0].lower() in {"
