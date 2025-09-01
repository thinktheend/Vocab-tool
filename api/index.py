import os
import re
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # optional
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")      # optional

# Strip accidental code fences safely
FENCE_RE = re.compile(
    r"^\s*```(?:html|xml|markdown)?\s*([\s\S]*?)\s*```\s*$",
    re.IGNORECASE
)

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
                raise ValueError("Server configuration error: OPENAI_API_KEY is not set.")

            client = OpenAI(
                api_key=api_key,
                base_url=OPENAI_BASE_URL or None,
                organization=OPENAI_ORG_ID or None,
            )

            # Single bigger pass for speed; your frontend validators handle strictness.
            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.9,
                max_tokens=120000,  # room for long monologues & large vocab
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert FCS assistant. Return ONLY full raw HTML (a complete, valid document). "
                            "STRICTLY follow the embedded contract in the user's HTML prompt: "
                            "• Obey all UI selections/quantities exactly (levels, sections, counts, turns, sentences/turn). "
                            "• Nouns must include subcategories as header rows in the same table and be noun words only. "
                            "• Verbs: only the verb is highlighted (one <span class=\"en\"> and one <span class=\"es\"> per row). "
                            "• Descriptive: only the single descriptive word is highlighted (one <span class=\"en\"> and one <span class=\"es\"> per row). "
                            "• Descriptive rows must reference nouns/verbs introduced in this output. "
                            "• If 1 conversation & 1 speaker, produce a monologue obeying turns/sentences-per-turn and prefer maximum length. "
                            "• Level differences must align with CEFR notions included in the prompt. "
                            "• No empty <tbody>; self-check all constraints BEFORE responding. "
                            "Do NOT add explanations or code fences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            ai_content = (completion.choices[0].message.content or "").strip()

            m = FENCE_RE.match(ai_content)
            if m:
                ai_content = m.group(1).strip()

            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"content": ai_content}).encode("utf-8"))

        except Exception as e:
            print(f"AN ERROR OCCURRED: {e}")
            self.send_response(500)
            self._send_cors_headers()
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "An internal server error occurred.",
                "details": str(e)
            }).encode("utf-8"))
