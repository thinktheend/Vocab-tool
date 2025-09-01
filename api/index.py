import os
import re
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

# Optional environment overrides (leave unset if not needed)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")

# Strip accidental code fences from model output
FENCE_RE = re.compile(r"^\s*```(?:html|xml|markdown)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE)

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

            # Single pass, strong contract. Plenty of tokens for long outputs.
            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.9,
                max_tokens=15000,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert FCS assistant. Return ONLY full raw HTML (a complete, valid document). "
                            "STRICTLY follow the embedded contract in the user's HTML prompt: "
                            "• Obey all UI selections/quantities exactly (levels, sections, counts, turns, sentences/turn). "
                            "• Nouns section = noun words/phrases ONLY (no sentences), grouped by subcategory header rows; NO highlighting in Nouns. "
                            "• Verbs section = full sentences; highlight ONLY the verb (exactly one <span class=\"en\">…</span> in English cell and one <span class=\"es\">…</span> in Spanish cell). "
                            "• Descriptive section = full sentences; highlight ONLY the descriptive word (exactly one <span class=\"en\">…</span> and one <span class=\"es\">…</span>); "
                            "  each descriptive sentence must reference nouns/verbs introduced in this output. "
                            "• The REQUIRED QUANTITY is the number of DISTINCT red-highlighted Spanish terms (<span class=\"es\">…</span>) across Verbs + Descriptive; "
                            "  this count MUST fall within the level/UI min–max range. Do NOT count Nouns. "
                            "• If #conversations=1 AND #speakers=1, produce a monologue obeying exact turns/sentences-per-turn and maximize length. "
                            "• Enforce CEFR-like differences between levels. "
                            "• No empty <tbody>; SELF-CHECK all constraints BEFORE responding. "
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
