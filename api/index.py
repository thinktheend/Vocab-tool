import os
import re
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")

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

            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.9,
                max_tokens=12000,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert FCS assistant. Return ONLY full raw HTML (a valid document). "
                            "STRICTLY follow the embedded contract inside the user's HTML prompt. "
                            "Vocabulary generator rules to enforce: "
                            "• Nouns section = noun words/phrases ONLY (no sentences), grouped by subcategory header rows; "
                            "  the Spanish noun word in each row MUST be wrapped in <span class=\"es\">…</span>. "
                            "• Verbs section = full sentences; highlight ONLY the verb — exactly one <span class=\"en\">…</span> in the English cell "
                            "  and one <span class=\"es\">…</span> in the Spanish cell per row. "
                            "• Descriptive section = full sentences; highlight ONLY the descriptive word — exactly one <span class=\"en\">…</span> "
                            "  and one <span class=\"es\">…</span> per row; each sentence references nouns/verbs introduced in this output. "
                            "• REQUIRED QUANTITY = number of DISTINCT red-highlighted Spanish terms (<span class=\"es\">…</span>) across Nouns + Verbs + Descriptive. "
                            "  This count MUST be within the min–max for the selected level/UI. If short, EXPAND by adding unique items until inside range "
                            "(prefer adding to Nouns distributed across subcategories, then Verbs, then Descriptive). "
                            "• Avoid repeating highlighted Spanish terms across these three sections unless unavoidable. "
                            "• Run a self-check BEFORE responding. "
                            "Conversation generator is unchanged; do not alter it. "
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
