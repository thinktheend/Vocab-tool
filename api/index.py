import os
import re
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")

# Unwrap code fences if a provider adds them.
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

            # Bigger budget for strict quantity & longer FIBs
            max_tokens = min(int(os.getenv("MODEL_MAX_TOKENS", "8000")), 16384)

            completion = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                temperature=0.2,  # lower = more compliant to counts
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert FCS assistant. Return ONLY full raw HTML (a valid document). "
                            "Strictly follow the embedded contract inside the user's HTML prompt. "
                            "ABSOLUTE LENGTH COMPLIANCE IS REQUIRED: when numeric ranges are provided, "
                            "produce at least the minimum and not more than the maximum. "
                            "Conversation-specific constraints: "
                            "• Conversations must not contain blanks/underscores (\"________\"). Only the FIB section may use blanks. "
                            "• Every sentence in BOTH columns must begin with a speaker name in brackets, e.g., \"[Ana] ...\" "
                            "• FIB rows must be ONE sentence per column; English colors EXACTLY one target with <span class=\"en\">…</span>; "
                            "  Spanish uses \"(English target) ________\" with parentheses immediately BEFORE the blank. "
                            "• Answer Key must be a single-column list of answer words (no numbering column). "
                            "• \"Vocabulary Used\" must include all unique Answer Key entries at minimum, deduplicated, and be a two-column list (EN | ES). "
                            "Vocabulary generator constraints: "
                            "• Do NOT include any 'Full Vocabulary Index' section. "
                            "• Treat each table row as one item toward the requested total; target the upper bound. "
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
