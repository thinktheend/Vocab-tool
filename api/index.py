import os
import re
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")

# If a provider wraps HTML in a code fence, unwrap it.
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

            # Single-shot HTML response.
            completion = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                temperature=0.8,
                max_tokens=min(int(os.getenv("MODEL_MAX_TOKENS", "1000")), 16384),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert FCS assistant. Return ONLY full raw HTML (a valid document). "
                            "Strictly follow the embedded contract inside the user's HTML prompt. "
                            "Vocabulary generator rules (do not change UI/format): "
                            "• NOUNS: words/phrases only (no sentences) with subcategory header rows when required; "
                            "  the Spanish noun is wrapped in <span class=\"es\">…</span> (red). "
                            "• VERBS: full sentences using He/She/It/They + is/are going to + [infinitive]; "
                            "  highlight ONLY the verb (one <span class=\"en\">…</span> in the English cell, "
                            "  one <span class=\"es\">…</span> in the Spanish cell). "
                            "• ADJECTIVES: full sentences with is/are + adjective; highlight ONLY the adjective "
                            "  (one <span class=\"en\">…</span> and one <span class=\"es\">…</span>). "
                            "• ADVERBS: full sentences that reuse verbs, highlight ONLY the adverb "
                            "  (one <span class=\"en\">…</span> and one <span class=\"es\">…</span>). "
                            "Common Phrases/Questions and the Full Vocabulary Index must follow the contract. "
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
