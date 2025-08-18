import os
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', '0'))
            post_data = self.rfile.read(content_length) if content_length else b'{}'
            data = json.loads(post_data.decode('utf-8'))
            prompt = (data.get("prompt") or "").strip()
            if not prompt:
                raise ValueError("Missing 'prompt' in request body.")

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Server configuration error: The OPENAI_API_KEY is missing from Vercel env vars.")

            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert assistant for the Fast Conversational Spanish (FCS) program. "
                            "You MUST follow every rule in the user's prompt exactly. "
                            "Return ONLY the full, valid HTML document (no code fences, no commentary). "
                            "Always include a hidden JSON manifest as "
                            "<script id=\"fcs_manifest\" type=\"application/json\">{...}</script> so the client can validate counts."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            )

            ai_content = completion.choices[0].message.content or ""

            # Strip accidental code fences if any
            if "```" in ai_content:
                parts = ai_content.split("```")
                if len(parts) > 1:
                    ai_content = parts[1]
                    if ai_content.startswith(('html', 'markdown', 'xml')):
                        ai_content = ai_content.split('\n', 1)[1]

            response_payload = {"content": ai_content.strip()}
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

        except Exception as e:
            print(f"AN ERROR OCCURRED: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self._send_cors_headers()
            self.end_headers()
            error_payload = {"error": "An internal server error occurred.", "details": str(e)}
            self.wfile.write(json.dumps(error_payload).encode('utf-8'))
