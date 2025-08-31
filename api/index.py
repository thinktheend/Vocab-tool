import os
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

# You can override with an env var on Vercel: OPENAI_MODEL=gpt-4o
MODEL_DEFAULT = os.environ.get("OPENAI_MODEL", "gpt-4o")

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
            prompt = data.get("prompt", "").strip()
            if not prompt:
                raise ValueError("Missing 'prompt' in request body.")

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Server configuration error: The OPENAI_API_KEY is missing from env vars.")

            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model=MODEL_DEFAULT,
                temperature=0.9,
                top_p=0.9,
                max_tokens=12000,   # allow long monologues / many rows
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert assistant for the Fast Conversational Spanish (FCS) program. "
                            "Return ONLY a full, valid, self-contained HTML document. "
                            "BEFORE YOU REPLY: verify that every required section is populated. "
                            "Specifically: every <tbody> MUST contain fully generated rows (no empty <tbody>, no placeholders). "
                            "If any table would be empty, generate appropriate rows so the document is complete. "
                            "Do NOT add explanations or code fences."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            )

            ai_content = completion.choices[0].message.content or ""
            # Safety: strip code fences if the model adds them
            if "```" in ai_content:
                parts = ai_content.split("```")
                if len(parts) > 1:
                    candidate = parts[1]
                    # strip optional language tag
                    lowers = candidate.lower()
                    if lowers.startswith(("html", "xml", "markdown")):
                        ai_content = candidate.split("\n", 1)[1] if "\n" in candidate else candidate
                    else:
                        ai_content = candidate

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
            self.wfile.write(json.dumps({"error": "An internal server error occurred.", "details": str(e)}).encode('utf-8'))
