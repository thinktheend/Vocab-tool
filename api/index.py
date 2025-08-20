import os
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        # Allow Kajabi (iframe) and local dev
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_POST(self):
        try:
            # Read request
            content_length = int(self.headers.get('Content-Length', '0'))
            post_data = self.rfile.read(content_length) if content_length else b'{}'
            data = json.loads(post_data.decode('utf-8'))
            prompt = data.get("prompt", "").strip()
            if not prompt:
                raise ValueError("Missing 'prompt' in request body.")

            # API key
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Server configuration error: The OPENAI_API_KEY is missing.")

            # Call OpenAI with tighter controls to avoid timeouts and runaway prompts
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.25,
                max_tokens=3200,  # cap output size
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert assistant for the Fast Conversational Spanish (FCS) program. "
                            "Return ONLY a full self-contained HTML document. "
                            "Before you reply, confirm that every section in the user template is filled; no <tbody> may be empty. "
                            "For vocabulary, enforce the exact minimum and maximum counts per section and per level, "
                            "and provide a counts summary table. "
                            "Use <span class='en'> for English (blue) and <span class='es'> for Spanish (red) terms consistently. "
                            "Do not include commentary, code fences, or greetings."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                timeout=210  # seconds (less than typical 300s limits)
            )

            html = completion.choices[0].message.content or ""
            # Strip accidental code fences
            if "```" in html:
                parts = html.split("```")
                if len(parts) > 1:
                    html = parts[1]
                    if html.startswith(('html', 'markdown', 'xml')):
                        html = html.split('\n', 1)[1]

            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"content": html.strip()}).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
