import os
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

# This handler is compatible with Vercel Python Serverless Functions.
# Endpoint: /api/index  (expects JSON body: {"prompt": "<full HTML prompt>"} )

MODEL_DEFAULT = os.environ.get("OPENAI_MODEL", "gpt-4o")  # fast & strong; fallback via env if needed

class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        # Allow embedding (Kajabi iframe) and local dev
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
                raise ValueError("Server configuration error: The OPENAI_API_KEY is missing from Vercel env vars.")

            # Call OpenAI (fast model by default, low temperature for consistency)
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model=MODEL_DEFAULT,
                temperature=0.9,
                top_p=0.9,
                max_tokens=10000,   # allow long monologues & vocab sets
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert assistant for the Fast Conversational Spanish (FCS) program. "
                            "Return ONLY the raw HTML (a full, valid, self-contained document). "
                            "BEFORE YOU REPLY: validate that every required section in the user's template is populated. "
                            "No table may have an empty <tbody>. "
                            "If any section would be empty or under-specified, you MUST generate appropriate content to fill it. "
                            "Strictly follow per-turn sentence bounds, vocabulary counts, headers, and section presence EXACTLY as specified. "
                            "Do NOT add greetings, explanations, or code fences like ```html."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            )

            ai_content = completion.choices[0].message.content or ""

            # Strip code fences just in case
            if "```" in ai_content:
                parts = ai_content.split("```")
                if len(parts) > 1:
                    # parts[1] may start with a language tag (html, xml, markdown)
                    candidate = parts[1]
                    if candidate.lower().startswith(('html', 'markdown', 'xml')):
                        ai_content = candidate.split('\n', 1)[1] if '\n' in candidate else candidate
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
            error_payload = {"error": "An internal server error occurred.", "details": str(e)}
            self.wfile.write(json.dumps(error_payload).encode('utf-8'))
