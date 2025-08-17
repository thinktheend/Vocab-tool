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
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            prompt = data.get("prompt")

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Server configuration error: The OPENAI_API_KEY is missing from Vercel's environment variables.")

            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert assistant for the Fast Conversational Spanish program. You must follow all rules and formatting instructions provided by the user precisely. Your final output must be ONLY the raw content (HTML or Markdown) as requested, with absolutely no commentary, greetings, or extra text like ```html."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            ai_content = completion.choices[0].message.content
            
            if "```" in ai_content:
                parts = ai_content.split("```")
                if len(parts) > 1:
                    ai_content = parts[1]
                    if ai_content.startswith(('html', 'markdown')):
                        ai_content = ai_content.split('\n', 1)[1]
            
            response_payload = {"content": ai_content.strip()}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

        except Exception as e:
            print(f"AN ERROR OCCURRED: {e}") 
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            error_payload = {"error": "An internal server error occurred.", "details": str(e)}
            self.wfile.write(json.dumps(error_payload).encode('utf-8'))
