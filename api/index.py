import os
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

# This is a standard Python serverless function for Vercel, without Flask.
class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            # --- Read the request from the frontend ---
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            prompt = data.get("prompt")

            # --- Get the OpenAI API Key from Vercel's environment variables ---
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Server configuration error."}).encode('utf-8'))
                return

            # --- Call the OpenAI API ---
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert assistant for the Fast Conversational Spanish program. You must follow all rules provided by the user precisely. Your final output must be ONLY the raw content (HTML or Markdown) as requested, with absolutely no commentary or extra text like ```html."},
                    {"role": "user", "content": prompt}
                ]
            )

            # --- Process and send the response back to the frontend ---
            ai_content = completion.choices[0].message.content
            
            # Create a JSON object to send back, as the frontend expects
            response_payload = {"content": ai_content}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            # Add CORS headers to allow the Kajabi/HTML page to access the response
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

        except Exception as e:
            # Handle any crashes and send back a useful error message
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "An internal server error occurred.", "details": str(e)}).encode('utf-8'))
        return
