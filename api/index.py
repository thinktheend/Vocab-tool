import os
import json
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

def add_cors_headers(response):
    response.headers.set('Access-Control-Allow-Origin', '*')
    response.headers.set('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type')
    return response

@app.route('/api/index', methods=['POST', 'OPTIONS'])
def handler():
    if request.method == 'OPTIONS':
        return add_cors_headers(app.make_response(('', 204)))

    if request.method == 'POST':
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return add_cors_headers(jsonify({"error": "Server configuration error."})), 500
            
            data = request.get_json()
            prompt = data.get("prompt") # The entire detailed prompt comes from the frontend now
            
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini", # Using a powerful and reliable model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for the Fast Conversational Spanish program. You must follow all rules and formatting instructions exactly as provided in the prompt. Your final output should be the raw content (HTML or Markdown) as requested."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # THE FIX: Return the AI's content directly
            ai_content = completion.choices[0].message.content
            
            response = jsonify({"content": ai_content})
            response.status_code = 200

        except Exception as e:
            print(f"AN ERROR OCCURRED: {e}") 
            response = jsonify({"error": "An internal server error occurred.", "details": str(e)})
            response.status_code = 500
            
        return add_cors_headers(response)
