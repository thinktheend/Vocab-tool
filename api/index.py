import os
import json
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# --- Reusable function to add headers for Kajabi ---
def add_cors_headers(response):
    response.headers.set('Access-Control-Allow-Origin', '*')
    response.headers.set('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type')
    return response

@app.route('/', defaults={'path': ''}, methods=['POST', 'OPTIONS'])
@app.route('/<path:path>', methods=['POST', 'OPTIONS'])
def handle_request(path):
    # Handle the browser's "preflight" request for security
    if request.method == 'OPTIONS':
        return add_cors_headers(app.make_response(('', 204)))

    # Handle the actual data request
    if request.method == 'POST':
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                # Security: Never reveal the key is missing, just a config error
                return add_cors_headers(jsonify({"error": "Server configuration error."})), 500
            
            data = request.get_json()
            prompt = data.get("prompt")
            
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini", # Using a powerful and reliable model
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for the Fast Conversational Spanish program. You must always respond with a valid JSON object containing the requested content. Do not include any extra commentary or text outside of the JSON object itself."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # THE FIX: Properly unwrap the AI's response to get the clean text
            ai_content_text = completion.choices[0].message.content
            
            # The AI might return JSON as a string, so we parse it to be safe
            ai_response_json = json.loads(ai_content_text)

            # Send back the clean, unwrapped JSON object
            response = jsonify(ai_response_json)
            response.status_code = 200

        except Exception as e:
            # This helps in debugging if something else goes wrong
            print(f"AN ERROR OCCURRED: {e}") 
            response = jsonify({"error": "An internal server error occurred.", "details": str(e)})
            response.status_code = 500
            
        return add_cors_headers(response)
