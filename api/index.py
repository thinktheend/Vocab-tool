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

# Note: Vercel routes requests to this file automatically.
# We just need one handler function.
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
            prompt_payload = data.get("prompt") # The entire JSON object from the frontend
            
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                # Note: No response_format needed if the model is instructed correctly.
                messages=[
                    {"role": "system", "content": "You are an assistant for the Fast Conversational Spanish program. The user will provide a detailed JSON prompt. You must generate the requested content as raw Markdown or HTML, as specified in the user's prompt. Your entire response should be a single JSON object with ONE key: 'result', and the value should be the generated content as a string."},
                    {"role": "user", "content": json.dumps(prompt_payload)}
                ]
            )
            
            ai_json_string = completion.choices[0].message.content
            ai_response_dict = json.loads(ai_json_string)
            response = jsonify(ai_response_dict)
            response.status_code = 200

        except Exception as e:
            print(f"AN ERROR OCCURRED: {e}") 
            response = jsonify({"error": "An internal server error occurred.", "details": str(e)})
            response.status_code = 500
            
        return add_cors_headers(response)
