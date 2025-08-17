import os
import json
from flask import Flask, request
from openai import OpenAI

# This is the standard entry point for a Vercel serverless function
app = Flask(__name__)

# The function Vercel will call is the Flask app itself
@app.route('/', defaults={'path': ''}, methods=['POST', 'OPTIONS'])
@app.route('/<path:path>', methods=['POST', 'OPTIONS'])
def handler(path):
    # Set up headers for the browser's security check (CORS)
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
    }

    if request.method == 'OPTIONS':
        return ('', 204, headers)

    if request.method == 'POST':
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return ({"error": "Server configuration error."}, 500, headers)
            
            data = request.get_json()
            prompt = data.get("prompt") # The entire payload is the prompt now
            
            # Reconstruct the text prompt for the AI from the payload
            prompt_data = data.get("prompt", {})
            prompt_text = f"Tool: {prompt_data.get('toolType')}\n"
            prompt_text += f"Topic: {prompt_data.get('topic')}\n"
            prompt_text += f"Level: {prompt_data.get('level')}\n"

            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                # The AI must now return a JSON object with a 'content' key
                response_format={"type": "json_object"}, 
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for the Fast Conversational Spanish program. You MUST respond with a valid JSON object with a single key 'content', where the value is the markdown text result."},
                    {"role": "user", "content": json.dumps(prompt_payload)}
                ]
            )
            
            # Get the raw text, which is a JSON string
            ai_json_string = completion.choices[0].message.content
            # Parse it into a Python dictionary
            ai_response_dict = json.loads(ai_json_string)

            # Return the dictionary as a JSON response
            return (ai_response_dict, 200, headers)

        except Exception as e:
            print(f"AN ERROR OCCURRED: {e}") 
            return ({"error": "An internal server error occurred.", "details": str(e)}, 500, headers)

    return ({"error": "Method not allowed."}, 405, headers)
