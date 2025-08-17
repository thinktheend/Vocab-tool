import os
import json
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# This function builds the prompt string based on the payload from the frontend
def build_prompt(payload):
    tool_type = payload.get("toolType")
    
    # Vocabulary Prompt Builder
    if tool_type == "vocabulary":
        topic = payload.get('topic', 'N/A')
        level = payload.get('level', 'N/A')
        tmode = payload.get('tmode', 'default')
        tmin = payload.get('tmin', '0')
        tmax = payload.get('tmax', '0')
        vocab_include = payload.get('vocab_include', 'None')
        return f"""OUTPUT CONTRACT: You MUST respond with a single, complete HTML file with embedded styles. Do not include markdown, commentary, or any text outside of the HTML document. Adhere to all rules from the FCS VOCABULARY CREATOR documentation exactly. Pay close attention to word counts, grammar for the specified level, and the required sentence structures. The output MUST be ready to display directly in a browser.\n\n--- USER INPUT ---\nTopic: {topic}\nLevel: {level}\nTotal Words Mode: {tmode}\nMin Words: {tmin}\nMax Words: {tmax}\nVocabulary to Include: {vocab_include}"""

    # Conversation Prompt Builder
    elif tool_type == "conversation":
        return f"""OUTPUT CONTRACT: You must respond with a single, complete HTML file with embedded styles. Adhere to all rules from the FCS CONVERSATION CREATOR documentation exactly. Key Rules: Format conversations and practices in two columns. Bold and color vocabulary. Keep grammar strictly within the specified level. Use the ________ (ENGLISH) format for Fill-in-the-Blanks.\n\n--- USER INPUT ---\nTopic: {payload.get('topic')}\nLevel: {payload.get('level')}\nVocabulary List: {payload.get('vocab') or "auto"}\nNum Convos: {payload.get('numConvos')}\nNum Speakers: {payload.get('numSpeakers')}\nAdd FIB: {payload.get('addFib')}\nFIB Focus: {payload.get('fibFocus')}\nTone: {payload.get('tone')}"""

    # Test Prompt Builder
    elif tool_type == "test":
        return f"""OUTPUT CONTRACT: Create a 3-part Spanish test as a single HTML file with styles. Adhere to all rules from the FCS TEST CREATOR documentation precisely. Key Rules: Return ONLY raw HTML. Structure MUST be Part 1 (ES->EN), Part 2 (EN->ES), Part 3 (Oral Prompts). Use only the provided vocabulary and level-specific grammar. Place answer keys at the very end if requested.\n\n--- USER INPUT ---\nLevel: {payload.get('level')}\nInclude Answer Keys: {payload.get('keys')}\nVocabulary List:\n{payload.get('vocab')}"""
    
    else:
        raise ValueError("Invalid toolType specified in payload.")


# Main Vercel Handler
@app.route('/api/index', methods=['POST', 'OPTIONS'])
def handler():
    headers = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'POST, OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type' }
    if request.method == 'OPTIONS':
        return ('', 204, headers)

    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return ({"error": "Server configuration error."}, 500, headers)
        
        # THE FIX: Get the 'prompt' object from the request, then get the toolType from inside it.
        data = request.get_json()
        payload = data.get("prompt")
        if not payload or 'toolType' not in payload:
            return ({"error": "Invalid request format."}, 400, headers)

        final_prompt = build_prompt(payload)
        
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert assistant for the Fast Conversational Spanish program. Follow all rules from the user. Your output must be ONLY the raw HTML content requested, with no extra text like ```html."},
                {"role": "user", "content": final_prompt}
            ]
        )
        ai_content = completion.choices[0].message.content
        return ({"content": ai_content}, 200, headers)

    except Exception as e:
        print(f"AN ERROR OCCURRED: {e}")
        return ({"error": "An internal server error occurred.", "details": str(e)}, 500, headers)
