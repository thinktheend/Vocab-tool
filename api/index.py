import os
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

# --- PROMPT GENERATION RULES (The "Brain") ---

def build_vocab_prompt(payload):
    # This function builds the vocabulary prompt using your exact rules
    topic = payload.get('topic', 'N/A')
    level = payload.get('level', 'N/A')
    tmode = payload.get('tmode', 'default')
    tmin = payload.get('tmin', '0')
    tmax = payload.get('tmax', '0')
    vocab_include = payload.get('vocab_include', 'None')

    prompt = f"""
OUTPUT CONTRACT: You MUST respond with a single, complete HTML file with embedded styles. Do not include markdown, commentary, or any text outside of the HTML document. The final output MUST be ready to display directly in a browser.

STYLE RULES:
- Font: Calibri, size 10pt.
- Colors: English is bold blue (<span class="en">), Spanish is bold red (<span class="es">). Question answers use .ans-en and .ans-es for italics.
- Headings: <h1> for the main title, <h2> for sections, all centered.
- Tables: Use standard HTML tables (<table>) with borders and padding.
- No Asterisks: Use <span> tags exclusively for styling.

GENERATION LOGIC (NON-NEGOTIABLE):
1.  **Section Order**: Nouns (with subcategories), Verbs, Descriptive Words, Common Phrases, Common Questions. You MUST include all sections.
2.  **Word Counts by Level**: Adhere STRICTLY to the word counts for each section based on the selected Level.
    - Level 1: Nouns <= 35, Verbs <= 15, Descriptive <= 12, Questions <= 8, Phrases <= 12.
    - Level 2: Nouns 30-55, Verbs 12-20, Descriptive 10-18, Questions 7-12, Phrases 10-18.
    - Level 3: Nouns 50-75, Verbs 18-30, Descriptive 18-28, Questions 12-18, Phrases 15-25.
    - Level 4: Nouns >= 70, Verbs >= 25, Descriptive >= 25, Questions >= 18, Phrases >= 25.
    - Level 5: Nouns >= 90, Verbs >= 30, Descriptive >= 30, Questions >= 25, Phrases >= 35.
3.  **Total Words**: Honor the Total Min/Max. 'Total Words' counts ONLY Nouns + Verbs + Descriptive Words. Adjust section counts proportionally to meet the total.
4.  **Sentence Structure Rules**:
    - Nouns: Full sentences (e.g., "Here is the <span class='en'>airplane</span>."). Use natural variations like "This is..." if needed.
    - Verbs: Full sentences (e.g., "I am going to <span class='en'>fly</span>...").
    - Descriptive Words: Full sentences using ser/estar.
5.  **Spanish Gender Rule**: English side shows masculine only. Spanish side shows masculine, with feminine in parentheses if applicable.
6.  **Alphabetization**: ALL sections and subcategories MUST be alphabetized by the English term.
7.  **Labels**: Start each section with an in-table label row (e.g., <tr><td class="subcategory" colspan="2">Nouns / Sustantivos</td></tr>).

--- USER INPUT ---
Topic: {topic}
Level: {level}
Total Words Mode: {tmode}
Min Words: {tmin}
Max Words: {tmax}
Vocabulary to Include: {vocab_include}
"""
    return prompt

def build_convo_prompt(payload):
    # This function builds the conversation prompt using your exact rules
    topic = payload.get('topic', 'N/A')
    level = payload.get('level', 'N/A')
    vocab = payload.get('vocab', 'auto')
    # ... add all other payload keys here ...
    
    prompt = f"""
OUTPUT CONTRACT: You must respond with a single, complete HTML file with embedded styles. Do not include markdown or any commentary. Adhere to all rules from the FCS CONVERSATION CREATOR documentation exactly. Key Rules: Format conversations and practices in two columns. Bold and color vocabulary. Keep grammar strictly within the specified level. Use the ________ (ENGLISH) format for Fill-in-the-Blanks. The output MUST be ready to display directly in a browser.

--- USER INPUT ---
Topic: {topic}
Level: {level}
Vocabulary List: {vocab}
Num Convos: {payload.get('numConvos', '1')}
Num Speakers: {payload.get('numSpeakers', '2')}
Add FIB: {payload.get('addFib', 'yes')}
FIB Focus: {payload.get('fibFocus', 'Mixed')}
Tone: {payload.get('tone', 'Neutral, casual')}
"""
    return prompt

def build_test_prompt(payload):
    # This function builds the test prompt using your exact rules
    level = payload.get('level', 'N/A')
    vocab = payload.get('vocab', 'NONE')
    keys = payload.get('keys', True)
    
    prompt = f"""
OUTPUT CONTRACT: Create a 3-part Spanish test. Produce the output as a single HTML file with embedded styles. Do not include markdown or commentary. Adhere to all rules from the FCS TEST CREATOR documentation precisely. Key Rules: Return ONLY raw HTML. Structure MUST be Part 1 (ES->EN), Part 2 (EN->ES), Part 3 (Oral Prompts). Use only the provided vocabulary and level-specific grammar. Place answer keys at the very end if requested. The output MUST be ready to display directly in a browser.

--- USER INPUT ---
Level: {level}
Include Answer Keys: {keys}
Vocabulary List:
{vocab}
"""
    return prompt

# --- Main HTTP Handler (The "Engine") ---
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
            payload = json.loads(post_data) # This is the object from the frontend
            
            tool_type = payload.get("toolType")
            final_prompt = ""

            if tool_type == "vocabulary":
                final_prompt = build_vocab_prompt(payload)
            elif tool_type == "conversation":
                final_prompt = build_convo_prompt(payload)
            elif tool_type == "test":
                final_prompt = build_test_prompt(payload)
            else:
                raise ValueError("Invalid toolType specified.")

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Server configuration error: API key is missing.")

            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert assistant for the Fast Conversational Spanish program. Your task is to generate educational content. You must follow all rules provided by the user precisely. Your final output must be ONLY the raw content (HTML or Markdown) as requested, with absolutely no commentary or extra text."},
                    {"role": "user", "content": final_prompt}
                ]
            )
            
            ai_content = completion.choices[0].message.content
            
            # Clean the response
            if "```html" in ai_content:
                ai_content = ai_content.split("```html")[1].split("```")[0].strip()
            
            response_payload = {"content": ai_content}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            error_payload = {"error": "An internal server error occurred.", "details": str(e)}
            self.wfile.write(json.dumps(error_payload).encode('utf-8'))
