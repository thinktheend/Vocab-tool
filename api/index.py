import os
import openai
from flask import Flask, request, Response

# Configure OpenAI API (ensure API key is set in environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
# (If session usage is needed for conversation, a secret key would be set, not shown for brevity)

def generate_vocabulary_list(topic: str, min_words: int, max_words: int, include_words_text: str = "",
                              include_nouns: bool = True, include_verbs: bool = True,
                              include_adjectives: bool = True, include_adverbs: bool = True,
                              include_phrases: bool = True, include_questions: bool = True) -> str:
    """Generate a formatted Spanish vocabulary list (HTML string) for the given topic and settings."""
    # 1. Determine the exact midpoint of the range for total vocabulary count
    min_val = int(min_words)
    max_val = int(max_words)
    if min_val > max_val:
        # Swap if inputs were inverted
        min_val, max_val = max_val, min_val
    total_vocab = (min_val + max_val) // 2  # use floor if not an integer midpoint
    
    # 2. Calculate how many words to allocate to each selected section (using weighted distribution)
    # Weights: Nouns=4, Verbs=4, Adjectives=1, Adverbs=1 (only include if that section is enabled)
    section_weights = []
    if include_nouns:
        section_weights.append(("Nouns", 4))
    if include_verbs:
        section_weights.append(("Verbs", 4))
    if include_adjectives:
        section_weights.append(("Adjectives", 1))
    if include_adverbs:
        section_weights.append(("Adverbs", 1))
    if not section_weights:
        raise ValueError("No vocabulary sections selected")  # at least one section should be checked
    
    total_weight = sum(w for _, w in section_weights)
    # Determine counts per section such that sum equals total_vocab
    section_counts = {}
    allocated = 0
    for i, (section, weight) in enumerate(section_weights):
        if i < len(section_weights) - 1:
            # integer division for each, last section gets the remainder
            count = (total_vocab * weight) // total_weight
            section_counts[section] = count
            allocated += count
        else:
            section_counts[section] = total_vocab - allocated  # remainder to last section
    
    # 3. Build the prompt with all instructions and constraints
    prompt_lines = []
    prompt_lines.append(f"Topic: {topic}")
    prompt_lines.append(f"Total unique Spanish vocabulary words (across all sections): {total_vocab}.")
    prompt_lines.append("Sections and word counts:")
    for section, weight in section_weights:
        count = section_counts.get(section, 0)
        if section == "Nouns":
            prompt_lines.append(f"- {section}: {count} nouns, listed as English – Spanish pairs.")
        elif section == "Verbs":
            prompt_lines.append(f"- {section}: {count} verbs, presented in example sentences (English sentence with Spanish translation).")
        elif section == "Adjectives":
            prompt_lines.append(f"- {section}: {count} adjectives, demonstrated in example sentences (English sentence with Spanish translation).")
        elif section == "Adverbs":
            prompt_lines.append(f"- {section}: {count} adverbs, demonstrated in example sentences (English sentence with Spanish translation).")
    if include_phrases:
        prompt_lines.append("- Common Phrases: include a few useful phrases (e.g. ~5) relevant to the topic, with English and Spanish.")
    if include_questions:
        prompt_lines.append("- Common Questions: include a few useful questions (e.g. ~5) someone might ask for this topic, with English and Spanish.")
    prompt_lines.append("Formatting and requirements:")
    prompt_lines.append("* Provide the vocabulary list as an HTML <table> with two columns: English and Español.")
    prompt_lines.append("* Begin the table with a header row: <th>English</th> and <th>Español</th>.")
    prompt_lines.append("* Within the table, group words by section. For each section, first output a full-width row as a section header with the section name and the number of items, e.g. 'Nouns (" + str(section_counts.get('Nouns',0)) + ")'.")
    prompt_lines.append("* Under each section header, list each vocabulary item on its own table row. Column 1: English word or phrase; Column 2: Spanish translation.")
    prompt_lines.append("* **Nouns:** Include the definite article with each Spanish noun ('el' or 'la'), and if a noun has a feminine form, show it in parentheses. Example: The receptionist – <span class=\"es\">El recepcionista (la recepcionista)</span>.")
    prompt_lines.append("* **Verbs, Adjectives, Adverbs:** Do NOT list as isolated words. Instead, present each in a meaningful example sentence. In the table, the English column should have an English sentence and the Spanish column the Spanish translation. **Highlight the Spanish vocabulary word** in each sentence by wrapping it in `<span class=\"es\">...</span>`.")
    prompt_lines.append("* Ensure **all Spanish vocabulary words** (the new words to learn) are wrapped in `<span class=\"es\">...</span>` in the table. Do not use the span for common words that are not part of the vocabulary list.")
    prompt_lines.append(f"* The total number of distinct Spanish words in `<span class='es'>` across all sections must be exactly **{total_vocab}**. **No duplicates**: do not repeat any Spanish vocabulary word in more than one section.")
    if include_words_text:
        # Include any user-specified vocabulary words
        # Split by commas/newlines and clean
        import re
        user_words = [w.strip() for w in re.split(r'[\n,;]+', include_words_text) if w.strip()]
        if user_words:
            prompt_lines.append("* Include **all** of the following specific words in the list (use the appropriate English/Spanish form and put in the correct section): " + ", ".join(user_words) + ". These count toward the total and should not be duplicated across sections.")
    prompt_lines.append("* **Common Phrases / Questions (if included):** After the table, output these sections with each phrase or question on a separate line (English sentence, then Spanish translation on the next line). If a Spanish translation contains a vocabulary word from the list, wrap that word in `<span class=\"es\">...</span>`. Do not highlight words that are not in the main list.")
    prompt_lines.append("* Use at most ~20% of the vocabulary words in the phrases/questions (limited reuse). Each vocabulary word should appear at most once in the Phrases section and at most once in the Questions section.")
    prompt_lines.append("* Output nothing except the formatted vocabulary list, phrases, and questions. **No explanations, no additional commentary.**")
    full_prompt = "\n".join(prompt_lines)
    
    # 4. Call the OpenAI GPT-4 API with the constructed prompt
    messages = [
        {"role": "system", "content": "You are a Spanish vocabulary list generator. Follow the format and requirements strictly."},
        {"role": "user", "content": full_prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.8  # user-defined decoding parameter
        # (no frequency_penalty or presence_penalty specified as per instructions)
    )
    generated_text = response['choices'][0]['message']['content']
    
    # 5. Prepend the topic header and return the complete HTML content
    final_html = f"<h2>Vocabulary — {topic}</h2>\n" + generated_text
    return final_html

# Route for generating vocabulary (POST)
@app.route("/generate_vocabulary", methods=["POST"])
def generate_vocabulary():
    # Extract form inputs
    topic = request.form.get("topic", "").strip()
    min_range = request.form.get("min", "").strip() or request.form.get("min_words", "").strip()
    max_range = request.form.get("max", "").strip() or request.form.get("max_words", "").strip()
    include_words_text = request.form.get("include_words", "").strip()
    # Check which sections are selected
    include_nouns = bool(request.form.get("nouns"))
    include_verbs = bool(request.form.get("verbs"))
    include_adjectives = bool(request.form.get("adjectives"))
    include_adverbs = bool(request.form.get("adverbs"))
    include_phrases = bool(request.form.get("phrases")) or bool(request.form.get("common_phrases"))
    include_questions = bool(request.form.get("questions")) or bool(request.form.get("common_questions"))
    # Generate the vocabulary list HTML
    try:
        result_html = generate_vocabulary_list(topic, min_range, max_range, include_words_text,
                                               include_nouns, include_verbs,
                                               include_adjectives, include_adverbs,
                                               include_phrases, include_questions)
    except Exception as e:
        # Return error as HTTP 400
        return Response(f"Error generating vocabulary: {e}", status=400)
    # Return the generated HTML content
    return Response(result_html, mimetype="text/html")

# Conversation tab logic (unchanged)
conversation_history = {}  # simple in-memory store: per-session chat history

@app.route("/conversation", methods=["POST"])
def conversation():
    user_message = request.form.get("message", "").strip()
    session_id = request.form.get("session_id", "default")
    if not user_message:
        return Response("No message provided", status=400)
    # Initialize history for this session if not exists
    if session_id not in conversation_history:
        conversation_history[session_id] = []
        # (Optional: can add a system message or persona initialization here if needed)
    history = conversation_history[session_id]
    # Append user message
    history.append({"role": "user", "content": user_message})
    try:
        # Use the same model as before (assuming GPT-3.5 for conversation unless specified otherwise)
        chat_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=history,
            temperature=0.8
        )
    except Exception as e:
        return Response(f"Error in conversation: {e}", status=500)
    assistant_reply = chat_response['choices'][0]['message']['content']
    # Append assistant reply to history
    history.append({"role": "assistant", "content": assistant_reply})
    return Response(assistant_reply, mimetype="text/plain")

# ... (Any other routes or logic in the original file would remain here, unchanged) ...

if __name__ == "__main__":
    app.run(debug=True)
