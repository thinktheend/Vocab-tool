import os
import re
import openai
from flask import Flask, request, render_template

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuration: preserve any existing settings for model and quotas
MODEL = "gpt-4"  # Using GPT-4o for generation (unchanged)
# (Assume other configuration variables like section quotas, prompts, etc. are defined above)

@app.route('/generate', methods=['POST'])
def generate_vocabulary():
    # Get user inputs from form (e.g., topic and word count range)
    topic = request.form.get('topic')
    min_words = int(request.form.get('min_words', 0))
    max_words = int(request.form.get('max_words', 0))
    if min_words <= 0 or max_words < min_words:
        return "Invalid word count range", 400

    # Determine the target number of vocabulary words as the midpoint of the range (quantity enforcement unchanged)
    total_target = (min_words + max_words) // 2
    # Preserve section quotas as per original logic (no changes made here)
    # For example, if quotas are defined as a percentage or fixed counts per section:
    # nouns_target = int(total_target * NOUNS_RATIO)
    # verbs_target = int(total_target * VERBS_RATIO)
    # adj_target = ...
    # adv_target = ...
    # (The exact quota logic is assumed to be defined elsewhere or above)
    nouns_target = ...  # (placeholder for original logic)
    verbs_target = ...  # (placeholder for original logic)
    adjs_target = ...   # (placeholder for original logic)
    advs_target = ...   # (placeholder for original logic)

    # Prepare prompts for GPT-4o for each section (not changed)
    # For example:
    prompt_nouns = f"List {nouns_target} useful Spanish nouns (with articles) for the topic '{topic}', with English translations."
    prompt_verbs = f"List {verbs_target} useful Spanish verbs (in context sentences) for the topic '{topic}', with English translations."
    # ... similarly for adjectives, adverbs, etc.
    # (Exact prompt phrasing as originally used is assumed to be retained)

    # Call GPT-4o to generate each section (using the same logic as before)
    try:
        nouns_response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt_nouns}]
        )
        verbs_response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt_verbs}]
        )
        # ... (calls for adjectives, adverbs, common phrases, questions as needed)
    except openai.Error as e:
        return f"Error generating content: {e}", 500

    # Extract the content from GPT responses (assuming GPT returns a formatted list or lines)
    nouns_content = nouns_response['choices'][0]['message']['content']
    verbs_content = verbs_response['choices'][0]['message']['content']
    # ... (similarly for other sections)

    # Parse or split the GPT output into lists of (English, Spanish) pairs for each section.
    # Assuming GPT output each item on a new line in "English - Spanish" format or similar.
    nouns_pairs = []
    for line in nouns_content.splitlines():
        # e.g., line format: "The doctor - el médico"
        if not line.strip():
            continue
        parts = re.split(r'\s*-\s*', line, maxsplit=1)
        if len(parts) == 2:
            eng, spa = parts[0].strip(), parts[1].strip()
            nouns_pairs.append((eng, spa))
    verbs_pairs = []
    for line in verbs_content.splitlines():
        # e.g., line format: "He is going to check in his luggage. - Él va a facturar su equipaje."
        if not line.strip():
            continue
        parts = re.split(r'\s*-\s*', line, maxsplit=1)
        if len(parts) == 2:
            eng, spa = parts[0].strip(), parts[1].strip()
            verbs_pairs.append((eng, spa))
    # ... (similar parsing for adjectives, adverbs, etc.)

    # Now build the HTML output, section by section.
    output_html = ""

    # Nouns Section
    if nouns_pairs:
        output_html += f"<h2>Nouns ({len(nouns_pairs)})</h2>\n<table>\n"
        for eng, spa in nouns_pairs:
            # >>> If the noun has a gender counterpart, combine forms
            if spa.lower().startswith("el "):
                # Derive feminine form of the noun
                masculine_article = "el"
                noun_base = spa[len("el "):]  # strip "el " prefix
                fem_article = "la"
                fem_base = noun_base  # default assume same base
                if noun_base.endswith("o"):
                    fem_base = noun_base[:-1] + "a"
                elif noun_base.endswith(("or", "ón", "ín", "án")):
                    # Add 'a' to form feminine, and remove accent if present on penultimate syllable
                    # e.g., capitán -> capitana, león -> leona, profesor -> profesora
                    # Remove accent from the base if it has one (to handle words like capitán, francés, etc.)
                    # This removes any acute accent marks in the word
                    fem_base = re.sub(r'á', 'a', noun_base)
                    fem_base = re.sub(r'é', 'e', fem_base)
                    fem_base = re.sub(r'í', 'i', fem_base)
                    fem_base = re.sub(r'ó', 'o', fem_base)
                    fem_base = re.sub(r'ú', 'u', fem_base)
                    fem_base += "a"
                elif noun_base.endswith(("ista", "ante", "ente")) or noun_base.endswith("e"):
                    # Invariant or neutral-ending noun (use same base for feminine)
                    fem_base = noun_base
                else:
                    # Default: just add 'a'
                    fem_base = noun_base + "a"
                spa_display = f'<span class="es">{masculine_article} {noun_base}</span> ({fem_article} <span class="es">{fem_base}</span>)'
            elif spa.lower().startswith("la "):
                # If we have a feminine form given and no masculine present, derive masculine
                fem_article = "la"
                noun_base = spa[len("la "):]
                masculine_article = "el"
                masc_base = noun_base
                if noun_base.endswith("a") and not noun_base.endswith(("ista", "ta")):
                    # Replace final 'a' with 'o' for masculine (for common cases like médica -> médico)
                    masc_base = noun_base[:-1] + "o"
                elif noun_base.endswith("ora") and noun_base[:-1].endswith("or"):
                    # If feminine ends in 'ora' (like enfermera), masculine ends in 'or'
                    masc_base = noun_base[:-1]  # remove the trailing 'a', e.g., "enfermera" -> "enfermer"
                elif noun_base.endswith(("ista", "ante", "ente")) or noun_base.endswith("e"):
                    masc_base = noun_base  # same form for masculine
                else:
                    masc_base = noun_base  # default to same if unsure
                # Remove any accent in masculine base if present before adding article (similar accent handling as above)
                masc_base = re.sub(r'á', 'a', masc_base)
                masc_base = re.sub(r'é', 'e', masc_base)
                masc_base = re.sub(r'í', 'i', masc_base)
                masc_base = re.sub(r'ó', 'o', masc_base)
                masc_base = re.sub(r'ú', 'u', masc_base)
                spa_display = f'<span class="es">{masculine_article} {masc_base}</span> ({fem_article} <span class="es">{noun_base}</span>)'
            else:
                # No gendered article detected, just output as is (e.g., plural or no article given)
                spa_display = f'<span class="es">{spa}</span>'
            output_html += f"<tr><td>{eng}</td><td>{spa_display}</td></tr>\n"
        output_html += "</table>\n"

    # Verbs Section (with example sentences)
    if verbs_pairs:
        output_html += f"<h2>Verbs in Sentences ({len(verbs_pairs)})</h2>\n<table>\n"
        for eng, spa in verbs_pairs:
            # >>> Highlight only the main verb in Spanish (exclude 'ir a' auxiliary from the red span)
            spa_display = spa
            # Regex to find patterns like "___ a <verb>..." at start (covers "voy/vas/va/vamos/vais/van a ")
            match = re.match(r'^((?:[Vv]oy|[Vv]as|[Vv]a|[Vv]amos|[Vv]ais|[Vv]an) a )(.+)$', spa)
            if match:
                prefix = match.group(1)    # e.g. "Él va a " or "Ella va a " (including pronoun if present)
                main_verb_phrase = match.group(2)  # the rest after "a "
                # If the sentence starts with a pronoun like "Él" or "Ella", ensure it stays outside the span as well.
                # We include everything up to and including "a " in prefix.
                spa_display = f'{prefix}<span class="es">{main_verb_phrase}</span>'
            else:
                # If no "ir a" construction, highlight the whole Spanish phrase as before
                spa_display = f'<span class="es">{spa}</span>'
            output_html += f"<tr><td>{eng}</td><td>{spa_display}</td></tr>\n"
        output_html += "</table>\n"

    # ... (Similarly handle Adjectives, Adverbs, Common Phrases, Common Questions sections, unchanged in logic)
    # For brevity, those sections are not fully expanded here, but they would follow the same pattern:
    # output_html += f"<h2>Adjectives ({len(adjs_pairs)})</h2>..." and use <span class="es"> for Spanish parts as originally.

    # Finally, render the result in an HTML template or return directly
    return render_template('vocab_output.html', content=output_html)
