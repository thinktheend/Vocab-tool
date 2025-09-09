# api/index.py
# Vercel-compatible serverless handler using BaseHTTPRequestHandler.
# Quantity enforcement preserved; UI/format unchanged.

import os
import re
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")

# Unwrap code fences if the provider adds them.
FENCE_RE = re.compile(r"^\s*```(?:html|xml|markdown)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE)

# Detect which tool is invoking us (based on the HTML prompt banner)
IS_VOCAB_RE = re.compile(r"FCS\s+VOCABULARY\s+OUTPUT", re.IGNORECASE)

# Parse "Vocabulary range: X–Y ..." from the embedded Markdown in the prompt.
RANGE_RE = re.compile(
    r"Vocabulary\s+range:\s*(\d+)\s*[\-\u2010-\u2015\u2212]\s*(\d+)",
    re.IGNORECASE,
)

TOPIC_RE = re.compile(r"<title>\s*Vocabulary\s*[—-]\s*(.*?)\s*</title>", re.IGNORECASE | re.DOTALL)

def parse_vocab_range(prompt_text: str):
    m = RANGE_RE.search(prompt_text or "")
    if not m:
        return (None, None)
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)

def parse_topic(prompt_text: str) -> str:
    m = TOPIC_RE.search(prompt_text or "")
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return "Topic"

def midpoint(lo: int, hi: int) -> int:
    return max(lo, min(hi, (lo + hi) // 2))

def quotas_30_30_15_15(total: int):
    """Return per-section targets (nouns, verbs, adjectives, adverbs) that sum to 'total'."""
    n = round(total * 0.30)
    v = round(total * 0.30)
    a = round(total * 0.15)
    d = round(total * 0.15)
    diff = total - (n + v + a + d)
    order = ["n", "v", "a", "d"]
    i = 0
    while diff != 0:
        tgt = order[i]
        if diff > 0:
            if tgt == "n": n += 1
            elif tgt == "v": v += 1
            elif tgt == "a": a += 1
            else: d += 1
            diff -= 1
        else:
            if tgt == "n" and n > 0: n -= 1; diff += 1
            elif tgt == "v" and v > 0: v -= 1; diff += 1
            elif tgt == "a" and a > 0: a -= 1; diff += 1
            elif tgt == "d" and d > 0: d -= 1; diff += 1
        i = (i + 1) % 4
    return n, v, a, d

def phrases_questions_row_targets(total_vocab_midpoint: int):
    """
    Compute required rows for Common Phrases and Common Questions.
    Per your requirement: ALWAYS generate them; never exceed 10.
    We target a compact band 8–10 to keep them 'enough' without exceeding 10.
    """
    rows = max(8, min(10, round(total_vocab_midpoint / 18)))  # yields 8–10 typically
    return rows, rows

# -----------------------
# New: Your long vocabulary prompt injected safely
# (interpreted as guidance; rendering remains your HTML skeleton)
# -----------------------
def build_user_guidance_prompt(topic: str, lo: int, hi: int) -> str:
    text = f"""You are an expert assistant for the FCS program.
 You must always follow every instruction below exactly.
 Never ask me follow-up questions, never stop early, and never skip or merge sections.
Topic: “{topic}”
 Vocabulary range: {lo}–{hi} distinct Spanish vocabulary words (target the upper bound).
0. Topic Title
At the very top, present the topic title in a two-column Markdown table row.
Format exactly like this (only once, at the very top):
Title | Título
 Topic in English | Topic in Spanish
1. Nouns
Present nouns in a two-column Markdown table.
The header row must have column-specific descriptions, for example:
 | Nouns | Sustantivos |
Use bold article + noun in both languages.
If more than 20 nouns, subdivide into logical categories such as People, Places, Equipment.
Only write “1. Nouns” once at the start of this section, not before every subcategory.
Subcategories must have a short label such as People, Places.
Alphabetize English terms within each category.
All nouns must be relevant to the topic.
After each subcategory’s table, insert one completely blank row with just spaces,
 then the new section title, then one blank line before the new table begins.
2. Verbs in Sentences
Present in a two-column Markdown table.
The header row must have column-specific descriptions, for example:
 | Verbs in Sentences | Verbos en oraciones |
Always use third-person sentences in English with “is/are going to + verb.”
Reflexive verbs must place the pronoun after the infinitive (example: va a descansarse).
Bold “to + verb” in English and the infinitive in Spanish.
Include sentences where:
nouns from Part 1 are objects
nouns from Part 1 are subjects
no nouns, just a third-person subject (he, she, it, they)
Verbs must be relevant to the topic.
End the table with one completely blank row with just spaces, then the new section title, then another blank line.
3. Adjectives
Present in a two-column Markdown table.
The header row must have column-specific descriptions, for example:
 | Adjectives | Adjetivos |
Sentences must pair nouns from Part 1 with “is/are + adjective.”
Only the adjective should be bold.
Each major noun must appear with at least two adjectives (contrasting or related).
End with the same spacing rule.
4. Adverbs
Present in a two-column Markdown table.
The header row must have column-specific descriptions, for example:
 | Adverbs | Adverbios |
Sentences must reuse verbs from Part 2, each modified with an adverb.
Only the adverb should be bold.
Adverbs must fit the context of the topic.
End with the same spacing rule.
5. Common Phrases
Present in a two-column Markdown table.
The header row must have column-specific descriptions, for example:
 | Common Phrases | Frases comunes |
Phrases must be relevant to the topic.
Write no more than 7 phrases.
End with the same spacing rule.
6. Common Questions
Present in a two-column Markdown table.
The header row must have column-specific descriptions, for example:
 | Common Questions | Preguntas comunes |
Do not include answers.
Write no more than 7 questions.
Questions must be relevant to the topic.
End with the same spacing rule.
7. Conversations
Create two sample conversations in a two-column table.
Each conversation is between two people.
Each conversation (1 & 2) must include 8 turns total (4 per person).
Each turn in each conversation (1 & 2) must contain 2–3 sentences/questions. (at least half of the turns include a question, and at least half of the turns include 3 sentences/questions
Each sentence in each turn in each conversation (1 & 2) must be 1–15 words long.
Conversations must include a natural mix of statements and questions.
Vary:
the number of sentences/questions per turn (sometimes 2, sometimes 3).
the length of sentences within a turn (some short, some longer).
Must use nouns, verbs, adjectives, and/or phrases from earlier sections.
Conversation 1 grammar rules: Do not use past tenses, subjunctive, or imperative.
Conversation 2 grammar rules: May use any grammar, verb tenses and moods.
Special formatting and spacing rules for Conversations:
After the line “7. Conversations”, insert one completely blank row.
Then start Conversation 1 with a header row inside the table, with column-specific titles:
Left column: (English) Who is speaking/about what.
Right column: (Español) Who is speaking/about what.
After Conversation 1’s table, insert two completely blank rows
Then Conversation 2 begins with its own header row inside the table, following the same column-specific titles.
After Conversation 2’s table, insert one completely blank row before the Monologue section.
8. Monologue
 Write a lecture-style monologue with 10–15 sentences.
Present in a two-column Markdown table.
It must be one paragraph, not separate lines.
It should feel like an explanation, guidance, or commentary about the topic.
Must use vocabulary from earlier sections.
Special formatting rule for Monologue:
The header row of the table should contain column-specific titles:
Left column: “Monologue: [English description of topic]”
Right column: “Monólogo: [Spanish description of topic]”
⚠ Both columns must contain a complete monologue: the left side in English and the right side in Spanish.
Final Rules
Use Markdown tables for all parts (0–8).
Every section must include a header row with column-specific descriptions, not just raw content.
After every table or subsection, always insert:
one completely blank row (just spaces),
then the new section title,
then one blank line before the next table.
Always produce the full 8 parts (title → monologue).
Never stop early. Never ask me if you should continue.
Always keep vocabulary count between 200–250 distinct Spanish words.
"""
    # IMPORTANT: We instruct the model (in the system message below) to
    # use these structure ideas but RENDER ONLY the HTML skeleton sections,
    # and to IGNORE sections 7 & 8 visually.
    return text

def build_system_message(base_system: str, user_prompt: str) -> str:
    """
    If this is a Vocabulary prompt and we can read the range, append a STRICT contract
    that forces midpoint counts while preserving the front-end skeleton, and require
    Common Phrases/Questions (≤10 each). Also enforce color rules.
    """
    if not IS_VOCAB_RE.search(user_prompt or ""):
        return base_system  # Conversation/Test: unchanged

    lo, hi = parse_vocab_range(user_prompt)
    if lo is None or hi is None:
        return base_system

    topic = parse_topic(user_prompt)
    target_total = midpoint(lo, hi)
    n, v, a, d = quotas_30_30_15_15(target_total)
    phrases_min, questions_min = phrases_questions_row_targets(target_total)
    max_reuse = max(1, (target_total * 20 + 99) // 100)  # ceil(20% of total)

    # Guidance prompt from user request (used as constraints, but we keep rendering in HTML)
    guidance = build_user_guidance_prompt(topic, lo, hi)

    contract = f"""

STRICT ONE-SHOT COUNTING CONTRACT (Vocabulary ONLY; do NOT change UI/format):
• TARGET TOTAL (sections 1–4 only): EXACTLY {target_total} Spanish vocabulary items counted by
  the number of <span class="es">…</span> target words in Nouns, Verbs, Adjectives, Adverbs.
• PER-SECTION QUOTAS (enforce exactly):
  – Nouns: {n}
  – Verbs: {v}
  – Adjectives: {a}
  – Adverbs: {d}
• COMMON PHRASES & COMMON QUESTIONS — MANDATORY:
  – Populate BOTH sections with table rows inside their existing <tbody>.
  – Number of rows in EACH section: between {max(8, phrases_min)} and 10 inclusive (NEVER exceed 10).
  – Reuse only vocabulary from sections 1–4 (no new vocabulary). Distinct reused words
    across BOTH sections combined must be ≤ {max_reuse} (≈20% of {target_total}).
  – Rows in these sections do NOT count toward the {target_total} total.
• COLORING & LINGUISTICS:
  – Verbs: English cell must color the “to + verb/particle” portion with <span class="en">…</span>;
    Spanish cell must color ONLY the infinitive with <span class="es">…</span>. NEVER color “voy/vas/va/vamos/vais/van a”.
  – Nouns: Spanish cell uses article; IF a noun commonly has both genders, show masculine first
    and append the feminine in parentheses, e.g., el médico (la médica), el cliente (la cliente).
  – Adjectives: sentences with “is/are + adjective”; highlight ONLY the adjective.
  – Adverbs: sentences that reuse verbs; highlight ONLY the adverb.
• RENDERING BOUNDARIES — CRITICAL:
  – You MUST use the HTML skeleton from the user's prompt AS-IS (no Markdown, no new sections).
  – IGNORE the visual rendering of the user guidance's parts 0, 7, and 8. ONLY populate:
    Nouns; Verbs in Sentences; Adjectives; Adverbs; Common Phrases; Common Questions.
  – Insert ONLY <tr> row content into each existing <tbody>. Do NOT add extra tables or headers.
• SELF-CHECK BEFORE SENDING:
  – Ensure exact per-section quotas and grand total in sections 1–4.
  – Ensure BOTH Common sections exist and have 8–10 rows each (≤10).
  – Ensure well-formed HTML that fits the provided skeleton.

# REFERENCE GUIDANCE FROM USER (structure only — still render into provided HTML):
{guidance}
"""
    return base_system + contract


# -----------------------
# Post-processing helpers
# -----------------------

_VOWEL_MAP = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")

def derive_feminine(base: str) -> str:
    """Simple heuristic for feminine forms."""
    w = base.strip()
    if not w:
        return base
    raw = w.translate(_VOWEL_MAP)  # strip acute accents for transforms
    lower = raw.lower()

    if lower.endswith("o"):
        return raw[:-1] + "a"
    if lower.endswith(("or",)):
        return raw + "a"
    if lower.endswith(("on", "in", "an")):  # campeón -> campeona; capitán -> capitana
        return raw + "a"
    if lower.endswith(("ista", "ante", "ente", "e")):
        return raw  # invariant (still shown in parentheses with la …)
    return raw

def _replace_in_section(html: str, section_title_regex: str, replacer) -> str:
    m = re.search(rf'(<h2>\s*{section_title_regex}\s*</h2>)(.*?)(</div>)',
                  html, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return html
    head, body, tail = m.group(1), m.group(2), m.group(3)
    new_body = replacer(body)
    return html.replace(m.group(0), f"{head}{new_body}{tail}", 1)

def fix_verbs_highlight(body_html: str) -> str:
    """
    In Verbs section tbody: ensure only the Spanish infinitive is red (not 'va a'),
    and ensure English 'to + verb/particle' is blue.
    """
    def fix_one_tbody(tb):
        s = tb

        # ---- Spanish: move highlight to infinitive (never 'va a')
        s = re.sub(
            r'<span\s+class="es">([^<]*?)\b(voy|vas|va|vamos|vais|van)\s+a\s+([a-záéíóúüñ/]+)\b([^<]*?)</span>',
            r'\1\2 a <span class="es">\3</span>\4',
            s, flags=re.IGNORECASE
        )
        s = re.sub(
            r'<span\s+class="es">\s*(voy|vas|va|vamos|vais|van)\s+a\s+([a-záéíóúüñ/]+)\s*</span>',
            r'\1 a <span class="es">\2</span>',
            s, flags=re.IGNORECASE
        )
        s = re.sub(
            r'<span\s+class="es">\s*(voy|vas|va|vamos|vais|van)\s*</span>\s*a\s+([a-záéíóúüñ/]+)',
            r'\1 a <span class="es">\2</span>',
            s, flags=re.IGNORECASE
        )
        s = re.sub(
            r'<span\s+class="es">\s*va\s*a\s*</span>\s*([a-záéíóúüñ/]+)',
            r'va a <span class="es">\1</span>',
            s, flags=re.IGNORECASE
        )

        # ---- English: ensure “to + verb(/particle)” has <span class="en"> on the verb/particle
        def en_cell_fix(match):
            cell = match.group(0)
            if 'class="en"' in cell:
                return cell  # already colored somewhere; keep as-is

            # Find "to <verb or phrasal verb>" and wrap ONLY the verb phrase in <span class="en">
            def wrap_to_phrase(mv):
                verb_phrase = mv.group(2)
                return f'{mv.group(1)}to <span class="en">{verb_phrase}</span>'

            cell2 = re.sub(r'(to\s+)([a-z-]+(?:\s+(?:up|down|in|on|off|out|over|back|away))?)',
                           wrap_to_phrase, cell, flags=re.IGNORECASE)
            return cell2

        # apply English fix only to the first column <td> of each row
        def per_row(row_m):
            row = row_m.group(0)
            tds = list(re.finditer(r'<td[^>]*>.*?</td>', row, flags=re.IGNORECASE | re.DOTALL))
            if len(tds) >= 1:
                first_td = tds[0]
                start, end = first_td.span()
                fixed_first = en_cell_fix(first_td)
                row = row[:start] + fixed_first + row[end:]
            return row

        s = re.sub(r'<tr[^>]*>.*?</tr>', per_row, s, flags=re.IGNORECASE | re.DOTALL)
        return s

    def repl(section_html):
        return re.sub(
            r'(<tbody[^>]*>)(.*?)(</tbody>)',
            lambda m: f'{m.group(1)}{fix_one_tbody(m.group(2))}{m.group(3)}',
            section_html, flags=re.IGNORECASE | re.DOTALL
        )

    return _replace_in_section(body_html, r'Verbs\s+in\s+Sentences', repl)

def add_feminine_in_nouns(body_html: str) -> str:
    """
    Inside Nouns section tbody, append feminine in parentheses when cell is
    just 'el X' and no parentheses exist yet.
    """
    def fix_one_tbody(tb):
        s = tb

        def add_fem(m):
            word = m.group(2)
            fem = derive_feminine(word)
            return f'{m.group(1)}{word}</span> (la <span class="es">{fem}</span>)'

        def per_td(td_match):
            td = td_match.group(0)
            if "(" in td:  # already has parentheses of some sort
                return td
            td2 = re.sub(
                r'(<span\s+class="es">\s*el\s+)([a-záéíóúüñ]+)\s*</span>',
                add_fem,
                td, flags=re.IGNORECASE
            )
            return td2

        s = re.sub(
            r'<td[^>]*>.*?</td>',
            per_td,
            s, flags=re.IGNORECASE | re.DOTALL
        )
        return s

    def repl(section_html):
        return re.sub(
            r'(<tbody[^>]*>)(.*?)(</tbody>)',
            lambda m: f'{m.group(1)}{fix_one_tbody(m.group(2))}{m.group(3)}',
            section_html, flags=re.IGNORECASE | re.DOTALL
        )

    return _replace_in_section(body_html, r'Nouns', repl)


class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length else b"{}"

            try:
                data = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in request body.")

            prompt = (data.get("prompt") or "").strip()
            if not prompt:
                raise ValueError("Missing 'prompt' in request body.")

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Server configuration error: OPENAI_API_KEY is not set.")

            client = OpenAI(
                api_key=api_key,
                base_url=OPENAI_BASE_URL or None,
                organization=OPENAI_ORG_ID or None,
            )

            # Larger budget to avoid truncation under strict length rules.
            max_tokens = min(int(os.getenv("MODEL_MAX_TOKENS", "7000")), 16384)

            # Base system message = your last known-good text (unchanged)
            base_system = (
                "You are an expert FCS assistant. Return ONLY full raw HTML (a valid document). "
                "Strictly follow the embedded contract inside the user's HTML prompt. "
                "ABSOLUTE LENGTH COMPLIANCE: When ranges are provided (counts or sentences/words), "
                "produce at least the minimum and not more than the maximum. Do not under-deliver. "
                "If needed, compress prose while keeping counts intact. "
                "Vocabulary generator rules (do not change UI/format): "
                "• NOUNS: words/phrases only (no sentences) with subcategory header rows when required; "
                "  the Spanish noun is wrapped in <span class=\"es\">…</span> (red). "
                "• VERBS: full sentences using He/She/It/They + is/are going to + [infinitive]; "
                "  highlight ONLY the verb (one <span class=\"en\">…</span> in the English cell, "
                "  one <span class=\"es\">…</span> in the Spanish cell). "
                "• ADJECTIVES: full sentences with is/are + adjective; highlight ONLY the adjective "
                "  (one <span class=\"en\">…</span> and one <span class=\"es\">…</span>). "
                "• ADVERBS: full sentences that reuse verbs, highlight ONLY the adverb "
                "  (one <span class=\"en\">…</span> and one <span class=\"es\">…</span>). "
                "• FIB (when present): English cell colors ONLY the target English word with <span class=\"en\">…</span>; "
                "  Spanish cell replaces the target Spanish word with its English translation in parentheses (no blank line). "
                "Common Phrases/Questions must follow the contract. "
                "Do NOT add explanations or code fences."
            )

            # Add the vocabulary count contract (only when the prompt is the Vocab generator)
            system_message = build_system_message(base_system, prompt)

            completion = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                temperature=0.8,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
            )

            ai_content = (completion.choices[0].message.content or "").strip()

            # Unwrap code fences if present
            m = FENCE_RE.match(ai_content)
            if m:
                ai_content = m.group(1).strip()

            # --- Post-processing fixes (format preserved) ---
            # 1) Ensure verbs highlighting rules (ES infinitive only; EN "to + verb" blue)
            ai_content = fix_verbs_highlight(ai_content)
            # 2) Append feminine forms in nouns where applicable
            ai_content = add_feminine_in_nouns(ai_content)

            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"content": ai_content}).encode("utf-8"))

        except Exception as e:
            print(f"AN ERROR OCCURRED: {e}")
            self.send_response(500)
            self._send_cors_headers()
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "An internal server error occurred.",
                "details": str(e)
            }).encode("utf-8"))
