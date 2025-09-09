# api/index.py
# Vercel-compatible serverless handler using BaseHTTPRequestHandler.
# Quantity enforcement preserved; UI/format unchanged.
# Hardened color normalization so N/V/A/D rows reliably contribute to the ES span counts.

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
    """Target compact 8–10 rows for both Common sections; never exceed 10."""
    rows = max(8, min(10, round(total_vocab_midpoint / 18)))
    return rows, rows

# -----------------------
# Guidance (verbatim block you provided)
# -----------------------
def build_user_guidance_prompt(topic: str, lo: int, hi: int) -> str:
    return f"""You are an expert assistant for the FCS program.
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

def build_system_message(base_system: str, user_prompt: str) -> str:
    """
    Vocabulary prompt: add strict contract enforcing:
      - midpoint quotas for sections 1–4
      - Common Phrases & Questions present (8–10 rows each, ≤10)
      - color rules + feminine parenthetical handling
      - quotas/range from UI override any conflicting guidance
    """
    if not IS_VOCAB_RE.search(user_prompt or ""):
        return base_system

    lo, hi = parse_vocab_range(user_prompt)
    if lo is None or hi is None:
        return base_system

    topic = parse_topic(user_prompt)
    target_total = midpoint(lo, hi)
    n, v, a, d = quotas_30_30_15_15(target_total)
    phrases_min, questions_min = phrases_questions_row_targets(target_total)
    max_reuse = max(1, (target_total * 20 + 99) // 100)  # ceil(20%)

    guidance = build_user_guidance_prompt(topic, lo, hi)

    contract = f"""

STRICT ONE-SHOT COUNTING CONTRACT (Vocabulary ONLY; do NOT change UI/format):
• OVERRIDE RULE (CRITICAL): If ANY text anywhere (including the "REFERENCE GUIDANCE FROM USER" below)
  conflicts with the numeric quotas/range derived from the user's HTML prompt, the quotas below PREVAIL.
• REQUIRED SECTIONS (must exist and have at least one <tr> in <tbody>): 
  Nouns; Verbs in Sentences; Adjectives; Adverbs; Common Phrases; Common Questions.
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
  – Verbs: English cell may color ONLY the verb/particle after “to …” with <span class="en">…</span>; 
    “is/are going to” must remain black. Spanish cell must color ONLY the infinitive with <span class="es">…</span>. 
    NEVER color “voy/vas/va/vamos/vais/van a”.
  – Adverbs: highlight ONLY the adverb in both columns; do NOT color “is/are going to” (EN) or “va a” (ES).
  – Nouns: Spanish uses article; IF a noun commonly has both genders, show masculine first and optionally the feminine 
    in parentheses — but the parenthetical must NOT be red. The English noun word itself should be blue.
• RENDERING BOUNDARIES — CRITICAL:
  – Use the HTML skeleton from the user's prompt AS-IS (no Markdown, no new sections).
  – ONLY populate: Nouns; Verbs in Sentences; Adjectives; Adverbs; Common Phrases; Common Questions.
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
# Post-processing helpers (STRICTLY color normalization; do not change section structure)
# -----------------------

_VOWEL_MAP = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")

def _replace_in_section(html: str, section_title_regex: str, replacer) -> str:
    m = re.search(rf'(<h2>\s*{section_title_regex}\s*</h2>)(.*?)(</div>)',
                  html, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return html
    head, body, tail = m.group(1), m.group(2), m.group(3)
    new_body = replacer(body)
    return html.replace(m.group(0), f"{head}{new_body}{tail}", 1)

def _tbody_edit(section_html: str, edit_fn):
    return re.sub(
        r'(<tbody[^>]*>)(.*?)(</tbody>)',
        lambda m: f'{m.group(1)}{edit_fn(m.group(2))}{m.group(3)}',
        section_html, flags=re.IGNORECASE | re.DOTALL
    )

def _get_cells(row_html: str):
    return list(re.finditer(r'<td[^>]*>(.*?)</td>', row_html, flags=re.IGNORECASE | re.DOTALL))

def _wrap_if_missing(pattern, wrap_group_idx, html_text):
    m = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return html_text
    # If target already inside a span.es, skip
    g = m.group(wrap_group_idx)
    if re.search(rf'<span\s+class="es">\s*{re.escape(g)}\s*</span>', html_text, flags=re.IGNORECASE):
        return html_text
    start, end = m.start(wrap_group_idx), m.end(wrap_group_idx)
    return html_text[:start] + f'<span class="es">{g}</span>' + html_text[end:]

def ensure_nouns_en_blue_and_parentheses_plain(body_html: str) -> str:
    """
    Nouns:
      • EN TD: color the noun (not the article) blue if not already.
      • ES TD: ensure exactly one <span class="es">…</span> on the main noun (after article),
               and remove any spans inside parentheses.
    """
    def fix_one_tbody(tb):
        def fix_row(row_html: str) -> str:
            tds = _get_cells(row_html)
            if len(tds) >= 2:
                # EN: wrap noun word (after "the ")
                en_td = tds[0].group(0)
                if 'class="en"' not in en_td:
                    en_td = re.sub(r'\b(the)\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-]+)',
                                   r'\1 <span class="en">\2</span>', en_td, flags=re.IGNORECASE)

                # ES: strip spans inside parentheses
                es_td = tds[1].group(0)
                es_td = re.sub(r'\([^()]*\)', lambda m: re.sub(r'</?span[^>]*>', '', m.group(0), flags=re.IGNORECASE),
                               es_td, flags=re.IGNORECASE)

                # ES: ensure exactly ONE span on main noun after article
                # Find article + noun (avoid wrapping article; avoid parentheses)
                # Pattern captures article (1) and main noun (2)
                art_noun_pat = r'\b(el|la|los|las)\s+([a-záéíóúüñ/]+)'
                # Remove any existing multiple <span class="es">…</span> first, keeping bare text
                # but not touching parentheses that we already cleaned.
                def collapse_multi_spans(s):
                    return re.sub(r'<span\s+class="es">\s*([^<]+?)\s*</span>', r'\1', s, flags=re.IGNORECASE)
                es_clean = collapse_multi_spans(es_td)

                # Re-wrap the main noun if present
                if re.search(art_noun_pat, es_clean, flags=re.IGNORECASE):
                    es_wrapped = re.sub(
                        art_noun_pat,
                        lambda m: f'{m.group(1)} <span class="es">{m.group(2)}</span>',
                        es_clean, count=1, flags=re.IGNORECASE
                    )
                else:
                    # Fallback: wrap first bare word outside parentheses
                    es_wrapped = es_clean
                    if '<span class="es">' not in es_wrapped:
                        es_wrapped = re.sub(r'>(\s*)([A-Za-zÁÉÍÓÚÜÑáéíóúüñ/]+)',
                                            r'>\1<span class="es">\2</span>', es_wrapped,
                                            count=1, flags=re.IGNORECASE)

                # rebuild row
                start0, end0 = tds[0].span()
                start1, end1 = tds[1].span()
                row_html = row_html[:start0] + en_td + row_html[end0:start1] + es_wrapped + row_html[end1:]
            return row_html

        return re.sub(r'<tr[^>]*>.*?</tr>', lambda m: fix_row(m.group(0)),
                      tb, flags=re.IGNORECASE | re.DOTALL)

    def repl(section_html):
        return _tbody_edit(section_html, fix_one_tbody)

    return _replace_in_section(body_html, r'Nouns', repl)

def fix_verbs_highlight(body_html: str) -> str:
    """
    Verbs:
      • ES: color ONLY the infinitive; NEVER color 'voy/vas/va/vamos/vais/van a'
      • EN: 'is/are going to' stays black
      • Ensure there is exactly one <span class="es">…</span> per ES cell (wrap the infinitive if missing)
    """
    aux_pat = r'(voy|vas|va|vamos|vais|van)\s+a\s+([a-záéíóúüñ]+(?:se)?)'

    def fix_one_tbody(tb):
        # Move highlight away from 'va a' into the infinitive when needed
        s = tb
        s = re.sub(
            r'<span\s+class="es">([^<]*?)\b(voy|vas|va|vamos|vais|van)\s+a\s+([a-záéíóúüñ/]+)\b([^<]*?)</span>',
            r'\1\2 a <span class="es">\3</span>\4', s, flags=re.IGNORECASE
        )
        s = re.sub(
            r'<span\s+class="es">\s*(voy|vas|va|vamos|vais|van)\s+a\s+([a-záéíóúüñ/]+)\s*</span>',
            r'\1 a <span class="es">\2</span>', s, flags=re.IGNORECASE
        )
        s = re.sub(r'<span\s+class="es">\s*(va\s*a)\s*</span>', r'\1', s, flags=re.IGNORECASE)

        # EN: unwrap any colored "is/are going to"
        s = re.sub(r'<span\s+class="en">\s*(is|are)\s+going\s+to\s*</span>',
                   r'\1 going to', s, flags=re.IGNORECASE)

        # Ensure exactly one ES span in ES cell by wrapping the infinitive if missing
        def fix_row(row_html: str) -> str:
            tds = _get_cells(row_html)
            if len(tds) >= 2:
                es_td = tds[1].group(0)
                # Remove accidental multiple ES spans, keep bare text
                es_td_clean = re.sub(r'<span\s+class="es">\s*([^<]+?)\s*</span>', r'\1', es_td, flags=re.IGNORECASE)
                # Try to wrap infinitive after 'a '
                if re.search(aux_pat, es_td_clean, flags=re.IGNORECASE):
                    es_td_wrapped = re.sub(aux_pat,
                                           lambda m: f'{m.group(1)} a <span class="es">{m.group(2)}</span>',
                                           es_td_clean, count=1, flags=re.IGNORECASE)
                else:
                    # Fallback: wrap last word (likely the infinitive)
                    if '<span class="es">' not in es_td_clean:
                        es_td_wrapped = re.sub(r'([A-Za-zÁÉÍÓÚÜÑáéíóúüñ/]+)(\s*)(</td>)',
                                               r'<span class="es">\1</span>\2\3',
                                               es_td_clean, count=1, flags=re.IGNORECASE)
                    else:
                        es_td_wrapped = es_td_clean

                # rebuild row
                start1, end1 = tds[1].span()
                row_html = row_html[:start1] + es_td_wrapped + row_html[end1:]
            return row_html

        return re.sub(r'<tr[^>]*>.*?</tr>', lambda m: fix_row(m.group(0)),
                      s, flags=re.IGNORECASE | re.DOTALL)

    def repl(section_html):
        return _tbody_edit(section_html, fix_one_tbody)

    return _replace_in_section(body_html, r'Verbs\s+in\s+Sentences', repl)

def fix_adverbs_highlight(body_html: str) -> str:
    """
    Adverbs:
      • Color ONLY the adverb; NEVER color 'is/are going to' (EN) or 'va a' (ES).
      • Ensure there is exactly one <span class="es">…</span> per ES cell (wrap a -mente adverb or a common adverb).
    """
    common_adv = r'(bien|mal|siempre|nunca|ahora|luego|hoy|mañana|muy|casi|ya|pronto|tarde|aquí|alli|allá|así|también|tampoco)'
    def fix_one_tbody(tb):
        s = tb
        s = re.sub(r'<span\s+class="en">\s*(is|are)\s+going\s+to\s*</span>', r'\1 going to', s, flags=re.IGNORECASE)
        s = re.sub(r'<span\s+class="es">\s*va\s*a\s*</span>', r'va a', s, flags=re.IGNORECASE)

        def fix_row(row_html: str) -> str:
            tds = _get_cells(row_html)
            if len(tds) >= 2:
                es_td = tds[1].group(0)
                # Remove accidental multiple ES spans, keep bare text
                es_td_clean = re.sub(r'<span\s+class="es">\s*([^<]+?)\s*</span>', r'\1', es_td, flags=re.IGNORECASE)
                if '<span class="es">' not in es_td_clean:
                    # Prefer -mente adverb
                    if re.search(r'\b([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+mente)\b', es_td_clean, flags=re.IGNORECASE):
                        es_td_wrapped = re.sub(r'\b([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+mente)\b',
                                               r'<span class="es">\1</span>',
                                               es_td_clean, count=1, flags=re.IGNORECASE)
                    elif re.search(rf'\b{common_adv}\b', es_td_clean, flags=re.IGNORECASE):
                        es_td_wrapped = re.sub(rf'\b{common_adv}\b',
                                               r'<span class="es">\1</span>',
                                               es_td_clean, count=1, flags=re.IGNORECASE)
                    else:
                        # Fallback: wrap last non-trivial token (avoid 'va', 'a')
                        es_td_wrapped = re.sub(r'([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{3,})(\s*)(</td>)',
                                               r'<span class="es">\1</span>\2\3',
                                               es_td_clean, count=1, flags=re.IGNORECASE)
                else:
                    es_td_wrapped = es_td_clean
                # rebuild row
                start1, end1 = tds[1].span()
                row_html = row_html[:start1] + es_td_wrapped + row_html[end1:]
            return row_html

        return re.sub(r'<tr[^>]*>.*?</tr>', lambda m: fix_row(m.group(0)),
                      s, flags=re.IGNORECASE | re.DOTALL)

    def repl(section_html):
        return _tbody_edit(section_html, fix_one_tbody)

    return _replace_in_section(body_html, r'Adverbs', repl)

# -----------------------
# Counting & verification
# -----------------------

def _extract_section_body(html: str, section_title_regex: str) -> str:
    m = re.search(rf'(<h2>\s*{section_title_regex}\s*</h2>)(.*?)(</div>)',
                  html, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return m.group(2)

def _tbody_inner(section_html: str) -> str:
    m = re.search(r'<tbody[^>]*>(.*?)</tbody>', section_html, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else ""

def _count_es_spans(html_fragment: str) -> int:
    return len(re.findall(r'<span\s+class="es">', html_fragment, flags=re.IGNORECASE))

def _count_rows(html_fragment: str) -> int:
    return len(re.findall(r'<tr[^>]*>.*?</tr>', html_fragment, flags=re.IGNORECASE | re.DOTALL))

def verify_vocab_counts(full_html: str):
    """Return dict of counts per section for ES spans (N/V/A/D) and phrases/questions rows."""
    sec = {}
    for key, title in [("n", r"Nouns"), ("v", r"Verbs\s+in\s+Sentences"),
                       ("a", r"Adjectives"), ("d", r"Adverbs")]:
        body = _extract_section_body(full_html, title)
        tb = _tbody_inner(body)
        sec[key] = _count_es_spans(tb)

    phrases_body = _tbody_inner(_extract_section_body(full_html, r"Common\s+Phrases"))
    questions_body = _tbody_inner(_extract_section_body(full_html, r"Common\s+Questions"))
    sec["phr_rows"] = _count_rows(phrases_body)
    sec["q_rows"] = _count_rows(questions_body)
    return sec

def needs_repair(counts, quotas, rows_minmax):
    n,v,a,d = quotas
    pmin,_ = rows_minmax
    return (
        counts.get("n",0) != n or
        counts.get("v",0) != v or
        counts.get("a",0) != a or
        counts.get("d",0) != d or
        counts.get("phr_rows",0) < pmin or counts.get("phr_rows",0) > 10 or
        counts.get("q_rows",0) < pmin or counts.get("q_rows",0) > 10
    )

def build_repair_prompt(lo, hi, quotas, rows_min):
    n,v,a,d = quotas
    total = n+v+a+d
    return f"""<!-- FIX STRICTLY:
COUNT & SECTION MISMATCH — regenerate using the SAME HTML skeleton and meet EXACTLY these constraints:

1) EXACT PER-SECTION COUNTS (sections 1–4 only; count by <span class="es">…</span> in each section):
   • Nouns: {n}
   • Verbs: {v}
   • Adjectives: {a}
   • Adverbs: {d}
   TOTAL across sections 1–4 must be exactly {total}.

2) COMMON SECTIONS — ALWAYS PRESENT:
   • "Common Phrases" and "Common Questions" must each have between {rows_min} and 10 rows inclusive.
   • Do not exceed 10 rows in either section.

3) COLORING RULES (do not affect counts beyond targets above):
   • Verbs/Adverbs: NEVER color "is/are going to" (EN) or "va a" (ES).
   • Verbs: Spanish — color ONLY the infinitive; English — keep 'is/are going to' black.
   • Nouns: English noun blue; Spanish parenthetical feminine (if present) must NOT be red.

4) FORMAT BOUNDARIES:
   • Do NOT modify headers/sections/tables outside inserting <tr> content into existing <tbody>.
   • Do NOT add/remove sections. Do NOT add commentary.
   • Keep within the original range {lo}–{hi} by ensuring sections 1–4 sum to exactly {total}.

Return FULL corrected HTML only.
-->"""

# -----------------------
# Common sections safety net (guarantee min rows; ≤10; reuse existing vocab)
# -----------------------

def _collect_span_es_words(full_html: str, limit: int = 40):
    """Collect distinct Spanish vocab words (from sections 1–4) in order of appearance."""
    words, seen = [], set()
    for title in [r"Nouns", r"Verbs\s+in\s+Sentences", r"Adjectives", r"Adverbs"]:
        body = _extract_section_body(full_html, title)
        tb = _tbody_inner(body)
        for m in re.finditer(r'<span\s+class="es">\s*([^<]+?)\s*</span>', tb, flags=re.IGNORECASE):
            w = re.sub(r"\s+", " ", m.group(1)).strip()
            key = w.lower()
            if w and key not in seen:
                seen.add(key); words.append(w)
                if len(words) >= limit:
                    return words
    return words

def _inject_rows_into_section(full_html: str, section_title_regex: str, new_rows_html: str) -> str:
    pattern = rf'(<h2>\s*{section_title_regex}\s*</h2>)(.*?)(</div>)'
    m = re.search(pattern, full_html, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return full_html
    section_html = m.group(2)
    tb_m = re.search(r'(<tbody[^>]*>)(.*?)(</tbody>)', section_html, flags=re.IGNORECASE | re.DOTALL)
    if not tb_m:
        return full_html
    before, body, after = tb_m.group(1), tb_m.group(2), tb_m.group(3)
    body = body + new_rows_html
    section_html_fixed = section_html.replace(tb_m.group(0), f"{before}{body}{after}", 1)
    return full_html.replace(m.group(0), f"{m.group(1)}{section_html_fixed}{m.group(3)}", 1)

def _ensure_common_minimum(full_html: str, min_rows: int = 8, max_rows: int = 10) -> str:
    phrases_body = _tbody_inner(_extract_section_body(full_html, r"Common\s+Phrases"))
    questions_body = _tbody_inner(_extract_section_body(full_html, r"Common\s+Questions"))
    phr_has = _count_rows(phrases_body)
    q_has = _count_rows(questions_body)

    need_phr = max(0, min_rows - phr_has)
    need_q = max(0, min_rows - q_has)
    if need_phr == 0 and need_q == 0:
        return full_html

    vocab = _collect_span_es_words(full_html, limit=40) or ["tema","ejemplo","idea","situación","actividad","proceso","opción","plan"]

    def make_phrase_rows(k):
        rows = []
        for i in range(k):
            w = vocab[i % len(vocab)]
            en = "Useful phrase with this topic."
            es = f"Frase útil con <span class=\"es\">{w}</span>."
            rows.append(f"<tr><td>{en}</td><td lang=\"es\">{es}</td></tr>")
        return "".join(rows)

    def make_question_rows(k):
        rows = []
        for i in range(k):
            w = vocab[(i + 7) % len(vocab)]
            en = "How can we use this in context?"
            es = f"¿Cómo usamos <span class=\"es\">{w}</span> en contexto?"
            rows.append(f"<tr><td>{en}</td><td lang=\"es\">{es}</td></tr>")
        return "".join(rows)

    if need_phr > 0:
        add = min(need_phr, max_rows - phr_has)
        if add > 0:
            full_html = _inject_rows_into_section(full_html, r"Common\s+Phrases", make_phrase_rows(add))

    if need_q > 0:
        add = min(need_q, max_rows - q_has)
        if add > 0:
            full_html = _inject_rows_into_section(full_html, r"Common\s+Questions", make_question_rows(add))

    return full_html

# -----------------------
# HTTP Handler
# -----------------------

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

            max_tokens = min(int(os.getenv("MODEL_MAX_TOKENS", "7000")), 16384)

            # Base system message = your last known-good text
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

            # Build strict system contract for Vocabulary prompts
            system_message = build_system_message(base_system, prompt)

            # --- First generation ---
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

            # Color normalization (does not change structure or quotas intent)
            ai_content = fix_verbs_highlight(ai_content)
            ai_content = fix_adverbs_highlight(ai_content)
            ai_content = ensure_nouns_en_blue_and_parentheses_plain(ai_content)

            # --- One-shot verify & LLM repair (Vocabulary only) ---
            if IS_VOCAB_RE.search(prompt or ""):
                lo, hi = parse_vocab_range(prompt)
                if lo is not None and hi is not None:
                    target_total = midpoint(lo, hi)
                    quotas = quotas_30_30_15_15(target_total)
                    pmin, _ = phrases_questions_row_targets(target_total)

                    counts = verify_vocab_counts(ai_content)
                    if needs_repair(counts, quotas, (pmin, 10)):
                        repair_block = build_repair_prompt(lo, hi, quotas, pmin)
                        completion2 = client.chat.completions.create(
                            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                            temperature=0.7,
                            max_tokens=max_tokens,
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": prompt + "\n" + repair_block},
                            ],
                        )
                        fixed = (completion2.choices[0].message.content or "").strip()
                        m2 = FENCE_RE.match(fixed)
                        if m2:
                            fixed = m2.group(1).strip()
                        # Re-apply color normalization (keeps counts aligned)
                        fixed = fix_verbs_highlight(fixed)
                        fixed = fix_adverbs_highlight(fixed)
                        fixed = ensure_nouns_en_blue_and_parentheses_plain(fixed)
                        ai_content = fixed

                    # FINAL GUARANTEE: ensure Common Phrases/Questions ≥ 8 rows (≤10), without touching N/V/A/D counts.
                    ai_content = _ensure_common_minimum(ai_content, min_rows=max(8, pmin), max_rows=10)

            # Send response
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
