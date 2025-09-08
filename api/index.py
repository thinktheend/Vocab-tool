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

def parse_vocab_range(prompt_text: str):
    m = RANGE_RE.search(prompt_text or "")
    if not m:
        return (None, None)
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)

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
    Compute required minimum rows for Common Phrases and Common Questions.
    Floor 10; scale gently with total vocab (does not affect vocab count).
    """
    rows = max(10, min(30, round(total_vocab_midpoint / 8)))
    return rows, rows

def build_system_message(base_system: str, user_prompt: str) -> str:
    """
    If this is a Vocabulary prompt and we can read the range, append a STRICT contract
    that forces one-shot midpoint counts while preserving the front-end skeleton,
    and require non-empty, scaled Common Phrases/Questions.
    """
    if not IS_VOCAB_RE.search(user_prompt or ""):
        return base_system  # Conversation/Test: unchanged

    lo, hi = parse_vocab_range(user_prompt)
    if lo is None or hi is None:
        return base_system

    target_total = midpoint(lo, hi)
    n, v, a, d = quotas_30_30_15_15(target_total)
    phrases_min, questions_min = phrases_questions_row_targets(target_total)
    max_reuse = max(1, (target_total * 20 + 99) // 100)  # ceil(20% of total)

    # Contract focuses on QUANTITY for sections 1–4 and adds mandatory
    # population of Common Phrases/Questions without affecting the count.
    contract = f"""

STRICT ONE-SHOT COUNTING CONTRACT (Vocabulary ONLY; do NOT change UI/format):
• TARGET TOTAL (sections 1–4 only): EXACTLY {target_total} Spanish vocabulary items counted by
  the number of <span class="es">…</span> target words in Nouns, Verbs, Adjectives, Adverbs.
• PER-SECTION QUOTAS (enforce exactly):
  – Nouns: {n}
  – Verbs: {v}
  – Adjectives: {a}
  – Adverbs: {d}
• OUTPUT BOUNDARIES — CRITICAL:
  – Use the HTML skeleton provided in the user's prompt AS-IS.
  – Insert ONLY <tr> row content into the existing <tbody> of each section.
  – Do NOT print extra text outside the document. No commentary or code fences.
• COMMON PHRASES & COMMON QUESTIONS — MANDATORY:
  – Populate BOTH sections with table rows inside their existing <tbody>.
  – Minimum rows: Common Phrases ≥ {phrases_min}; Common Questions ≥ {questions_min}.
  – Reuse only vocabulary from sections 1–4 (no new vocabulary). Total distinct reused words
    across BOTH sections combined must be ≤ {max_reuse} (≈20% of {target_total}).
  – Rows in these sections do NOT count toward the {target_total} total.
• HIGHLIGHTING / WHAT COUNTS:
  – Count ONLY the red Spanish target words wrapped in <span class="es">…</span> in sections 1–4.
  – Nouns: Spanish cell uses article; IF a noun commonly has both genders, show masculine first
    and append the feminine in parentheses, e.g., el médico (la médica), el cliente (la cliente).
  – Verbs: Sentences use “He/She/It/They + is/are going to + [infinitive]”.
    HIGHLIGHT ONLY the infinitive verb token (e.g., facturar, despegar, aterrizar).
    NEVER wrap “voy/vas/va/vamos/vais/van a” in red.
    Example (ES cell):
      Él va a <span class="es">facturar</span> su equipaje.
      Ellos van a <span class="es">aterrizar</span> pronto.
  – Adjectives: sentences with “is/are + adjective”; highlight ONLY the adjective.
  – Adverbs: sentences that reuse verbs; highlight ONLY the adverb.
• SELF-CHECK BEFORE SENDING:
  – Ensure the exact per-section quotas and the exact grand total are satisfied in sections 1–4.
  – Ensure BOTH Common sections meet their row minimums and only reuse allowed vocabulary.
  – Ensure well-formed HTML that fits the provided skeleton.
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
        return raw  # invariant
    # default: return as-is (we won't guess)
    return raw

# Limit replacements to a specific section window
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
    Inside Verbs section tbody, ensure only the infinitive is red, not 'voy/vas/va/vamos/vais/van a'.
    """

    # Only operate inside the <tbody>...</tbody> range(s)
    def fix_one_tbody(tb):
        s = tb

        # Move highlight from the whole chunk to just the infinitive after 'a '
        # e.g. <span class="es">Él va a facturar su equipaje</span>
        # -> Él va a <span class="es">facturar</span> su equipaje
        s = re.sub(
            r'<span\s+class="es">([^<]*?)\b(voy|vas|va|vamos|vais|van)\s+a\s+([a-záéíóúüñ]+)\b([^<]*?)</span>',
            r'\1\2 a <span class="es">\3</span>\4',
            s, flags=re.IGNORECASE
        )
        # Case: <span class="es">va a facturar</span> -> va a <span class="es">facturar</span>
        s = re.sub(
            r'<span\s+class="es">\s*(voy|vas|va|vamos|vais|van)\s+a\s+([a-záéíóúüñ]+)\s*</span>',
            r'\1 a <span class="es">\2</span>',
            s, flags=re.IGNORECASE
        )
        # Case: <span class="es">va a</span> facturar -> va a <span class="es">facturar</span>
        s = re.sub(
            r'<span\s+class="es">\s*(voy|vas|va|vamos|vais|van)\s+a\s*</span>\s*([a-záéíóúüñ]+)',
            r'\1 a <span class="es">\2</span>',
            s, flags=re.IGNORECASE
        )
        # Case: <span class="es">va</span> a facturar -> va a <span class="es">facturar</span>
        s = re.sub(
            r'<span\s+class="es">\s*(voy|vas|va|vamos|vais|van)\s*</span>\s*a\s+([a-záéíóúüñ]+)',
            r'\1 a <span class="es">\2</span>',
            s, flags=re.IGNORECASE
        )
        return s

    def repl(section_html):
        # apply for each tbody in the section (usually one)
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
            # m.groups(): prefix('el ' span start), word, span end
            word = m.group(2)
            # skip if already followed by '('
            after = m.group(0)
            # derive
            fem = derive_feminine(word)
            if not fem or fem.lower() == word.lower():
                # invariant or unknown — still show feminine article if invariant
                return f'{m.group(1)}{word}</span> (la <span class="es">{fem}</span>)'
            return f'{m.group(1)}{word}</span> (la <span class="es">{fem}</span>)'

        # Only add when no "(la ...)" already present in the same TD
        # Pattern: <td> ... <span class="es">el WORD</span> ... </td>
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

        # apply replacement only in the ES (second) <td> of noun rows
        # We conservatively apply to all <td> blocks in tbody; header rows use <th> so safe.
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
            # 1) Ensure verbs highlight only the infinitive (never 'va a')
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
