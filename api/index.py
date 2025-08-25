import os
import re
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

# ---------- Helpers: prompt type detection ----------

def is_vocab_prompt(prompt: str) -> bool:
    return "<!-- FCS VOCABULARY OUTPUT" in prompt

def is_convo_prompt(prompt: str) -> bool:
    return "<!-- FCS CONVERSATION OUTPUT" in prompt

# ---------- Helpers: HTML cleanup ----------

def strip_code_fences(s: str) -> str:
    # Remove common triple-backtick or xml/markdown fences if ever present
    fence = "```"
    if s.strip().startswith(fence):
        # ```lang\n...\n```
        parts = s.split(fence)
        if len(parts) >= 3:
            body = parts[1]
            # If language tag present, drop the first line
            if "\n" in body:
                first, rest = body.split("\n", 1)
                if first.strip().lower() in ("html", "xml", "markdown"):
                    return rest.strip()
            return body.strip()
    return s.strip()

# ---------- Helpers: safe HTML text extraction ----------

TAG_RE = re.compile(r"<[^>]+>")

def html_text(s: str) -> str:
    return TAG_RE.sub("", s or "").replace("&nbsp;", " ").strip()

# ---------- Parse requirements from PROMPT (what the UI asked for) ----------

def parse_vocab_requirements_from_prompt(prompt: str):
    """
    Returns: (min_count, max_count, included_sections)
      - included_sections is a set among {"nouns","verbs","descriptive","phrases","questions"}
    Strategy:
      1) Find explicit BETWEEN X and Y in the prompt rules if present.
      2) Otherwise infer from Level in the <h1> line inside the prompt.
      3) Find "Generate ONLY these sections: ..." to learn which sections are requested.
    """
    # 1) Try explicit BETWEEN X and Y numbers from the prompt rules
    m = re.search(r"BETWEEN\s+(\d+)\s+and\s+(\d+)", prompt, flags=re.IGNORECASE)
    min_count = None
    max_count = None
    if m:
        min_count = int(m.group(1))
        max_count = int(m.group(2))

    # 2) If not found, infer from Level in the prompt's H1 (e.g., "— Level 4—Conversational")
    if min_count is None or max_count is None:
        # Level bands
        bands = {
            1: (33, 55),
            2: (60, 85),
            3: (90, 120),
            4: (120, 180),
            5: (160, 240),
        }
        m2 = re.search(r"Level\s+([1-5])", prompt)
        if m2:
            lvl = int(m2.group(1))
            min_count, max_count = bands[lvl]
        else:
            # fallback conservative
            min_count, max_count = 33, 55

    # 3) Which sections to include
    included_sections = set(["nouns", "verbs", "descriptive", "phrases", "questions"])
    ms = re.search(r"Generate ONLY these sections:\s*([^.\n]+)", prompt, flags=re.IGNORECASE)
    if ms:
        raw = ms.group(1).lower()
        want = set()
        if "nouns" in raw: want.add("nouns")
        if "verbs" in raw: want.add("verbs")
        if "descriptive" in raw: want.add("descriptive")
        if "phrases" in raw: want.add("phrases")
        if "questions" in raw: want.add("questions")
        if want:
            included_sections = want

    return min_count, max_count, included_sections

def parse_convo_ui_from_prompt(prompt: str):
    """
    Extract UI JSON block: UI = {...}
    Returns a dict with keys used by front-end:
      - numConvos, numSpeakers, turnsApprox, sentences_per_turn{min,max},
        includeEN, addFIB, fibFocus, showFIBKey, showVocabList
    """
    m = re.search(r"UI\s*=\s*(\{[\s\S]*?\})", prompt)
    if not m:
        return None
    ui_str = m.group(1)
    try:
        ui = json.loads(ui_str)
        # Derive fibCount like UI does
        turns = max(1, int(ui.get("turnsApprox", 10)))
        fib_count = max(8, min(30, round(turns * 0.8)))
        ui["_fibCount"] = fib_count
        return ui
    except Exception:
        return None

# ---------- Validators (on the generated HTML) ----------

def _extract_section_tbody(html: str, section_title_regex: str) -> str:
    """
    Returns inner HTML of <tbody> for a given section (matched by h2 text).
    """
    pattern = re.compile(
        rf'<div class="section">\s*<h2[^>]*>\s*{section_title_regex}\s*</h2>.*?<tbody>(.*?)</tbody>',
        flags=re.IGNORECASE | re.DOTALL
    )
    m = pattern.search(html)
    return m.group(1) if m else ""

def validate_vocab_output(html: str, min_count: int, max_count: int, included_sections: set):
    """
    Checks:
      - unique <span class="es">…</span> across Nouns/Verbs/Descriptive within [min,max]
      - Sections present only as requested (optional but helpful)
    Returns: "" if OK, else a semicolon-joined error string.
    """
    problems = []

    # Section presence enforcement (ONLY those requested)
    title_map = {
        "nouns": r"Nouns",
        "verbs": r"Verbs",
        "descriptive": r"Descriptive Words",
        "phrases": r"Common Phrases",
        "questions": r"Common Questions",
    }

    # Ensure requested present & others absent
    for key, title_re in title_map.items():
        tbody = _extract_section_tbody(html, rf"\s*{title_re}")
        if key in included_sections:
            if not tbody or not re.search(r"<tr", tbody, flags=re.I):
                problems.append(f"missing or empty required section: {key}")
        else:
            if tbody and re.search(r"<tr", tbody, flags=re.I):
                problems.append(f"forbidden extra section present: {key}")

    # Count unique Spanish vocab across Nouns/Verbs/Descriptive
    def es_terms_in(section_key: str):
        if section_key not in title_map:
            return []
        tb = _extract_section_tbody(html, rf"\s*{title_map[section_key]}")
        if not tb:
            return []
        return [html_text(x).strip().lower()
                for x in re.findall(r'<span\s+class="es"\s*>(.*?)</span>', tb, flags=re.I | re.S)]

    es_all = set()
    for key in ("nouns", "verbs", "descriptive"):
        if key in included_sections:
            for term in es_terms_in(key):
                if term:
                    es_all.add(term)

    count = len(es_all)
    if count < min_count or count > max_count:
        problems.append(f"Spanish vocab count {count} outside required range {min_count}–{max_count}")

    return "; ".join(problems)

def _extract_convo_sections(html: str):
    """
    Returns list of (tbody_html, has_english_col) for each Conversation section.
    """
    out = []
    # Find each Conversation section
    sec_re = re.compile(
        r'<div class="section">\s*<h2[^>]*>\s*Conversation\s+\d+\s*</h2>.*?<table[^>]*>.*?<thead>(.*?)</thead>.*?<tbody>(.*?)</tbody>',
        flags=re.I | re.S
    )
    for m in sec_re.finditer(html):
        thead = m.group(1)
        tbody = m.group(2)
        # Determine if English column exists
        has_en = bool(re.search(r'>\s*English\s*<', thead, flags=re.I))
        out.append((tbody, has_en))
    return out

def _split_rows(tbody_html: str):
    return re.findall(r"<tr[^>]*>(.*?)</tr>", tbody_html or "", flags=re.I | re.S)

def _split_cells(tr_html: str):
    return re.findall(r"<td[^>]*>(.*?)</td>", tr_html or "", flags=re.I | re.S)

def _sentence_count(text: str):
    t = " ".join((text or "").split())
    # Count sentence-like chunks; ensure at least 1
    parts = re.findall(r"[^.!?]+[.!?]+", t)
    if parts:
        return len(parts)
    return 1 if t else 0

def validate_convo_output(html: str, ui: dict):
    """
    Enforce:
      - Exactly ui["numConvos"]
      - Each conversation has exactly ui["turnsApprox"] rows
      - Each row Spanish (and English if included) respects min/max sentences
      - FIB table (if addFIB == 'yes') with exactly ui["_fibCount"] rows
      - FIB English cell contains <span class="en">…</span> and no blanks
        and Spanish cell contains "(english) ________"
    Returns "" if OK, else errors joined by '; '.
    """
    problems = []
    numConvos = int(ui.get("numConvos", 1))
    turns = max(1, int(ui.get("turnsApprox", 10)))
    min_s = int(ui.get("sentences_per_turn", {}).get("min", 1))
    max_s = int(ui.get("sentences_per_turn", {}).get("max", min_s))
    include_en = (ui.get("includeEN", "yes") == "yes")
    add_fib = (ui.get("addFIB", "no") == "yes")
    fib_rows_needed = int(ui.get("_fibCount", 8))
    show_fib_key = (ui.get("showFIBKey", "no") == "yes")
    show_vocab = (ui.get("showVocabList", "no") == "yes")

    # Conversation sections
    convs = _extract_convo_sections(html)
    if len(convs) != numConvos:
        problems.append(f"need exactly {numConvos} conversation sections, found {len(convs)}")

    for i, (tbody, has_en) in enumerate(convs, start=1):
        rows = _split_rows(tbody)
        if len(rows) != turns:
            problems.append(f"Conversation {i}: need exactly {turns} turns, found {len(rows)}")
        for ri, tr in enumerate(rows, start=1):
            cells = _split_cells(tr)
            if include_en:
                if len(cells) < 2:
                    problems.append(f"Conversation {i} row {ri}: missing cells for EN/ES")
                    continue
                en_txt = html_text(cells[0])
                es_txt = html_text(cells[1])
                c_en = _sentence_count(en_txt)
                c_es = _sentence_count(es_txt)
                if c_es < min_s or c_es > max_s:
                    problems.append(f"Conversation {i} row {ri} (ES): {c_es} sentences (must be {min_s}-{max_s})")
                if c_en < min_s or c_en > max_s:
                    problems.append(f"Conversation {i} row {ri} (EN): {c_en} sentences (must be {min_s}-{max_s})")
            else:
                if len(cells) < 1:
                    problems.append(f"Conversation {i} row {ri}: missing ES cell")
                    continue
                es_txt = html_text(cells[0])
                c_es = _sentence_count(es_txt)
                if c_es < min_s or c_es > max_s:
                    problems.append(f"Conversation {i} row {ri} (ES): {c_es} sentences (must be {min_s}-{max_s})")

    # FIB section
    if add_fib:
        fib_tb = _extract_section_tbody(html, r"Practice\s+—\s+Fill\-in\-the\-Blank")
        if not fib_tb:
            problems.append("FIB: missing section/table")
        else:
            fib_rows = _split_rows(fib_tb)
            if len(fib_rows) != fib_rows_needed:
                problems.append(f"FIB: need exactly {fib_rows_needed} rows, found {len(fib_rows)}")
            for idx, tr in enumerate(fib_rows, start=1):
                cells = _split_cells(tr)
                if len(cells) < 2:
                    problems.append(f"FIB row {idx}: needs two cells (English | Español)")
                    continue
                en_cell, es_cell = cells[0], cells[1]
                en_txt = html_text(en_cell)
                es_txt = html_text(es_cell)
                # English must contain a blue target span and no blanks
                m_en = re.search(r'<span\s+class="en"\s*>(.*?)</span>', en_cell, flags=re.I | re.S)
                if not m_en or not m_en.group(1).strip():
                    problems.append(f"FIB row {idx}: English must include <span class=\"en\">target</span>")
                if re.search(r"_{2,}", en_txt):
                    problems.append(f"FIB row {idx}: English must NOT contain blanks")
                # Spanish must contain "(english) ________"
                if m_en:
                    hint = html_text(m_en.group(1))
                    # Require hint immediately before blank
                    if not re.search(r"\(" + re.escape(hint) + r"\)\s*_+", es_txt, flags=re.I):
                        problems.append(f"FIB row {idx}: Spanish must contain '({hint}) ________'")
                if not re.search(r"_\s*{6,}|_{6,}", es_txt):
                    # tolerate spaces between underscores
                    if "______" not in es_txt and "________" not in es_txt:
                        problems.append(f"FIB row {idx}: missing blank '________'")

    # Optional: answer key / vocab sections presence if toggled on
    if show_fib_key:
        if not re.search(r'<h2[^>]*>\s*Answer\s+Key\s*</h2>', html, flags=re.I):
            problems.append("missing Answer Key section")
    if show_vocab:
        if not re.search(r'<h2[^>]*>\s*Vocabulary\s+Used\s*</h2>', html, flags=re.I):
            problems.append("missing Vocabulary Used section")

    return "; ".join(problems)

# ---------- Core OpenAI call with internal regeneration loop ----------

def call_openai_html(client: OpenAI, base_system: str, user_prompt: str, max_attempts: int = 3):
    """
    Calls OpenAI, validates based on prompt type, and retries internally with a violation notice
    until constraints are satisfied or attempts exhausted. Returns final HTML.
    """
    # Pre-parse expectations from the *PROMPT* so we can validate the *OUTPUT*
    vocab_requirements = None
    convo_ui = None
    prompt_type = "generic"
    if is_vocab_prompt(user_prompt):
        prompt_type = "vocab"
        vocab_requirements = parse_vocab_requirements_from_prompt(user_prompt)
    elif is_convo_prompt(user_prompt):
        prompt_type = "convo"
        convo_ui = parse_convo_ui_from_prompt(user_prompt)

    violation_note = ""
    html = ""
    for attempt in range(max_attempts):
        # Build messages, optionally appending a violation block for regeneration
        user_content = user_prompt
        if violation_note:
            user_content = f"""{user_prompt}

<!-- VIOLATION NOTICE:
{violation_note}
Regenerate STRICTLY COMPLYING with ALL constraints. Do not change structure or class names. Return ONLY full valid HTML.
-->"""

        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.4,  # stick to rules
            messages=[
                {
                    "role": "system",
                    "content": base_system
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        )
        ai_content = (completion.choices[0].message.content or "").strip()
        html = strip_code_fences(ai_content)

        # Validate based on prompt type
        if prompt_type == "vocab" and vocab_requirements:
            vmin, vmax, incl = vocab_requirements
            problems = validate_vocab_output(html, vmin, vmax, incl)
            if not problems:
                return html
            violation_note = problems

        elif prompt_type == "convo" and convo_ui:
            problems = validate_convo_output(html, convo_ui)
            if not problems:
                return html
            violation_note = problems

        else:
            # Generic: return as-is if no specific validator
            return html

    # If here, return last html (best effort) even if problems remained
    return html

# ---------- HTTP Handler ----------

class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        # Allow Kajabi (iframe) and local dev
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_POST(self):
        try:
            # Read request
            content_length = int(self.headers.get('Content-Length', '0'))
            post_data = self.rfile.read(content_length) if content_length else b'{}'
            data = json.loads(post_data.decode('utf-8'))
            prompt = data.get("prompt", "").strip()
            if not prompt:
                raise ValueError("Missing 'prompt' in request body.")

            # API key
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Server configuration error: The OPENAI_API_KEY is missing from Vercel env vars.")

            # Base system message: strict, self-checking, HTML-only
            base_system = (
                "You are an expert assistant for the Fast Conversational Spanish (FCS) program. "
                "Return ONLY the raw HTML (a full, valid, self-contained document). "
                "BEFORE YOU REPLY: do an internal SELF-CHECK to ensure every requirement in the user's template is satisfied EXACTLY. "
                "No table may have an empty <tbody>. "
                "If any requirement would be violated (counts, sections, headers, sentence bounds, etc.), FIX the content yourself "
                "BEFORE sending the final HTML. Do NOT add explanations, greetings, or code fences."
            )

            client = OpenAI(api_key=api_key)

            # Call model with internal validator/retry (invisible to end user)
            final_html = call_openai_html(client, base_system=base_system, user_prompt=prompt, max_attempts=3)

            response_payload = {"content": final_html}

            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

        except Exception as e:
            print(f"AN ERROR OCCURRED: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self._send_cors_headers()
            self.end_headers()
            error_payload = {"error": "An internal server error occurred.", "details": str(e)}
            self.wfile.write(json.dumps(error_payload).encode('utf-8'))
