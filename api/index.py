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
    return f"""You are an expert assistant for FCS.
 ... (the guidance content remains as originally provided) ...
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
    # Determine which sections are included in prompt HTML
    hasNouns = bool(re.search(r'<h2>\s*Nouns\s*</h2>', user_prompt, flags=re.IGNORECASE))
    hasVerbs = bool(re.search(r'<h2>\s*Verbs\s+in\s+Sentences\s*</h2>', user_prompt, flags=re.IGNORECASE))
    hasAdj = bool(re.search(r'<h2>\s*Adjectives\s*</h2>', user_prompt, flags=re.IGNORECASE))
    hasAdv = bool(re.search(r'<h2>\s*Adverbs\s*</h2>', user_prompt, flags=re.IGNORECASE))
    hasPhr = bool(re.search(r'<h2>\s*Common\s+Phrases\s*</h2>', user_prompt, flags=re.IGNORECASE))
    hasQ = bool(re.search(r'<h2>\s*Common\s+Questions\s*</h2>', user_prompt, flags=re.IGNORECASE))

    # Calculate quotas distribution for active sections
    weights = {}
    total_weight = 0.0
    if hasNouns: weights['n'] = 0.30; total_weight += 0.30
    if hasVerbs: weights['v'] = 0.30; total_weight += 0.30
    if hasAdj:   weights['a'] = 0.15; total_weight += 0.15
    if hasAdv:   weights['d'] = 0.15; total_weight += 0.15

    quotas = {'n':0, 'v':0, 'a':0, 'd':0}
    if total_weight > 0:
        for key, weight in weights.items():
            quotas[key] = round(target_total * (weight / total_weight))
        diff = target_total - sum(quotas.values())
        keys = ['n','v','a','d']
        i = 0
        while diff != 0:
            k = keys[i % 4]
            if k in quotas:
                if diff > 0:
                    quotas[k] += 1
                    diff -= 1
                elif diff < 0 and quotas[k] > 0:
                    quotas[k] -= 1
                    diff += 1
            i += 1

    phrases_min, questions_min = phrases_questions_row_targets(target_total)
    max_reuse = max(1, (target_total * 20 + 99) // 100)

    guidance = build_user_guidance_prompt(topic, lo, hi)

    contract = "STRICT ONE-SHOT COUNTING CONTRACT (Vocabulary ONLY; do NOT change UI/format):\n"
    contract += "• OVERRIDE RULE (CRITICAL): If ANY text anywhere (including the \"REFERENCE GUIDANCE FROM USER\" below)\n"
    contract += "  conflicts with the numeric quotas/range derived from the user's HTML prompt, the quotas below PREVAIL.\n"
    # REQUIRED SECTIONS
    required_secs = []
    if hasNouns: required_secs.append("Nouns")
    if hasVerbs: required_secs.append("Verbs in Sentences")
    if hasAdj:   required_secs.append("Adjectives")
    if hasAdv:   required_secs.append("Adverbs")
    if hasPhr:   required_secs.append("Common Phrases")
    if hasQ:     required_secs.append("Common Questions")
    if required_secs:
        contract += "• REQUIRED SECTIONS (must exist and have at least one <tr> in <tbody>):\n"
        contract += "  " + "; ".join(required_secs) + ".\n"
    # Target total and quotas for selected main sections
    active_main = any([hasNouns, hasVerbs, hasAdj, hasAdv])
    if active_main:
        active_names = []
        if hasNouns: active_names.append("Nouns")
        if hasVerbs: active_names.append("Verbs")
        if hasAdj:   active_names.append("Adjectives")
        if hasAdv:   active_names.append("Adverbs")
        contract += f"• TARGET TOTAL (selected sections): EXACTLY {target_total} Spanish vocabulary items counted by\n"
        contract += "  the number of <span class=\"es\">…</span> target words in " + ", ".join(active_names) + ".\n"
        contract += "• PER-SECTION QUOTAS (enforce exactly):\n"
        if hasNouns: contract += f"  – Nouns: {quotas['n']}\n"
        if hasVerbs: contract += f"  – Verbs: {quotas['v']}\n"
        if hasAdj:   contract += f"  – Adjectives: {quotas['a']}\n"
        if hasAdv:   contract += f"  – Adverbs: {quotas['d']}\n"
    # Common sections
    if hasPhr and hasQ:
        contract += "• COMMON PHRASES & COMMON QUESTIONS — MANDATORY:\n"
        contract += f"  – Populate BOTH sections with table rows inside their existing <tbody>.\n"
        contract += f"  – Number of rows in EACH section: between {max(phrases_min,8)} and 10 inclusive (NEVER exceed 10).\n"
        contract += f"  – Reuse only vocabulary from sections 1–4 (no new vocabulary). Distinct reused words\n"
        contract += f"    across BOTH sections combined must be ≤ {max_reuse} (≈20% of {target_total}).\n"
        contract += f"  – Rows in these sections do NOT count toward the {target_total} total.\n"
    elif hasPhr:
        contract += "• COMMON PHRASES — MANDATORY:\n"
        contract += f"  – Populate the section with table rows inside its existing <tbody>.\n"
        contract += f"  – Number of rows: between {max(phrases_min,8)} and 10 inclusive (NEVER exceed 10).\n"
        contract += f"  – Reuse only vocabulary from sections 1–4 (no new vocabulary). Distinct reused words must be ≤ {max_reuse}.\n"
    elif hasQ:
        contract += "• COMMON QUESTIONS — MANDATORY:\n"
        contract += f"  – Populate the section with table rows inside its existing <tbody>.\n"
        contract += f"  – Number of rows: between {max(questions_min,8)} and 10 inclusive (NEVER exceed 10).\n"
        contract += f"  – Reuse only vocabulary from sections 1–4 (no new vocabulary). Distinct reused words must be ≤ {max_reuse}.\n"
    # Coloring & Linguistics (unchanged)
    contract += "• COLORING & LINGUISTICS:\n"
    contract += "  – Verbs: English cell may color ONLY the verb/particle after “to …” with <span class=\"en\">…</span>; \n"
    contract += "    “is/are going to” must remain black. Spanish cell must color ONLY the infinitive with <span class=\"es\">…</span>. \n"
    contract += "    NEVER color “voy/vas/va/vamos/vais/van a”.\n"
    contract += "  – Adverbs: highlight ONLY the adverb in both columns; do NOT color “is/are going to” (EN) or “va a” (ES).\n"
    contract += "  – Nouns: Spanish uses article; IF a noun commonly has both genders, show masculine first and optionally the feminine \n"
    contract += "    in parentheses — but the parenthetical must NOT be red. The English noun word itself should be blue.\n"
    # Rendering boundaries
    contract += "• RENDERING BOUNDARIES — CRITICAL:\n"
    contract += "  – Use the HTML skeleton from the user's prompt AS-IS (no Markdown, no new sections).\n"
    contract += "  – ONLY populate: " + "; ".join(required_secs) + ".\n"
    contract += "  – Insert ONLY <tr> row content into each existing <tbody>. Do NOT add extra tables or headers.\n"
    contract += "• SELF-CHECK BEFORE SENDING:\n"
    contract += "  – Ensure exact per-section quotas and grand total for selected sections.\n"
    contract += "  – Ensure each required section (if any) has at least one row.\n"
    contract += "  – Ensure well-formed HTML that fits the provided skeleton.\n\n"
    contract += "# REFERENCE GUIDANCE FROM USER (structure only — still render into provided HTML):\n"
    contract += guidance
    return base_system + contract

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

            max_tokens = min(int(os.getenv("MODEL_MAX_TOKENS", "10000")), 16384)

            # Base system message
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
                    quotas_n, quotas_v, quotas_a, quotas_d = quotas_30_30_15_15(target_total)
                    pmin, _ = phrases_questions_row_targets(target_total)

                    counts = verify_vocab_counts(ai_content)
                    if needs_repair(counts, (quotas_n, quotas_v, quotas_a, quotas_d), (pmin, 10)):
                        repair_block = build_repair_prompt(lo, hi, (quotas_n, quotas_v, quotas_a, quotas_d), pmin)
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
