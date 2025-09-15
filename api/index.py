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
# Guidance
# -----------------------
def build_user_guidance_prompt(topic: str, lo: int, hi: int) -> str:
    # ⬇️ Conversation-related guidance preserved here (per your request).
    return f"""You are an expert assistant for the FCS program.
 ...
Always keep vocabulary count between 200–250 distinct Spanish words.
"""

def build_system_message(base_system: str, user_prompt: str) -> str:
    """
    Vocabulary prompt: add strict contract enforcing.
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
STRICT ONE-SHOT COUNTING CONTRACT (Vocabulary ONLY):
...
# REFERENCE GUIDANCE FROM USER (structure only — still render into provided HTML):
{guidance}
"""
    return base_system + contract

# -----------------------
# Post-processing helpers (only for Vocabulary)
# -----------------------

# ✅ keep: ensure_nouns_en_blue_and_parentheses_plain, fix_verbs_highlight, fix_adverbs_highlight
# ✅ keep: verify_vocab_counts, needs_repair, build_repair_prompt
# ✅ keep: _ensure_common_minimum (phrases/questions safety net)

# (functions unchanged — same as your provided code)

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
            data = json.loads(raw.decode("utf-8"))

            prompt = (data.get("prompt") or "").strip()
            if not prompt:
                raise ValueError("Missing 'prompt' in request body.")

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set.")

            client = OpenAI(
                api_key=api_key,
                base_url=OPENAI_BASE_URL or None,
                organization=OPENAI_ORG_ID or None,
            )

            max_tokens = min(int(os.getenv("MODEL_MAX_TOKENS", "10000")), 16384)

            base_system = (
                "You are an expert FCS assistant. Return ONLY full raw HTML. "
                "Strictly follow the embedded contract inside the user's prompt. "
                "ABSOLUTE LENGTH COMPLIANCE rules apply. "
            )

            system_message = build_system_message(base_system, prompt)

            # --- Generate ---
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
            m = FENCE_RE.match(ai_content)
            if m:
                ai_content = m.group(1).strip()

            # Normalize highlights
            ai_content = fix_verbs_highlight(ai_content)
            ai_content = fix_adverbs_highlight(ai_content)
            ai_content = ensure_nouns_en_blue_and_parentheses_plain(ai_content)

            # --- One-shot verify & repair ---
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
                        fixed = fix_verbs_highlight(fixed)
                        fixed = fix_adverbs_highlight(fixed)
                        fixed = ensure_nouns_en_blue_and_parentheses_plain(fixed)
                        ai_content = fixed

                    ai_content = _ensure_common_minimum(ai_content, min_rows=max(8, pmin), max_rows=10)

            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"content": ai_content}).encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self._send_cors_headers()
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
