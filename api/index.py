# index.py
# Serverless handler for Vercel (/api/index) that calls OpenAI to generate HTML
# for the FCS AI Generator Suite. This version adds *one-shot* count enforcement
# for the Vocabulary Generator so the number of red-colored Spanish vocabulary
# items (sections 1–4 only) hits the exact midpoint of the user-selected range.

import os
import re
import json
from http.server import BaseHTTPRequestHandler
from openai import OpenAI

# Optional overrides via env
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")

# --- Regex helpers -----------------------------------------------------------

# Unwrap code fences if a provider ever adds them.
FENCE_RE = re.compile(
    r"^\s*```(?:html|xml|markdown)?\s*([\s\S]*?)\s*```\s*$",
    re.IGNORECASE,
)

# Detect which generator sent the prompt
IS_VOCAB_RE = re.compile(r"FCS\s+VOCABULARY\s+OUTPUT", re.IGNORECASE)
IS_CONVO_RE = re.compile(r"FCS\s+CONVERSATION\s+OUTPUT", re.IGNORECASE)

# Pull min/max vocabulary range out of the embedded Markdown prompt
# Handles hyphen, en dash, em dash, figure dash, etc.
RANGE_RE = re.compile(
    r"Vocabulary\s+range:\s*(\d+)\s*[\-\u2010-\u2015\u2212]\s*(\d+)",
    re.IGNORECASE,
)

# -----------------------------------------------------------------------------


def parse_vocab_range(prompt_text: str) -> tuple[int | None, int | None]:
    """
    Find "Vocabulary range: X–Y ..." inside the HTML comment block of the Vocab prompt.
    Returns (min, max) or (None, None) if not present.
    """
    m = RANGE_RE.search(prompt_text or "")
    if not m:
        return (None, None)
    try:
        lo = int(m.group(1))
        hi = int(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)
    except Exception:
        return (None, None)


def midpoint(lo: int, hi: int) -> int:
    """
    Exact midpoint (rounded down) within [lo, hi].
    """
    return max(lo, min(hi, (lo + hi) // 2))


def quota_30_30_15_15(total: int) -> tuple[int, int, int, int]:
    """
    Split 'total' into 30% nouns, 30% verbs, 15% adjectives, 15% adverbs.
    Any rounding remainder is distributed Nouns->Verbs->Adjectives->Adverbs.
    """
    n = int(round(total * 0.30))
    v = int(round(total * 0.30))
    a = int(round(total * 0.15))
    d = int(round(total * 0.15))
    # Fix rounding drift (could be +/- a couple)
    diff = total - (n + v + a + d)
    buckets = ["n", "v", "a", "d"]
    i = 0
    while diff != 0:
        if diff > 0:
            if buckets[i] == "n":
                n += 1
            elif buckets[i] == "v":
                v += 1
            elif buckets[i] == "a":
                a += 1
            else:
                d += 1
            diff -= 1
        else:
            if buckets[i] == "n" and n > 0:
                n -= 1
                diff += 1
            elif buckets[i] == "v" and v > 0:
                v -= 1
                diff += 1
            elif buckets[i] == "a" and a > 0:
                a -= 1
                diff += 1
            elif buckets[i] == "d" and d > 0:
                d -= 1
                diff += 1
        i = (i + 1) % 4
    return n, v, a, d


def build_system_message(is_vocab: bool, target_total: int | None,
                         quotas: tuple[int, int, int, int] | None) -> str:
    """
    Compose the system message. If it's a vocabulary prompt and we could derive
    the midpoint targets, we append a strict counting contract that the model
    must satisfy *in one shot*.
    """
    base = (
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
        "  Spanish cell replaces the target Spanish word with \"________\" and puts the English translation "
        "  in parentheses immediately BEFORE the blank. "
        "Common Phrases/Questions must follow the contract. "
        "Do NOT add explanations or code fences."
    )

    if not is_vocab or target_total is None or quotas is None:
        return base

    n, v, a, d = quotas

    # Extra, vocabulary-only strict counting contract. We do not change tables/UI;
    # we only instruct the model to hit exact totals on the first try.
    strict = (
        "\n\nCRITICAL COUNTING CONTRACT (Vocabulary ONLY — one-shot):\n"
        f"• TARGET TOTAL: Across sections 1–4 (Nouns, Verbs, Adjectives, Adverbs), the number of "
        f"red Spanish vocabulary items (i.e., <span class=\"es\">…</span> occurrences that mark the target words) "
        f"MUST equal EXACTLY {target_total}. No more, no fewer.\n"
        "• Do NOT put <span class=\"es\">…</span> in Common Phrases or Common Questions unless the original contract already requires it; "
        "only sections 1–4 are counted toward this total.\n"
        "• No duplicate target words within a section; prefer unique items.\n"
        "• Per-section quotas (enforce exactly):\n"
        f"  – Nouns: {n}\n"
        f"  – Verbs: {v}\n"
        f"  – Adjectives: {a}\n"
        f"  – Adverbs: {d}\n"
        "• Internally self-check the exact count of <span class=\"es\">…</span> target words in sections 1–4 before returning the final HTML. "
        "Respond once; do not apologize or say you will retry."
    )
    return base + strict


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

            # Detect which generator is calling
            is_vocab = bool(IS_VOCAB_RE.search(prompt))
            is_convo = bool(IS_CONVO_RE.search(prompt))

            # Try to derive a midpoint target for the vocabulary generator
            target_total = None
            quotas = None
            if is_vocab:
                lo, hi = parse_vocab_range(prompt)
                if lo is not None and hi is not None:
                    target_total = midpoint(lo, hi)  # exact midpoint (one-shot)
                    quotas = quota_30_30_15_15(target_total)

            # Build system message (adds strict counting contract for vocab, otherwise the default)
            system_message = build_system_message(is_vocab, target_total, quotas)

            # Larger budget to avoid truncation under strict length rules.
            max_tokens = min(int(os.getenv("MODEL_MAX_TOKENS", "7000")), 16384)

            # Choose model (default GPT-4o unless overridden)
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

            # Decoding settings tuned for one-shot compliance & diversity without retries
            # Lower temperature for determinism; frequency penalty discourages repeats.
            temperature = 0.25 if is_vocab else 0.6
            frequency_penalty = 0.7 if is_vocab else 0.0
            presence_penalty = 0.0

            completion = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
            )

            ai_content = (completion.choices[0].message.content or "").strip()
            m = FENCE_RE.match(ai_content)
            if m:
                ai_content = m.group(1).strip()

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
            self.wfile.write(
                json.dumps(
                    {
                        "error": "An internal server error occurred.",
                        "details": str(e),
                    }
                ).encode("utf-8")
            )
