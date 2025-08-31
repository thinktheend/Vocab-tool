import os, json, re, random
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

# --- Load .env locally (ignored in production if not present)
load_dotenv()

# --- OpenAI SDK (official)
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
# Create client lazily to avoid crash if key missing; we'll check before use.
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # good balance for JSON tasks

app = Flask(__name__, static_folder=".", static_url_path="")

# ----------------- Shared formatting logic (matches front-end) -----------------

def article_a_an(word: str) -> str:
    w = (word or "").strip()
    if not w: return "a"
    return "an" if re.match(r"^[aeiouAEIOU]", w) else "a"

def esc(s: str) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

def sort_by_en(rows: List[Dict[str,str]]) -> List[Dict[str,str]]:
    return sorted(rows, key=lambda r: (r.get("en") or "").lower())

def noun_sentence_en(noun_en: str, idx: int, smart_articles: bool=True) -> str:
    n = (noun_en or "").strip()
    a = article_a_an(n) if smart_articles else "a"
    if idx == 0: return f"Here is the {n}."
    if idx == 1: return f"It is {a} {n}."
    return f"The {n} is here."

def noun_sentence_es(noun_es_or_en: str, idx: int) -> str:
    n = (noun_es_or_en or "").strip()
    if idx == 0: return f"Aquí está el/la {n}."
    if idx == 1: return f"Es un/una {n}."
    return f"El/La {n} está aquí."

def verb_sentence_en(verb_en: str) -> str:
    v = re.sub(r"^\s*to\s+", "", verb_en or "", flags=re.I).strip()
    return f"I am going to {v}."

def verb_sentence_es(verb_es_or_en: str, verb_en: str) -> str:
    v = (verb_es_or_en or "").strip()
    if not v:
        v = re.sub(r"^\s*to\s+", "", verb_en or "", flags=re.I).strip()
    return f"Yo voy a {v}."

def desc_sentence_en(desc_en: str) -> str:
    return f"It is { (desc_en or '').strip() }."

def desc_sentence_es(desc_es_or_en: str) -> str:
    return f"Es { (desc_es_or_en or '').strip() }."

def build_excel_html(payload: Dict[str, Any]) -> str:
    topic = payload.get("topic") or "General"
    level = payload.get("level") or "1"
    nouns = payload.get("nouns") or []
    verbs = payload.get("verbs") or []
    descs = payload.get("descs") or []
    phrases = payload.get("phrases") or []
    questions = payload.get("questions") or []
    opts = payload.get("options") or {}
    smart = bool(opts.get("smartArt", True))
    altp = bool(opts.get("altPat", True))
    sorte = bool(opts.get("sortEn", True))

    if sorte:
        nouns = sort_by_en(nouns)
        verbs = sort_by_en(verbs)
        descs = sort_by_en(descs)
        phrases = sort_by_en(phrases)

    H = []
    H.append('<div class="excel">')
    H.append(f'<h1>{esc(f"Vocabulary: {topic} — Level {level}")}</h1>')

    # Nouns
    H.append('<h2>Nouns / Sustantivos</h2>')
    H.append('<table><thead><tr><th>English</th><th>Español</th></tr></thead><tbody>')
    for i, r in enumerate(nouns):
        en_word = (r.get("en") or r.get("es") or "").strip()
        es_word = (r.get("es") or r.get("en") or "").strip()
        pat = (i % 3) if altp else 0
        en_sent = noun_sentence_en(en_word, pat, smart)
        es_sent = noun_sentence_es(es_word, pat)
        H.append(f'<tr><td><span class="en">{"🔊"+esc(en_sent)}</span></td><td><span class="es">{"🔊"+esc(es_sent)}</span></td></tr>')
    H.append('</tbody></table>')

    # Verbs
    H.append('<h2>Verbs / Verbos</h2>')
    H.append('<table><thead><tr><th>English</th><th>Español</th></tr></thead><tbody>')
    for r in verbs:
        en_inf = (r.get("en") or "").strip()
        es_inf = (r.get("es") or "").strip()
        en_sent = verb_sentence_en(en_inf)
        es_sent = verb_sentence_es(es_inf, en_inf)
        H.append(f'<tr><td><span class="en">{"🔊"+esc(en_sent)}</span></td><td><span class="es">{"🔊"+esc(es_sent)}</span></td></tr>')
    H.append('</tbody></table>')

    # Descriptors
    H.append('<h2>Descriptive Words / Palabras descriptivas</h2>')
    H.append('<table><thead><tr><th>English</th><th>Español</th></tr></thead><tbody>')
    for r in descs:
        en_w = (r.get("en") or r.get("es") or "").strip()
        es_w = (r.get("es") or r.get("en") or "").strip()
        en_sent = desc_sentence_en(en_w)
        es_sent = desc_sentence_es(es_w)
        H.append(f'<tr><td><span class="en">{"🔊"+esc(en_sent)}</span></td><td><span class="es">{"🔊"+esc(es_sent)}</span></td></tr>')
    H.append('</tbody></table>')

    # Phrases
    if phrases:
        H.append('<h2>Common Phrases / Frases comunes</h2>')
        H.append('<table><thead><tr><th>English</th><th>Español</th></tr></thead><tbody>')
        for r in phrases:
            H.append(f'<tr><td><span class="en">{esc(r.get("en") or "")}</span></td><td><span class="es">{esc(r.get("es") or "")}</span></td></tr>')
        H.append('</tbody></table>')

    # Questions
    if questions:
        H.append('<h2>Common Questions / Preguntas comunes</h2>')
        H.append('<table><thead><tr><th>English</th><th>Español</th></tr></thead><tbody>')
        for r in questions:
            en = (r.get("en") or "")
            es = (r.get("es") or "")
            if "|" in en:
                q_en, a_en = [x.strip() for x in en.split("|", 1)]
            else:
                q_en, a_en = en, ""
            if "|" in es:
                q_es, a_es = [x.strip() for x in es.split("|", 1)]
            else:
                q_es, a_es = es, ""
            H.append(
                f'<tr><td><b class="en">{esc(q_en)}</b><br><i class="ans-en">{esc(a_en)}</i></td>'
                f'<td><b class="es">{esc(q_es)}</b><br><i class="ans-es">{esc(a_es)}</i></td></tr>'
            )
        H.append('</tbody></table>')

    H.append('</div>')
    return "\n".join(H)

# Conversations & Tests helpers
def scripted_reading(a: str, b: str, topic: str, turns: int) -> str:
    lines=[]
    t=(topic or "General").lower()
    for i in range(turns):
        s = a if i % 2 == 0 else b
        lead = "¿" if i % 5 == 0 else ("¡" if i % 7 == 0 else "—")
        lines.append(f"{s}: {lead}Hablamos de {t} paso por paso…")
        lines.append(f"(EN) {s}: We talk about {t} step by step…")
    return "\n".join(lines)

def guided_dialogue(a: str, b: str, topic: str, turns: int, mode: str) -> str:
    glue = ['y','pero','porque','entonces','aunque','si','cuando']
    present = ['Tengo que','Necesito','Quiero','Puedo','Voy a','Acabo de','Me gusta']
    past = ['Ayer','La semana pasada','Antes','Anoche','Hace poco']
    lines=[]
    t=(topic or "General").lower()
    for i in range(turns):
        s = a if i % 2 == 0 else b
        if mode == "present":
            es = f"{present[i % len(present)]} repasar {t} {glue[i % len(glue)]} terminar las notas."
            en = "I need to review the topic and finish the notes."
        else:
            es = f"{past[i % len(past)]} {glue[i % len(glue)]} hoy, revisamos {t} y comparamos resultados."
            en = "Yesterday and today, we reviewed the topic and compared results."
        lines.append(f"{s}: {es}")
        lines.append(f"(EN) {s}: {en}")
    return "\n".join(lines)

def make_fitb(topic: str) -> str:
    t=(topic or "General").lower()
    base=[
        f"Aquí está el informe de {t}.",
        "Es un sistema estable.",
        "Yo voy a revisar los datos.",
        "Necesito confirmar los resultados.",
        "El equipo está listo.",
        "Vamos a comparar las mediciones.",
        "Me gusta el diseño simple.",
        "La prueba fue precisa.",
        "El robot está funcionando.",
        "Aquí está la herramienta."
    ]
    out=[]
    for i, s in enumerate(base, start=1):
        words=s.split()
        blanks=max(1,int(len(words)*0.3))
        used=set()
        for _ in range(blanks):
            idx=1
            if len(words)>2:
                while True:
                    idx=random.randint(1,len(words)-2)
                    if idx not in used: used.add(idx); break
            words[idx]='____'
        out.append(f"{i}. {' '.join(words)}")
    out.append("\nAnswer Key:")
    for i, s in enumerate(base, start=1): out.append(f"{i}. {s}")
    return "\n".join(out)

# ----------------- AI helpers -----------------

def need_key() -> str:
    if not OPENAI_API_KEY:
        return "Missing OPENAI_API_KEY environment variable."
    return ""

def call_openai_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Calls the OpenAI responses API expecting a JSON object in the top-level text.
    We use a 'return JSON only' instruction and then parse.
    """
    if not client:
        raise RuntimeError("OpenAI client not initialized")

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.4,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ]
    )
    text = completion.choices[0].message.content.strip()
    # Attempt JSON extraction
    try:
        # In case the model wraps JSON in code fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
        data = json.loads(text)
        return data
    except Exception as e:
        # Try to find a JSON block within
        m = re.search(r"\{[\s\S]*\}$", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        raise RuntimeError(f"AI did not return valid JSON. Raw: {text[:400]}...")

# ----------------- Routes -----------------

@app.get("/")
def root():
    return send_from_directory(Path(".").resolve(), "index.html")

# Deterministic server generation (no AI)
@app.post("/api/generate_vocab")
def api_generate_vocab():
    data = request.get_json(force=True, silent=True) or {}
    html = build_excel_html(data)
    return jsonify({"html": html})

@app.post("/api/generate_conversations")
def api_generate_conversations():
    data = request.get_json(force=True, silent=True) or {}
    topic = data.get("topic") or "General"
    a = data.get("a") or "Ana"
    b = data.get("b") or "Luis"
    level = str(data.get("level") or "1")
    text = ""
    if level == "1":
        text += "LEVEL 1 — Fluent Reading & Pronunciation\nPurpose: reading accuracy (≥90%), natural speed, proper intonation; no production.\n\n"
        text += "Conversation 1\n" + scripted_reading(a,b,topic,14) + "\n\nConversation 2\n" + scripted_reading(a,b,topic,14)
    elif level == "2":
        text += "Level 2 — Present Tense, Reflexives & Glue Words\n\n" + guided_dialogue(a,b,topic,10,"present")
    else:
        text += "Level 3 — Past & Ongoing (mix present + pretérito/imperfecto; some progressive)\n\n" + guided_dialogue(a,b,topic,10,"mixed")
    return jsonify({"text": text})

@app.post("/api/generate_tests")
def api_generate_tests():
    data = request.get_json(force=True, silent=True) or {}
    topic = data.get("topic") or "General"
    mode = data.get("mode") or "read"
    if mode == "read":
        text = "LEVEL 1 — Reading & Pronunciation Test\nInstructions: Read aloud at natural conversational speed. Aim for ≥90% pronunciation accuracy. Use rising intonation for questions.\n\n"
        text += scripted_reading("A","B",topic,16)
    else:
        text = f"Fill-in-the-Blank (30% blanks) — Topic: {topic}\n\n" + make_fitb(topic)
    return jsonify({"text": text})

# ----------------- AI endpoints -----------------

@app.post("/api/ai_vocab")
def api_ai_vocab():
    err = need_key()
    if err:
        return jsonify({"error": err}), 400

    data = request.get_json(force=True, silent=True) or {}
    topic = data.get("topic") or "General"
    level = str(data.get("level") or "1")
    sort_en = bool((data.get("options") or {}).get("sortEn", True))

    system = (
        "You are an assistant for a Spanish-learning app (FCS). "
        "Return STRICT JSON only. No commentary."
    )
    user = f"""
Task: Create structured JSON vocabulary for topic "{topic}" and level {level}.
Rules (must follow all):
- Output JSON with keys exactly: nouns, verbs, descs, phrases, questions.
- nouns: 24–40 items, each {{ "en": <english noun lower case, singular>, "es": <spanish noun lower case, singular> }}
- verbs: 18–30 items, each {{ "en": "to <base infinitive in english>", "es": "<spanish infinitive>" }}
- descs: 15–25 items, adjectives/adverbs only, each {{ "en": "<english>", "es": "<spanish>" }}
- phrases: 10–16 items, each {{ "en": "<english phrase>", "es": "<spanish phrase>" }}
- questions: 8–14 items, each: {{ "en": "Question? | Sample answer.", "es": "¿Pregunta? | Respuesta de ejemplo." }}
Constraints from FCS:
- Cover the topic comprehensively but stick to level-appropriate, high-frequency items.
- No extra keys, no trailing text.
Return JSON ONLY.
"""
    try:
        j = call_openai_json(system, user)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # optional sorting
    if sort_en:
        for k in ["nouns","verbs","descs","phrases","questions"]:
            if isinstance(j.get(k), list):
                j[k] = sorted(j[k], key=lambda r: (r.get("en") or "").lower())

    html = build_excel_html({
        "topic": topic,
        "level": level,
        "nouns": j.get("nouns", []),
        "verbs": j.get("verbs", []),
        "descs": j.get("descs", []),
        "phrases": j.get("phrases", []),
        "questions": j.get("questions", []),
        "options": {"smartArt": True, "altPat": True, "sortEn": sort_en}
    })
    return jsonify({"html": html, "debug": j})

@app.post("/api/ai_conversations")
def api_ai_conversations():
    err = need_key()
    if err:
        return jsonify({"error": err}), 400

    data = request.get_json(force=True, silent=True) or {}
    topic = data.get("topic") or "General"
    a = data.get("a") or "Ana"
    b = data.get("b") or "Luis"
    level = str(data.get("level") or "1")

    system = "You are a conversation generator for FCS. Return RAW TEXT ONLY. No code fences."
    if level == "1":
        user = f"""
Level 1 — Fluent Reading & Pronunciation (scripted reading only, NO production).
Topic: {topic}
Speakers: {a}, {b}
Requirements:
- Produce exactly two sections: "Conversation 1" and "Conversation 2".
- Each conversation ~14 turns (one line per turn), alternating speakers.
- Use clear, natural Spanish with full coverage of topic vocabulary; include intonation cues (¿, ¡, …, —) where natural.
- Under each Spanish line, add an English line prefixed with "(EN) <speaker>: ..."
- At top, include:
  LEVEL 1 — Fluent Reading & Pronunciation
  Purpose: reading accuracy (≥90%), natural speed, proper intonation; no production.
Return plain text only.
"""
    elif level == "2":
        user = f"""
Level 2 — Present Tense, Reflexives & Glue Words.
Topic: {topic}
Speakers: {a}, {b}
Requirements:
- ~10 turns; alternate speakers.
- Use present tense with these frames naturally: tener que, necesitar, querer, ir a, poder, acabar de, gustar; include object pronouns and reflexives where natural; glue words: y, pero, porque, entonces, aunque, si, cuando.
- Under each Spanish line, add an English line prefixed with "(EN) <speaker>: ..."
Return plain text only.
"""
    else:
        user = f"""
Level 3 — Past & Ongoing.
Topic: {topic}
Speakers: {a}, {b}
Requirements:
- ~10 turns; alternate speakers.
- Mix present with pretérito/imperfecto; some progressive forms; avoid imperatives/subjunctive.
- Under each Spanish line, add an English line prefixed with "(EN) <speaker>: ..."
Return plain text only.
"""

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0.5,
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )
        text = completion.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"text": text})

@app.post("/api/ai_tests")
def api_ai_tests():
    err = need_key()
    if err:
        return jsonify({"error": err}), 400

    data = request.get_json(force=True, silent=True) or {}
    topic = data.get("topic") or "General"
    mode = data.get("mode") or "read"

    system = "You are a test generator for FCS. Return RAW TEXT ONLY."
    if mode == "read":
        user = f"""
Create a Level 1 Reading & Pronunciation Test for topic "{topic}".
- Include header: LEVEL 1 — Reading & Pronunciation Test
- Instructions: Read aloud at natural conversational speed; aim for ≥90% pronunciation accuracy; use rising intonation for questions.
- Then provide a single scripted passage of ~16 turns, alternating speakers A and B, each with the English translation beneath as (EN) A: ...
Return plain text only.
"""
    else:
        user = f"""
Create a Fill-in-the-Blank test for topic "{topic}".
- Header: Fill-in-the-Blank (30% blanks) — Topic: {topic}
- 10 lines of Spanish statements; blank about 30% of words with "____".
- Include an Answer Key after the 10 lines, with the full original lines.
Return plain text only.
"""

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0.4,
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )
        text = completion.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"text": text})

# ----------------- Dev server -----------------

if __name__ == "__main__":
    # For local dev: http://localhost:8000
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
