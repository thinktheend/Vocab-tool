from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import json
import re

app = Flask(__name__, static_folder=".", static_url_path="")

# ---------- Utilities & shared generation logic ----------

def article_a_an(word: str) -> str:
    w = (word or "").strip()
    if not w:
        return "a"
    return "an" if re.match(r"^[aeiouAEIOU]", w) else "a"

def esc(s: str) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

def sort_by_en(rows):
    return sorted(rows, key=lambda r: (r.get("en") or "").lower())

# Noun sentence rules (English + Spanish)
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

# Verb sentence rules (infinitive focus)
def verb_sentence_en(verb_en: str) -> str:
    v = re.sub(r"^\s*to\s+", "", verb_en or "", flags=re.I).strip()
    return f"I am going to {v}."

def verb_sentence_es(verb_es_or_en: str, verb_en: str) -> str:
    v = (verb_es_or_en or "").strip()
    if not v:
        v = re.sub(r"^\s*to\s+", "", verb_en or "", flags=re.I).strip()
    return f"Yo voy a {v}."

# Descriptive sentence rules
def desc_sentence_en(desc_en: str) -> str:
    return f"It is { (desc_en or '').strip() }."

def desc_sentence_es(desc_es_or_en: str) -> str:
    return f"Es { (desc_es_or_en or '').strip() }."

def build_excel_html(payload: dict) -> str:
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

    # Nouns section
    H.append('<h2>Nouns / Sustantivos</h2>')
    H.append('<table><thead><tr><th>English</th><th>Español</th></tr></thead><tbody>')
    for i, r in enumerate(nouns):
        en_word = (r.get("en") or r.get("es") or "").strip()
        es_word = (r.get("es") or r.get("en") or "").strip()
        pat = (i % 3) if altp else 0
        en_sent = noun_sentence_en(en_word, pat, smart)
        es_sent = noun_sentence_es(es_word, pat)
        H.append(f'<tr><td><span class="en">{esc("🔊"+en_sent)}</span></td><td><span class="es">{esc("🔊"+es_sent)}</span></td></tr>')
    H.append('</tbody></table>')

    # Verbs section
    H.append('<h2>Verbs / Verbos</h2>')
    H.append('<table><thead><tr><th>English</th><th>Español</th></tr></thead><tbody>')
    for r in verbs:
        en_inf = (r.get("en") or "").strip()
        es_inf = (r.get("es") or "").strip()
        en_sent = verb_sentence_en(en_inf)
        es_sent = verb_sentence_es(es_inf, en_inf)
        H.append(f'<tr><td><span class="en">{esc("🔊"+en_sent)}</span></td><td><span class="es">{esc("🔊"+es_sent)}</span></td></tr>')
    H.append('</tbody></table>')

    # Descriptive words
    H.append('<h2>Descriptive Words / Palabras descriptivas</h2>')
    H.append('<table><thead><tr><th>English</th><th>Español</th></tr></thead><tbody>')
    for r in descs:
        en_w = (r.get("en") or r.get("es") or "").strip()
        es_w = (r.get("es") or r.get("en") or "").strip()
        en_sent = desc_sentence_en(en_w)
        es_sent = desc_sentence_es(es_w)
        H.append(f'<tr><td><span class="en">{esc("🔊"+en_sent)}</span></td><td><span class="es">{esc("🔊"+es_sent)}</span></td></tr>')
    H.append('</tbody></table>')

    # Phrases
    if phrases:
        H.append('<h2>Common Phrases / Frases comunes</h2>')
        H.append('<table><thead><tr><th>English</th><th>Español</th></tr></thead><tbody>')
        for r in phrases:
            H.append(f'<tr><td><span class="en">{esc(r.get("en") or "")}</span></td><td><span class="es">{esc(r.get("es") or "")}</span></td></tr>')
        H.append('</tbody></table>')

    # Questions (Q bold, A italic under it)
    if questions:
        H.append('<h2>Common Questions / Preguntas comunes</h2>')
        H.append('<table><thead><tr><th>English</th><th>Español</th></tr></thead><tbody>')
        for r in questions:
            en = (r.get("en") or "")
            es = (r.get("es") or "")
            q_en, a_en = [x.strip() for x in en.split("|")] if "|" in en else (en, "")
            q_es, a_es = [x.strip() for x in es.split("|")] if "|" in es else (es, "")
            H.append(
                f'<tr><td><b class="en">{esc(q_en)}</b><br><i class="ans-en">{esc(a_en)}</i></td>'
                f'<td><b class="es">{esc(q_es)}</b><br><i class="ans-es">{esc(a_es)}</i></td></tr>'
            )
        H.append('</tbody></table>')

    H.append('</div>')
    return "\n".join(H)

# Conversations
def scripted_reading(a: str, b: str, topic: str, turns: int) -> str:
    lines = []
    t = (topic or "General").lower()
    for i in range(turns):
        s = a if i % 2 == 0 else b
        lead = "¿" if i % 5 == 0 else ("¡" if i % 7 == 0 else "—")
        lines.append(f"{s}: {lead}Hablamos de {t} paso por paso…")
        lines.append(f"(EN) {s}: We talk about {t} step by step…")
    return "\n".join(lines)

def guided_dialogue(a: str, b: str, topic: str, turns: int, mode: str) -> str:
    glue = ['y','pero','porque','entonces','aunque','si','cuando']
    present_frames = ['Tengo que','Necesito','Quiero','Puedo','Voy a','Acabo de','Me gusta']
    past_frames = ['Ayer','La semana pasada','Antes','Anoche','Hace poco']
    lines = []
    t = (topic or "General").lower()
    for i in range(turns):
      s = a if i % 2 == 0 else b
      if mode == "present":
          es = f"{present_frames[i % len(present_frames)]} repasar {t} {glue[i % len(glue)]} terminar las notas."
          en = "I need to review the topic and finish the notes."
      else:
          es = f"{past_frames[i % len(past_frames)]} {glue[i % len(glue)]} hoy, revisamos {t} y comparamos resultados."
          en = "Yesterday and today, we reviewed the topic and compared results."
      lines.append(f"{s}: {es}")
      lines.append(f"(EN) {s}: {en}")
    return "\n".join(lines)

# Tests
def make_fitb(topic: str) -> str:
    t = (topic or "General").lower()
    base = [
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
    out = []
    import random
    for i, s in enumerate(base, start=1):
        words = s.split()
        blanks = max(1, int(len(words) * 0.3))
        used = set()
        for _ in range(blanks):
            idx = 1
            if len(words) > 2:
                while True:
                    idx = random.randint(1, len(words)-2)
                    if idx not in used:
                        used.add(idx)
                        break
            words[idx] = "____"
        out.append(f"{i}. {' '.join(words)}")
    out.append("\nAnswer Key:")
    for i, s in enumerate(base, start=1):
        out.append(f"{i}. {s}")
    return "\n".join(out)

# ---------- Routes ----------

@app.route("/")
def root():
    # Serve the front-end file
    return send_from_directory(Path(".").resolve(), "index.html")

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

if __name__ == "__main__":
    # For local development
    app.run(host="0.0.0.0", port=8000, debug=True)
