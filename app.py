import os
import json
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = Flask(__name__, static_folder="public")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY tidak ditemukan di .env")

genai.configure(api_key=api_key)

# ── Model Configuration ──
GEMINI_MODEL = "gemini-2.5-flash-lite"

# ── System prompts per level ──
SYSTEM_PROMPTS = {
    "beginner": """Kamu adalah AI Tutor yang ramah dan sabar untuk pemula.
ATURAN JAWABAN:
ATURAN PENTING:
- Jika pertanyaan adalah basa-basi, sapaan, atau percakapan umum (contoh: "hi", "halo", "how are you", "siapa kamu") → jawab singkat dan natural seperti manusia, TANPA struktur apapun.
- Jika pertanyaan adalah topik edukasi/pembelajaran → gunakan struktur di bawah.
- Gunakan bahasa yang sangat sederhana, hindari istilah teknis
- Jika terpaksa pakai istilah teknis, selalu jelaskan artinya
- Berikan analogi sehari-hari agar mudah dipahami
- Struktur jawaban WAJIB:
  📌 **Penjelasan Singkat:** (1-2 kalimat simpel)
  📖 **Penjelasan Lengkap:** (mudah dipahami, pakai analogi)
  💡 **Contoh Nyata:** (contoh kehidupan sehari-hari)
  ✅ **Poin Penting:** (maksimal 3 poin simpel)
- Jawaban maksimal 350 kata
- Hindari paragraf panjang
- Akhiri dengan kalimat semangat
- IMPORTANT: Always respond in English only, regardless of the language the user uses.""",

    "intermediate": """Kamu adalah AI Tutor untuk pelajar tingkat menengah.

    ATURAN PENTING:
- Jika pertanyaan adalah basa-basi, sapaan, atau percakapan umum → jawab singkat dan natural, TANPA struktur apapun.
- Jika pertanyaan adalah topik edukasi/pembelajaran → gunakan struktur di bawah.
ATURAN JAWABAN:
- Gunakan bahasa yang jelas dan terstruktur
- Boleh pakai istilah teknis tapi tetap dijelaskan
- Hubungkan konsep dengan pengetahuan yang sudah umum diketahui
- Struktur jawaban WAJIB:
  🎯 **Konsep Utama:** (definisi dan inti)
  📚 **Penjelasan Detail:** (mendalam tapi tetap jelas)
  🔗 **Keterkaitan:** (hubungan dengan konsep lain)
  💡 **Contoh & Aplikasi:** (contoh praktis)
  📝 **Rangkuman:** (poin-poin kunci)
  - Jawaban sekitar 400-700 kata
- IMPORTANT: Always respond in English only, regardless of the language the user uses.""",

    "advanced": """Kamu adalah AI Tutor untuk pelajar tingkat lanjut.
    ATURAN PENTING:
- Jika pertanyaan adalah basa-basi, sapaan, atau percakapan umum → jawab singkat dan natural, TANPA struktur apapun.
- Jika pertanyaan adalah topik edukasi/pembelajaran → gunakan struktur di bawah.

ATURAN JAWABAN:
- Gunakan bahasa akademis dan teknis
- Berikan penjelasan mendalam dan komprehensif
- Sertakan nuansa, pengecualian, dan perspektif kritis
- Struktur jawaban WAJIB:
  🔬 **Definisi & Teori:** (formal dan presisi)
  ⚙️ **Analisis Mendalam:** (mekanisme, prinsip, teori pendukung)
  🔄 **Perspektif Kritis:** (pro/kontra, limitasi, debat akademis)
  📊 **Contoh Kompleks:** (kasus nyata atau studi kasus)
  🔭 **Implikasi & Pengembangan:** (relevansi lanjut, penelitian terkini)
- Jawaban boleh panjang dan mendalam
- Minimal 700 kata jika topik kompleks
- IMPORTANT: Always respond in English only, regardless of the language the user uses."""
}

# ── Chat sessions per level ──
chat_sessions = {}

def get_chat(level="intermediate"):
    if level not in chat_sessions:
        model = genai.GenerativeModel(GEMINI_MODEL)
        chat_sessions[level] = model.start_chat(history=[
            {"role": "user",  "parts": [SYSTEM_PROMPTS[level]]},
            {"role": "model", "parts": ["Understood. I'm ready to be your tutor. Please ask your question."]},
        ])
    return chat_sessions[level]


# ── Static files ──
@app.route("/")
def index():
    return send_from_directory("public", "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("public", filename)


# ── Chat endpoint ──
@app.post("/ask")
def ask():
    data      = request.get_json()
    question  = data.get("question", "").strip()
    question = f"[Respond in English only] {question}"
    level     = data.get("level", "intermediate")
    do_stream = data.get("stream", False)

    if not question:
        return jsonify({"answer": "Pertanyaan tidak boleh kosong."}), 400

    if level not in SYSTEM_PROMPTS:
        level = "intermediate"

    chat = get_chat(level)

    try:
        if do_stream:
            def generate():
                response = chat.send_message(question, stream=True)
                for chunk in response:
                    if chunk.text:
                        yield f"data: {json.dumps({'text': chunk.text})}\n\n"
                yield "data: [DONE]\n\n"

            return Response(
                stream_with_context(generate()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            )
        else:
            response = chat.send_message(question)
            return jsonify({"answer": response.text})

    except Exception as e:
        print(f"ERROR /ask: {e}")
        return jsonify({"answer": "Terjadi kesalahan di server. Silakan coba lagi."}), 500

# ── Quiz endpoint ──
@app.post("/quiz")
def quiz():

    data  = request.get_json()

    topic = data.get("topic", "").strip()
    count = int(data.get("count", 10))
    level = data.get("level", "intermediate")

    if not topic:
        return jsonify({
            "error": "Topik quiz tidak boleh kosong."
        }), 400

    # =====================================================
    # BLOCKED / NON-EDUCATIONAL TOPICS
    # =====================================================

    blocked_topics = [
        "bom",
        "narkoba",
        "hack akun",
        "membunuh",
        "dark web",
        "penipuan",
        "judi"
    ]

    lower_topic = topic.lower()

    for bad in blocked_topics:
        if bad in lower_topic:
            return jsonify({
                "error": "Topik tidak diperbolehkan."
            }), 400

    # =====================================================
    # LEVEL RULES
    # =====================================================

    level_rules = {

        "beginner": """
- gunakan bahasa sederhana
- fokus konsep dasar
- hindari istilah teknis rumit
- soal harus mudah dipahami
- gunakan contoh sehari-hari
- fokus definisi dan pemahaman dasar
- cocok untuk pemula
- When writing mathematical formulas, ALWAYS use proper LaTeX notation
- Inline math: $formula$ — example: $P(A|B)$
- Block/display math: $$formula$$ — example: $$P(A|B) = \\frac{P(B|A)P(A)}{P(B)}$$
- NEVER mix plain text inside LaTeX delimiters $...$
- NEVER write math notation without LaTeX (no unicode like 𝑃, 𝐴, 𝐵)
- Always close every $ delimiter properly
""",

        "intermediate": """
- gunakan konsep menengah
- mulai gunakan istilah teknis ringan
- kombinasikan teori dan aplikasi
- soal boleh berbentuk analisis sederhana
- butuh pemahaman konsep
- tingkat kesulitan sedang
- When writing mathematical formulas, ALWAYS use proper LaTeX notation
- Inline math: $formula$ — example: $P(A|B)$
- Block/display math: $$formula$$ — example: $$P(A|B) = \\frac{P(B|A)P(A)}{P(B)}$$
- NEVER mix plain text inside LaTeX delimiters $...$
- NEVER write math notation without LaTeX (no unicode like 𝑃, 𝐴, 𝐵)
- Always close every $ delimiter properly
""",

        "advanced": """
- gunakan analisis mendalam
- boleh menggunakan istilah teknis
- fokus reasoning dan problem solving
- gunakan studi kasus
- soal harus menantang
- evaluasi pemahaman konseptual mendalam
- When writing mathematical formulas, ALWAYS use proper LaTeX notation
- Inline math: $formula$ — example: $P(A|B)$
- Block/display math: $$formula$$ — example: $$P(A|B) = \\frac{P(B|A)P(A)}{P(B)}$$
- NEVER mix plain text inside LaTeX delimiters $...$
- NEVER write math notation without LaTeX (no unicode like 𝑃, 𝐴, 𝐵)
- Always close every $ delimiter properly
"""
    }

    level_desc = level_rules.get(
        level,
        level_rules["intermediate"]
    )

    # =====================================================
    # FEW-SHOT EXAMPLE
    # =====================================================

    few_shot_example = """
Contoh format yang WAJIB diikuti:

[
  {
    "q": "Apa yang dimaksud dengan fotosintesis?",
    "options": [
      "A. Proses pembuatan makanan oleh tumbuhan menggunakan cahaya",
      "B. Proses pernapasan tumbuhan",
      "C. Proses penyerapan air oleh akar",
      "D. Proses pertumbuhan daun baru"
    ],
    "answer": 0,
    "explanation": "Fotosintesis adalah proses tumbuhan membuat makanan dari cahaya matahari."
  },
  {
    "q": "Zat apa yang dihasilkan dari fotosintesis?",
    "options": [
      "A. Karbon dioksida dan air",
      "B. Glukosa dan oksigen",
      "C. Nitrogen dan hidrogen",
      "D. Protein dan lemak"
    ],
    "answer": 1,
    "explanation": "Fotosintesis menghasilkan glukosa dan oksigen."
  }
]
"""

    # =====================================================
    # MAIN PROMPT
    # =====================================================

    prompt = f"""
Kamu adalah AI pembuat soal ujian profesional.

=========================================================
TOPIK QUIZ
=========================================================

{topic}

=========================================================
LEVEL PENGGUNA
=========================================================

{level}

=========================================================
ATURAN LEVEL
=========================================================

{level_desc}

=========================================================
ATURAN QUIZ
=========================================================

- Buat {count} soal pilihan ganda
- Soal harus edukatif
- Soal harus relevan dengan topik
- Hindari soal ambigu
- Hindari pertanyaan terlalu pendek
- Gunakan bahasa yang jelas

=========================================================
VARIASI SOAL
=========================================================

Distribusi soal harus beragam:

- minimal 30% konsep dasar
- minimal 30% aplikasi
- minimal 20% analisis
- sisanya studi kasus atau reasoning

=========================================================
VALIDASI FORMAT
=========================================================

Setiap soal HARUS memiliki:

- q
- options
- answer
- explanation

Rules:
- options harus tepat 4
- answer HARUS angka 0-3
- explanation wajib diisi
- semua soal harus berbeda
- jangan duplikat pertanyaan

=========================================================
FEW-SHOT EXAMPLE
=========================================================

{few_shot_example}

=========================================================
OUTPUT
=========================================================

HANYA balas dengan JSON ARRAY.
TANPA markdown.
TANPA penjelasan tambahan.
TANPA teks lain.
All question text, options, and explanations MUST be in English.
"""

    try:

        model = genai.GenerativeModel(GEMINI_MODEL)

        response = model.generate_content(prompt)

        raw = response.text.strip()

        if raw.startswith("```json"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        questions = json.loads(raw)

        # =================================================
        # BASIC VALIDATION
        # =================================================

        validated = []

        for q in questions:

            if (
                "q" in q and
                "options" in q and
                "answer" in q and
                "explanation" in q and
                len(q["options"]) == 4 and
                isinstance(q["answer"], int)
            ):
                validated.append(q)

        return jsonify({
            "questions": validated
        })

    except Exception as e:

        print(f"ERROR /quiz: {e}")

        return jsonify({
            "error": "Gagal membuat quiz."
        }), 500


# =========================================================
# STUDY PLAN GENERATOR
# =========================================================

@app.post("/studyplan")
def generate_studyplan():
    data     = request.get_json()
    topic    = data.get("topic", "").strip()
    goal     = data.get("goal", "").strip()
    level    = data.get("level", "intermediate")
    duration = data.get("duration", "1month")

    if not topic:
        return jsonify({"error": "Topik tidak boleh kosong."}), 400

    # ── Konfigurasi per durasi ──
    duration_config = {
        "1week": {
            "label":      "1 Minggu",
            "unit":       "hari",
            "unit_en":    "day",
            "count":      7,
            "key":        "days",
            "desc":       "7 hari belajar harian yang intensif",
            "json_key":   "day",
        },
        "1month": {
            "label":      "1 Bulan",
            "unit":       "minggu",
            "unit_en":    "week",
            "count":      4,
            "key":        "weeks",
            "desc":       "4 minggu belajar mingguan",
            "json_key":   "week",
        },
        "3months": {
            "label":      "3 Bulan",
            "unit":       "bulan",
            "unit_en":    "month",
            "count":      3,
            "key":        "months",
            "desc":       "3 bulan belajar terstruktur per bulan",
            "json_key":   "month",
        },
        "6months": {
            "label":      "6 Bulan",
            "unit":       "bulan",
            "unit_en":    "month",
            "count":      6,
            "key":        "months",
            "desc":       "6 bulan belajar komprehensif per bulan",
            "json_key":   "month",
        },
        "1year": {
            "label":      "1 Tahun",
            "unit":       "bulan",
            "unit_en":    "month",
            "count":      12,
            "key":        "months",
            "desc":       "12 bulan belajar jangka panjang per bulan",
            "json_key":   "month",
        },
    }

    cfg = duration_config.get(duration, duration_config["1month"])

    prompt = f"""Kamu adalah AI Study Planner profesional.

Buat study plan untuk:
- Topik: {topic}
- Tujuan: {goal if goal else "Memahami dan menguasai topik secara menyeluruh"}
- Level: {level}
- Durasi: {cfg["desc"]}
- Jumlah unit: {cfg["count"]} {cfg["unit"]}

ATURAN LEVEL:
- beginner: fokus dasar, ringan, pengenalan konsep, aktivitas sederhana
- intermediate: teori + praktik, analisis ringan, istilah teknis ringan
- advanced: analisis mendalam, project based, studi kasus, teknis

ATURAN PENTING:
- Jumlah item dalam "{cfg["key"]}" WAJIB tepat {cfg["count"]}
- Setiap {cfg["unit"]} HARUS berbeda fokusnya
- Tingkat kesulitan meningkat bertahap
- Setiap unit harus punya: tema, tujuan, topik, aktivitas, resource
- All content (title, overview, theme, goals, topics, activities, resources, tips) MUST be in English.

FORMAT JSON WAJIB (HANYA JSON, tanpa teks lain):
{{
  "title": "judul study plan",
  "overview": "ringkasan singkat 1-2 kalimat",
  "duration_label": "{cfg["label"]}",
  "unit": "{cfg["unit"]}",
  "{cfg["key"]}": [
    {{
      "{cfg["json_key"]}": 1,
      "theme": "tema {cfg["unit"]} ini",
      "goals": ["tujuan 1", "tujuan 2"],
      "topics": ["topik 1", "topik 2", "topik 3"],
      "activities": ["aktivitas 1", "aktivitas 2"],
      "resources": ["resource 1", "resource 2"],
      "hours": 5
    }}
  ],
  "tips": ["tips 1", "tips 2", "tips 3"]
}}"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        raw      = response.text.strip().replace("```json", "").replace("```", "").strip()
        plan     = json.loads(raw)

        # Normalize: pastikan key "weeks" selalu ada di response untuk frontend
        # Frontend pakai plan.weeks, jadi mapping ke sana
        if cfg["key"] != "weeks" and cfg["key"] in plan:
            plan["weeks"] = plan[cfg["key"]]

        return jsonify({"plan": plan})

    except Exception as e:
        print(f"ERROR /studyplan: {e}")
        return jsonify({"error": "Gagal membuat study plan."}), 500

# =========================================================
# ANSWER EVALUATOR
# =========================================================

@app.post("/evaluate")
def evaluate():

    data = request.get_json()

    question = data.get("question", "").strip()
    answer   = data.get("answer", "").strip()
    level    = data.get("level", "intermediate")
    role     = data.get("role", "guru SMA")
    subject  = data.get("subject", "Umum")

    if not question or not answer:
        return jsonify({
            "error": "Question dan answer wajib diisi."
        }), 400

    prompt = f"""
Kamu adalah seorang {role} yang bertugas mengevaluasi jawaban pelajar secara profesional, objektif, edukatif, dan adaptif.

=========================================================
INFORMASI PELAJAR
=========================================================

KONTEKS / MATA PELAJARAN:
{subject}

PERTANYAAN:
{question}

JAWABAN PELAJAR:
{answer}

LEVEL PELAJAR:
{level}

ROLE EVALUATOR:
{role}

=========================================================
ATURAN PENILAIAN BERDASARKAN LEVEL
=========================================================

BEGINNER:
- fokus pada pemahaman dasar
- toleransi kesalahan kecil
- gunakan feedback sederhana
- feedback harus suportif
- hindari kritik terlalu keras
- gunakan bahasa ringan dan mudah dipahami

INTERMEDIATE:
- fokus pada ketepatan konsep
- evaluasi struktur jawaban
- harapkan penjelasan lebih lengkap
- gunakan feedback edukatif
- mulai nilai kualitas argumentasi

ADVANCED:
- fokus analisis mendalam
- evaluasi argumentasi
- evaluasi kualitas penjelasan
- evaluasi ketepatan akademik
- gunakan kritik konstruktif yang detail
- nilai kualitas analisis dan logika

=========================================================
ATURAN ROLE EVALUATOR
=========================================================

GURU SD:
- bahasa sangat sederhana
- nada ramah
- banyak motivasi
- fokus membangun rasa percaya diri pelajar

GURU SMP:
- edukatif dan ringan
- mulai membangun disiplin belajar
- gunakan penjelasan yang jelas dan terarah

GURU SMA:
- formal akademik standar sekolah
- objektif dan terstruktur
- evaluasi konsep dan logika dasar

DOSEN UNIVERSITAS:
- analitis
- lebih kritis
- fokus logika, argumentasi, dan kualitas penjelasan
- gunakan gaya evaluasi akademik

PAKAR AKADEMIK:
- evaluasi mendalam
- perspektif profesional
- sangat detail
- boleh menggunakan istilah teknis
- fokus kualitas konseptual dan analisis ilmiah

=========================================================
RUBRIK PENILAIAN
=========================================================

1. KEBENARAN
- apakah jawaban benar secara konsep

2. KELENGKAPAN
- apakah semua poin penting dijelaskan

3. KEJELASAN
- apakah jawaban mudah dipahami

4. STRUKTUR
- apakah jawaban rapi dan sistematis

=========================================================
ATURAN SKOR
=========================================================

- setiap rubrik maksimal 25
- total skor maksimal 100
- nilai harus realistis
- jangan terlalu mudah memberi nilai tinggi
- sesuaikan tingkat penilaian dengan level pelajar
- advanced harus dinilai lebih ketat dibanding beginner

=========================================================
SISTEM GRADE
=========================================================

Gunakan grade berikut secara realistis:

A = Sempurna
B = Bagus Sekali
C = Baik
D = Cukup
E = Belajar Lagi Ya!

ATURAN:
- jangan terlalu mudah memberi A
- beginner lebih toleran
- advanced lebih ketat
- dosen dan pakar lebih kritis
- grade harus sesuai kualitas jawaban

=========================================================
PENTING TENTANG GRADE
=========================================================

- Grade HARUS dinamis
- Jangan selalu memberi grade B
- Tentukan grade berdasarkan kualitas jawaban sebenarnya
- Gunakan standar berbeda sesuai:
  - level pelajar
  - role evaluator
  - konteks mata pelajaran

Contoh:
- beginner + guru SD → lebih toleran
- advanced + dosen → lebih ketat
- pakar akademik → evaluasi lebih kritis

Gunakan grade secara realistis.

=========================================================
ATURAN FEEDBACK
=========================================================

- feedback harus spesifik
- hindari feedback generic
- sebutkan kekuatan utama jawaban
- jelaskan bagian yang perlu diperbaiki
- gunakan bahasa sesuai role evaluator
- fokus membantu pelajar memahami kesalahannya

=========================================================
FORMAT JSON WAJIB
=========================================================

{{
  "score": 0-100,

  "grade": {{
    "letter": "A/B/C/D/E",
    "label": "Excellent/Very Good/Good/Fair/Needs More Practice",
    "description": Provide a description that reflects the quality of the answer based on the student's level, evaluator role, and subject context."
  }},

  "summary": "Ringkasan evaluasi secara singkat.",

  "rubric": {{

    "kebenaran": {{
      "score": 0-25,
      "max": 25,
      "feedback": "Feedback spesifik tentang ketepatan konsep."
    }},

    "kelengkapan": {{
      "score": 0-25,
      "max": 25,
      "feedback": "Feedback spesifik tentang kelengkapan jawaban."
    }},

    "kejelasan": {{
      "score": 0-25,
      "max": 25,
      "feedback": "Feedback spesifik tentang kejelasan penjelasan."
    }},

    "struktur": {{
      "score": 0-25,
      "max": 25,
      "feedback": "Feedback spesifik tentang struktur jawaban."
    }}

  }},

  "strengths": [
    "kelebihan 1",
    "kelebihan 2"
  ],

  "improvements": [
    "perbaikan 1",
    "perbaikan 2"
  ],

  "ideal_answer": "Contoh jawaban ideal sesuai konteks pertanyaan."
}}

=========================================================
PENTING
=========================================================

- HANYA balas dengan JSON
- TANPA markdown
- TANPA teks tambahan
- jangan gunakan ```json
- pastikan JSON valid
- All text values in the JSON MUST be in English
"""

    try:

        model = genai.GenerativeModel(GEMINI_MODEL)

        response = model.generate_content(prompt)

        clean_text = response.text.strip()

        if clean_text.startswith("```json"):
            clean_text = clean_text.replace(
                "```json",
                ""
            ).replace(
                "```",
                ""
            ).strip()

        result = json.loads(clean_text)

        return jsonify({
            "result": result
        })

    except Exception as e:

        print("EVALUATE ERROR:", e)

        return jsonify({
            "error": "Gagal mengevaluasi jawaban."
        }), 500
    

# ── Prompt Comparison endpoint ──
@app.post("/compare")
def compare():
    data     = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Pertanyaan tidak boleh kosong."}), 400

    prompts = {
    "zero_shot": f"""Answer the following question directly and concisely in English.

Question: {question}""",

    "structured": f"""Answer the following question in English using EXACTLY this structure and format:

Question: {question}

**Definition:** [1-2 sentences defining the core concept]
**Explanation:** [2-3 sentences explaining how it works]
**Key Points:**
- [key point 1]
- [key point 2]
- [key point 3]
**Example:** [1 concrete real-world example]
**Conclusion:** [1-2 sentences summarizing]

Follow the structure exactly. Use the bold headers as shown.""",

    "few_shot": f"""Here are examples of how to answer questions. Each answer follows the same pattern: what it is, how it works, why it matters. Always exactly 3 sentences.

Question: What is gravity?
Answer: Gravity is a fundamental force of nature that attracts objects with mass toward each other. It works by warping the fabric of space around massive objects, so Earth pulls everything on its surface downward. Without gravity, planets wouldn't form, atmospheres would drift away, and life as we know it couldn't exist.

Question: What is DNA?
Answer: DNA is a molecule found in every living cell that stores the genetic instructions for building and operating an organism. It works like a coded blueprint, using sequences of four chemical bases to spell out instructions that cells read when growing or repairing themselves. It matters because it determines inherited traits, drives evolution, and is the foundation of modern medicine and genetics.

Question: What is inflation?
Answer: Inflation is the rate at which the general price level of goods and services rises over time, reducing purchasing power. It works when more money circulates in an economy than there are goods to buy, causing sellers to raise prices. It matters because unchecked inflation erodes savings, destabilizes economies, and forces central banks to intervene with interest rate adjustments.

Now answer this question following the EXACT same 3-sentence pattern:
Question: {question}
Answer:"""
}

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        results = {}

        for name, prompt in prompts.items():
            response      = model.generate_content(prompt)
            results[name] = response.text

        analysis_prompt = f"""You are an expert in prompt engineering.

Analyze these three AI responses to the same question. Your job is NOT to pick a winner, but to explain what each prompting technique achieved and what tradeoffs it makes.

QUESTION: {question}

ZERO-SHOT RESPONSE:
{results["zero_shot"]}

STRUCTURED RESPONSE:
{results["structured"]}

FEW-SHOT RESPONSE:
{results["few_shot"]}

-Note: Zero-shot is expected to be brief and unformatted. 
That is by design, not a flaw.

Write a short analysis (3 sentences) in plain English:
- Sentence 1: What zero-shot achieved and its tradeoff
- Sentence 2: What structured achieved and its tradeoff
- Sentence 3: What few-shot achieved and its tradeoff

Do NOT declare a winner. Do NOT say one is better than the others.
"If you must recommend one, base it only on which technique suits 
a casual educational chatbot best, not which output looks most polished."
Reply with plain text only, no markdown, no bullet points."""

        analysis_response   = model.generate_content(analysis_prompt)
        results["analysis"] = analysis_response.text.strip()

        return jsonify({
            "results":  results,
            "question": question,
            "analysis": results["analysis"]
        })

    except Exception as e:
        print(f"ERROR /compare: {e}")
        return jsonify({"error": "Gagal memCompare Prompt."}), 500

@app.post("/reset")
def reset():
    data  = request.get_json()
    level = data.get("level", None)
    
    if level and level in chat_sessions:
        del chat_sessions[level]  # hapus sesi level tertentu
    else:
        chat_sessions.clear()     # hapus semua sesi
    
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("✓ AI Smart Tutor berjalan di http://localhost:3000")
    app.run(host="0.0.0.0", port=3000, debug=True)