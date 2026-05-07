"""
app.py  —  LexiSum AI Backend
==============================
Pipeline:  Upload → Extract → Chunk → Summarise (DistilBART via text2text-generation) → Translate (MarianMT) → Dashboard

Install dependencies:
    pip install flask mysql-connector-python pdfplumber python-docx werkzeug
    pip install transformers sentencepiece torch
"""

import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "hf_models")

os.environ["HF_HOME"]                         = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"]              = CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pdfplumber
import docx
import mysql.connector
from functools import wraps
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify

_summarizer = None
_translator = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        # manually load tokenizer+model since the installed transformers
        # release doesn’t support a summarization pipeline task
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print("Loading summarisation model (tokenizer+model)...")
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        model     = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
        _summarizer = (tokenizer, model)
    return _summarizer


def get_translator():
    global _translator
    if _translator is None:
        from transformers import MarianMTModel, MarianTokenizer
        print("Loading translation model...")
        model_name  = "Helsinki-NLP/opus-mt-en-mul"
        tokenizer   = MarianTokenizer.from_pretrained(model_name)
        model       = MarianMTModel.from_pretrained(model_name)
        _translator = (tokenizer, model)
    return _translator


app = Flask(__name__)
app.secret_key = "lexisum_secret_key"
app.config["UPLOAD_FOLDER"] = "temp_uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "",
    "database": "lexisum_db",
    "charset":  "utf8",
}

def get_db():
    return mysql.connector.connect(**DB_CONFIG)


# --- AUTH DECORATORS ---
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to access your dashboard.")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin"):
            flash("Admin access required.")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


# STEP 3 — EXTRACTION
def extract_text(file_path, ext):
    try:
        if ext == ".pdf":
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
            return text.strip()
        elif ext == ".docx":
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception as e:
        return f"Extraction error: {e}"


# STEP 4 — CHUNKING
def create_chunks(text, words_per_chunk=900):
    words = text.split()
    return [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]


# STEP 5 — SUMMARISATION
def summarise_text(text, detail: str = "Standard (Quick Overview)"):
    tokenizer, model = get_summarizer()
    chunks          = create_chunks(text)
    chunk_summaries = []

    # adjust lengths based on desired detail level
    if detail.lower().startswith("comprehensive"):
        max_len_chunk = 400
        min_len_chunk = 150
        max_len_merge = 600
        min_len_merge = 200
    else:
        max_len_chunk = 200
        min_len_chunk = 60
        max_len_merge = 300
        min_len_merge = 100

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        output_ids = model.generate(
            **inputs,
            max_length=max_len_chunk,
            min_length=min_len_chunk,
            do_sample=False,
        )
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        chunk_summaries.append(summary)

    merged = " ".join(chunk_summaries)
    if len(merged.split()) > 600:
        inputs = tokenizer(
            merged,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        output_ids = model.generate(
            **inputs,
            max_length=max_len_merge,
            min_length=min_len_merge,
            do_sample=False,
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return merged


# LANGUAGE TAG MAP for MarianMT
LANG_TAG_MAP = {
    "Tamil": ">>tam<<",
    "Hindi": ">>hin<<",
}

# STEP 6 — TRANSLATION (language-aware)
def translate_text(english_text, language="Tamil"):
    lang_tag = LANG_TAG_MAP.get(language)
    if not lang_tag:
        return english_text  # Return English as-is if no translation needed
    tokenizer, model = get_translator()
    tagged     = f"{lang_tag} {english_text}"
    tokens     = tokenizer([tagged], return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("user_dashboard"))
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        db  = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE email=%s AND password=%s", (email, password))
        user = cur.fetchone()
        db.close()
        if user:
            session["user_id"]   = user["id"]
            session["user_name"] = user["full_name"]
            return redirect(url_for("user_dashboard"))
        flash("Invalid email or password.")
    return render_template("login.html")


# FIX: Added GET so /register redirect doesn't 405; name field fixed to "name"
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return redirect(url_for("login"))
    name     = request.form.get("name", "").strip()
    email    = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    if not name or not email or not password:
        flash("All fields are required to register.")
        return redirect(url_for("login"))
    try:
        db  = get_db()
        cur = db.cursor()
        cur.execute(
            "INSERT INTO users (full_name, email, password) VALUES (%s, %s, %s)",
            (name, email, password)
        )
        db.commit()
        db.close()
        flash("Account created! Please log in.")
    except mysql.connector.IntegrityError:
        flash("Email already registered.")
    return redirect(url_for("login"))


@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if session.get("admin"):
        return redirect(url_for("admin_dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        db  = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM admins WHERE username=%s AND password=%s", (username, password))
        admin = cur.fetchone()
        db.close()
        if admin:
            
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        flash("Invalid admin credentials.")
    return render_template("admin_login.html")


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    db  = get_db()
    cur = db.cursor(dictionary=True)
    cur.execute(
        "SELECT d.id, u.full_name, d.filename, d.status, d.uploaded_at "
        "FROM documents d JOIN users u ON d.user_id = u.id "
        "ORDER BY d.uploaded_at DESC"
    )
    docs = cur.fetchall()
    db.close()
    return render_template("admin_dashboard.html", docs=docs)


@app.route("/user")
@login_required
def user_dashboard():
    user_id   = session.get("user_id")
    user_name = session.get("user_name", "Counsel")
    db  = get_db()
    cur = db.cursor(dictionary=True)
    cur.execute(
        "SELECT id, filename, status, uploaded_at FROM documents "
        "WHERE user_id=%s ORDER BY uploaded_at DESC LIMIT 10",
        (user_id,)
    )
    history = cur.fetchall()
    db.close()
    return render_template("user_dashboard.html", user_name=user_name, history=history)


@app.route("/upload", methods=["POST"])
@login_required
def handle_upload():
    user_id  = session.get("user_id")
    language = request.form.get("language", "Tamil")
    detail   = request.form.get("detail", "Standard (Quick Overview)")
    
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".docx", ".txt"):
        return jsonify({"error": "Unsupported file type. Use PDF, DOCX, or TXT."}), 400

    safe_name = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(save_path)

    db  = get_db()
    cur = db.cursor()
    cur.execute(
        "INSERT INTO documents (user_id, filename, status) VALUES (%s, %s, 'processing')",
        (user_id, safe_name)
    )
    db.commit()
    doc_id = cur.lastrowid
    db.close()

    # FIX: Declare outside try so they're always defined
    english_summary    = ""
    translated_summary = ""

    try:
        raw_text = extract_text(save_path, ext)
        if not raw_text or raw_text.startswith("Extraction error"):
            raise ValueError(raw_text or "Empty document — no text could be extracted.")

        english_summary    = summarise_text(raw_text, detail)
        translated_summary = translate_text(english_summary, language)

        db  = get_db()
        cur = db.cursor()
        cur.execute(
            "UPDATE documents SET status='completed', english_summary=%s, tamil_summary=%s WHERE id=%s",
            (english_summary, translated_summary, doc_id)
        )
        db.commit()
        db.close()

    except Exception as e:
        db  = get_db()
        cur = db.cursor()
        cur.execute("UPDATE documents SET status='error' WHERE id=%s", (doc_id,))
        db.commit()
        db.close()
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    return jsonify({
        "status":          "completed",
        "english_summary": english_summary,
        "tamil_summary":   translated_summary,
        "filename":        safe_name,
        "language":        language,
        "detail":          detail,
    })


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    # pre-load summariser to trigger model download (1.2 GB) ahead of first user request
    try:
        get_summarizer()
    except Exception as e:
        print("Warning: summariser preload failed:", e)
    app.run(debug=True)
