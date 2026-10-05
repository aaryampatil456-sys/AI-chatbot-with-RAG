# app.py - Flask application (chat API, admin login, document upload)
import os, uuid, hmac, sqlite3
from flask import (Flask, request, jsonify, session, render_template, redirect)
from werkzeug.utils import secure_filename
from config import *
from rag import answer_query
from vector_store import rebuild_all_documents

app = Flask(__name__)
app.secret_key = SECRET_KEY

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS chat_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
        question TEXT, answer TEXT, sources TEXT)""")
    conn.commit(); conn.close()

def save_chat(sid, question, answer, docs):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO chat_history (session_id, question, answer, sources) "
                 "VALUES (?, ?, ?, ?)",
                 (sid, question, answer, ", ".join(d["source"] for d in docs)))
    conn.commit(); conn.close()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question or len(question) > 1000:
        return jsonify({"error": "Please enter a valid question."}), 400
    answer, docs = answer_query(question)
    sid = session.setdefault("sid", uuid.uuid4().hex)
    save_chat(sid, question, answer, docs)
    return jsonify({"answer": answer,
                    "sources": [{"source": d["source"], "text": d["text"]} for d in docs]})

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        return render_template("login.html")
    user = request.form.get("username", "")
    pwd = request.form.get("password", "")
    if hmac.compare_digest(user, ADMIN_USER) and hmac.compare_digest(pwd, ADMIN_PASS):
        session["admin"] = True
        return redirect("/admin/dashboard")
    return render_template("login.html", error="Invalid credentials")

@app.route("/admin/dashboard")
def dashboard():
    if not session.get("admin"):
        return redirect("/admin/login")
    return render_template("dashboard.html", files=sorted(os.listdir(UPLOAD_DIR)))

@app.route("/admin/upload", methods=["POST"])
def upload():
    if not session.get("admin"):
        return "Unauthorized", 401
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith((".pdf", ".txt")):
        return "Only PDF or TXT files are allowed", 400
    path = os.path.join(UPLOAD_DIR, secure_filename(f.filename))
    f.save(path)
    rebuild_all_documents()       # load, clean, chunk, embed, index
    return redirect("/admin/dashboard")

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
