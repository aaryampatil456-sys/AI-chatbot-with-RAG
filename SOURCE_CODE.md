# AI Chatbot with RAG - Source Code

## 1. config.py

```python
# config.py
import os

EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE    = 500          # characters per chunk
CHUNK_OVERLAP = 50           # characters shared between chunks
TOP_K         = 3            # chunks retrieved per question
INDEX_PATH    = "vectorstore/faiss.index"
META_PATH     = "vectorstore/chunks.json"
UPLOAD_DIR    = "uploads"
DB_PATH       = "chatbot.db"
LLM_API_KEY   = os.getenv("LLM_API_KEY")      # never hard-code the key
ADMIN_USER    = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS    = os.getenv("ADMIN_PASS", "change-me")
SECRET_KEY    = os.getenv("SECRET_KEY", "dev-secret-change-me")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)
```

## 2. loader.py

```python
# loader.py - document loading, cleaning and chunking
from pypdf import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_text(path):
    """Read text from a PDF or TXT file."""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def clean_text(text):
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks
```

## 3. vector_store.py

```python
# vector_store.py - embeddings, FAISS index and retrieval
import os, json, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, INDEX_PATH, META_PATH, TOP_K, UPLOAD_DIR
from loader import load_text, clean_text, chunk_text

model = SentenceTransformer(EMBED_MODEL)

def build_index(chunks):
    """chunks = [{"text": "...", "source": "file.pdf"}, ...]"""
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])      # inner product = cosine
    index.add(np.array(vectors, dtype="float32"))
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

def retrieve(query, k=TOP_K):
    if not os.path.exists(INDEX_PATH):
        return []
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    q = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.array(q, dtype="float32"), k)
    results = []
    for score, i in zip(scores[0], ids[0]):
        if i != -1:
            results.append({"text": chunks[i]["text"],
                            "source": chunks[i]["source"],
                            "score": float(score)})
    return results

def rebuild_all_documents():
    """Load, clean, chunk and index every file in the upload folder."""
    chunks = []
    for name in sorted(os.listdir(UPLOAD_DIR)):
        text = clean_text(load_text(os.path.join(UPLOAD_DIR, name)))
        chunks += [{"text": c, "source": name} for c in chunk_text(text)]
    if chunks:
        build_index(chunks)
```

## 4. rag.py

```python
# rag.py - prompt building and answer generation
import anthropic
from config import LLM_API_KEY
from vector_store import retrieve

PROMPT = """You are a helpful assistant. Answer the question using ONLY the
context given below. If the answer is not present in the context, reply: "I
could not find this in the provided documents."

Context:
{context}

Question: {question}
Answer:"""

def llm_generate(prompt):
    client = anthropic.Anthropic(api_key=LLM_API_KEY)
    msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=800,
                                 messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text

def answer_query(question):
    docs = retrieve(question)
    if not docs:
        return "No documents have been uploaded yet.", []
    context = "\n\n".join(d["text"] for d in docs)
    prompt = PROMPT.format(context=context, question=question)
    answer = llm_generate(prompt)
    return answer, docs
```

## 5. app.py

```python
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
```

## 6. templates/index.html

```html
<!DOCTYPE html><html><head><meta charset="utf-8"><title>AI Chatbot with RAG</title></head>
<body style="font-family:sans-serif;max-width:700px;margin:40px auto">
<h2>Chat with your documents</h2>
<div id="log"></div>
<input id="q" style="width:80%" placeholder="Ask a question..."> <button onclick="ask()">Send</button>
<script>
async function ask(){
  const q=document.getElementById('q').value, log=document.getElementById('log');
  const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const d=await r.json();
  log.innerHTML+=`<p><b>You:</b> ${q}</p><p><b>Bot:</b> ${d.answer||d.error}</p>`;
  document.getElementById('q').value='';
}
</script></body></html>
```

## 7. templates/login.html

```html
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Admin Login</title></head>
<body style="font-family:sans-serif;max-width:320px;margin:60px auto">
<h2>Admin Login</h2>{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
<form method="post"><input name="username" placeholder="Username"><br><br>
<input name="password" type="password" placeholder="Password"><br><br><button>Login</button></form>
</body></html>
```

## 8. templates/dashboard.html

```html
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Admin Dashboard</title></head>
<body style="font-family:sans-serif;max-width:600px;margin:40px auto">
<h2>Upload documents</h2>
<form method="post" action="/admin/upload" enctype="multipart/form-data">
<input type="file" name="file" accept=".pdf,.txt"> <button>Upload</button></form>
<h3>Uploaded files</h3><ul>{% for f in files %}<li>{{ f }}</li>{% endfor %}</ul>
</body></html>
```

## 9. requirements.txt

```text
flask
pypdf
faiss-cpu
numpy
sentence-transformers
anthropic
```
