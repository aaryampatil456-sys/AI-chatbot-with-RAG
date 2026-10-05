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
