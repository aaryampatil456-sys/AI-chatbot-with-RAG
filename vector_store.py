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
