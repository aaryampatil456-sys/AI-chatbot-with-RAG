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
