"""Minimal RAG pipeline: load -> chunk -> embed -> FAISS search -> Groq LLM answer."""
import io
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def read_file(name, data):
    """Return text of a .pdf or .txt file given its name and raw bytes."""
    if name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def chunk_text(text, size=800, overlap=150):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


class RAGIndex:
    def __init__(self):
        self.chunks = []   # list of (source, text)
        self.index = None

    def build(self, files):
        """files: list of (filename, bytes)."""
        self.chunks = []
        for name, data in files:
            for c in chunk_text(read_file(name, data)):
                self.chunks.append((name, c))
        if not self.chunks:
            raise ValueError("No text found in the uploaded files.")
        vecs = get_embedder().encode([c for _, c in self.chunks], normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(np.asarray(vecs, dtype="float32"))

    def retrieve(self, query, k=4):
        q = get_embedder().encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), min(k, len(self.chunks)))
        return [(self.chunks[i][0], self.chunks[i][1], float(s)) for i, s in zip(ids[0], scores[0])]

    def answer(self, query, k=4):
        hits = self.retrieve(query, k)
        context = "\n\n".join(f"[{src}] {text}" for src, text, _ in hits)
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "Answer ONLY from the context given. If the answer is not in the context, say you don't know."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        return resp.choices[0].message.content, hits
