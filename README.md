# RAG Project: Chat with Your Documents

A Retrieval-Augmented Generation (RAG) application. Upload PDF or TXT files, ask questions in a chat box, and get answers that come **only from your documents**, together with the exact source chunks used.


## 1. Problem statement

Large language models (LLMs) are powerful, but on their own they:

- **hallucinate**, giving confident answers with no evidence,
- **don't know private data** such as college notes, reports or company files,
- **go out of date**, because they only know what they saw before their training cutoff.

RAG solves this by *retrieving* the relevant text from your own documents first and then asking the LLM to answer using only that text.

## 2. Features

- Upload one or more PDF/TXT files
- One-click processing (chunking, embedding, indexing)
- Chat interface built with Streamlit
- Answers grounded in the documents; the bot says it doesn't know when the answer isn't there
- "Sources used" panel showing each retrieved chunk and its similarity score

## 3. How it works

```
INDEXING (once per upload)
  PDF/TXT --> extract text --> split into chunks --> embed chunks --> store in FAISS index

QUERY (every question)
  question --> embed --> FAISS top-k search --> build prompt with chunks --> LLM --> answer + sources
```

| Step | What happens | Where |
|---|---|---|
| Load | `pypdf` extracts text from PDFs; TXT files are decoded as UTF-8 | `read_file()` in `rag.py` |
| Chunk | Text is split into 800-character pieces with 150 characters of overlap | `chunk_text()` |
| Embed | `all-MiniLM-L6-v2` converts each chunk to a 384-dimension vector (normalized) | `RAGIndex.build()` |
| Store | Vectors go into a FAISS `IndexFlatIP` index; with normalized vectors, inner product equals cosine similarity | `RAGIndex.build()` |
| Retrieve | The question is embedded the same way and the top 4 most similar chunks are returned | `RAGIndex.retrieve()` |
| Generate | Chunks plus the question are sent to Llama 3.3 70B on Groq, with an instruction to answer only from the context (temperature 0.2) | `RAGIndex.answer()` |

## 4. Tech stack

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Best ecosystem for ML and NLP |
| PDF parsing | pypdf | Simple, pure Python |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Small, fast, runs locally for free |
| Vector search | FAISS (`faiss-cpu`) | Fast similarity search, no server needed |
| LLM | Llama 3.3 70B via Groq API | Free API key, very fast responses |
| UI | Streamlit | A working web app in a few lines |

## 5. Project structure

```
rag-project/
├── app.py             # Streamlit UI: upload, process, chat, sources
├── rag.py             # RAG pipeline: load, chunk, embed, retrieve, generate
├── requirements.txt   # Python dependencies
├── .env.example       # Template for the API key
├── .gitignore         # Keeps .env and .venv out of Git
└── README.md
```

## 6. Setup and run

**Prerequisites:** Python 3.10 or newer and a free Groq API key from https://console.groq.com.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
copy .env.example .env            # Windows (use cp on Linux/macOS)
# open .env and set GROQ_API_KEY=your_key

# 4. Start the app
streamlit run app.py
```

The first run downloads the embedding model (about 90 MB), so it takes a little longer.

## 7. Usage

1. Open the app in your browser (Streamlit prints the address, usually http://localhost:8501).
2. In the sidebar, upload your PDF/TXT files and click **Process documents**.
3. Type a question in the chat box.
4. Open **Sources used** under the answer to verify it against the original text.

## 8. Configuration

Settings are plain values you can change in `rag.py`:

| Setting | Default | Effect |
|---|---|---|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model used for answers |
| `chunk_text(size, overlap)` | 800 / 150 | Larger chunks give more context per hit; smaller chunks give sharper matches |
| `k` | 4 | Number of chunks sent to the LLM |

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `KeyError: 'GROQ_API_KEY'` | Make sure `.env` exists in the project root and contains your key |
| "No text found" error | The PDF is probably scanned images; this version has no OCR |
| Slow first start | The embedding model is downloading; later runs are faster |
| `faiss` install fails | Use Python 3.10 to 3.12 and upgrade pip first |

## 10. Limitations and future work

**Limitations:** the index lives in memory and resets when the app restarts; scanned PDFs are not supported; fixed-size chunking can split one topic across two chunks; there is no automatic answer-quality evaluation.

**Future work:** persistent vector store (ChromaDB), OCR for scanned files, smarter chunking and re-ranking, chat history, and per-document filters.
