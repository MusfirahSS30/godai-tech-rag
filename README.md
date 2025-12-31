# Godai Tech RAG (Groq + LangChain + Chroma)

A small Retrieval-Augmented Generation (RAG) example using Groq LLMs, LangChain, and Chroma for vector storage. This project loads PDFs, creates a Chroma vector store, and provides both a CLI and a Streamlit UI for querying documents.

## 🔧 Files

- `create_database.py` — Load PDFs, split text, embed chunks, and persist to `./chroma/`.
- `query_data.py` — CLI to query the Chroma DB and generate answers via Groq.
- `app.py` — Streamlit UI to query documents and display sources.
- `compare_embeddings.py` — Small script to inspect and compare embedding vectors.
- `.env` — Environment variables (Groq API key, model).
- `requirements.txt` — Project dependencies.

## ✅ Prerequisites

- Python 3.10+ recommended
- A valid Groq API key with access to the chosen model
- Place source PDFs in the directory referenced by `DATA_DIR` inside `create_database.py` (default: `D:\Godai Tech`)

## Setup

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables in `.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

> Note: Ensure `GROQ_MODEL` contains only the model name (e.g., `llama-3.1-8b-instant`) — do **not** include a duplicate `GROQ_MODEL=` prefix.


4. Document Sources

The chatbot retrieves information from the following documents located in `/docs`:

- Godai Tec – Legal, Regulatory & Compliance Framework
- Godai Tec – Commercial Terms & Pricing Schedule
- Godai Tec – Corporate Policy & Operating Manual

These documents are included to allow full reproducibility of the RAG pipeline.


## Usage

- Create the Chroma DB (ensure PDFs are in `DATA_DIR`):

```bash
python create_database.py
```

- Query via CLI:

```bash
python query_data.py "Which act does Godai Tech act under?"
```

- Run Streamlit UI:

```bash
streamlit run app.py
```

## Troubleshooting

- If you see a `model_not_found` error from Groq, verify `GROQ_MODEL` in `.env` and confirm your Groq account has access to that model.
- If Chroma DB is missing, run `python create_database.py` to (re)create `./chroma/`.

## License

MIT — adapt as needed.

