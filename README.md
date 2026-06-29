<div align="center">

# 📄 SmartDocAI

**Explainable Retrieval-Augmented Generation using Google Gemini**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Google-Gemini_API-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-Educational-green?style=flat)](#license)

Upload a PDF → Ask a question → Get a grounded, explainable answer.

</div>

---

## What is SmartDocAI?

SmartDocAI is a full **RAG (Retrieval-Augmented Generation)** pipeline built from scratch. It ingests PDF documents, retrieves the most relevant passages using three independent NLP methods, and generates answers grounded strictly in your document using Google Gemini — with full visibility into every step.

Built as an NLP internship mini-project to understand the complete RAG pipeline from document ingestion to answer generation.

---

## Features

| | Feature |
|---|---|
| 📄 | PDF document ingestion via PyMuPDF |
| 🧹 | Text preprocessing — cleaning, stop-word removal, lemmatization |
| ✂️ | Overlapping sentence-based chunking |
| 🔍 | Three retrieval methods: **BoW**, **TF-IDF**, **Sentence Embeddings** |
| 🤖 | Gemini-powered answer generation with confidence scoring |
| 📊 | Side-by-side retrieval comparison across all three methods |
| ⏱️ | Full pipeline timing trace — see where every second goes |
| 🌐 | Interactive Streamlit UI |

---

## Pipeline

```
PDF Document
     │
     ▼
 PDF Loader          ← PyMuPDF, page-by-page extraction
     │
     ▼
 Preprocessing       ← lowercase · punctuation · stopwords · lemmatize
     │
     ▼
 Chunking            ← overlapping sentence windows with metadata
     │
     ▼
 ┌───────────────────────────────┐
 │  Retrieval Engine             │
 │                               │
 │  Bag of Words  ·  TF-IDF      │
 │  Sentence Embeddings          │  ← all-MiniLM-L6-v2
 └───────────────────────────────┘
     │
     ▼
 Best Chunk Selected
     │
     ▼
 Prompt Construction ← system role · question · context · output format
     │
     ▼
 Google Gemini API
     │
     ▼
 Final Answer + Confidence + Source Chunk
```

---

## Project Structure

```
SmartDocAI/
│
├── data/
│   └── uploads/            # PDF storage
│
├── src/
│   ├── app.py              # Streamlit UI
│   ├── pipeline.py         # Orchestrator — wires every module together
│   ├── pdf_loader.py       # PDF → DocumentResult
│   ├── preprocessing.py    # Text cleaning and normalization
│   ├── chunking.py         # Sentence-window chunking
│   ├── retrieval.py        # BoW · TF-IDF · Embeddings
│   ├── prompt_builder.py   # Structured prompt assembly
│   ├── llm.py              # Gemini API integration
│   └── test_pipeline.py    # Standalone pipeline test
│
├── .env                    # GEMINI_API_KEY goes here
├── requirements.txt
└── README.md
```

---

## Retrieval Methods

| Method | How it works | Strength | Limitation |
|--------|-------------|----------|------------|
| **Bag of Words** | Keyword frequency matching | Fast, interpretable | Exact match only |
| **TF-IDF** | Weighted keyword importance | Handles common words better | No semantic understanding |
| **Sentence Embeddings** | `all-MiniLM-L6-v2` cosine similarity | Captures meaning, not just words | Higher compute cost |

The embedding result is used as the final LLM context. All three are shown side-by-side in the UI for comparison.

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/SmartDocAI.git
cd SmartDocAI
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com).

> **Note on free tier limits:** `gemini-2.5-flash` allows 20 requests/day on the free tier. Switch to `gemini-1.5-flash` in `llm.py` for 1,500 requests/day at no cost.

### 5. Run the app

```bash
streamlit run src/app.py
```

---

## Usage

1. Upload a PDF using the sidebar
2. Click **Build Index** — the document is loaded, preprocessed, and chunked
3. Type a question in the main panel
4. Click **Generate Answer**
5. Explore the answer, retrieved chunks, similarity scores, and pipeline trace

---

## Tech Stack

- [Python 3.11+](https://python.org)
- [Streamlit](https://streamlit.io) — UI
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF parsing
- [scikit-learn](https://scikit-learn.org) — BoW & TF-IDF
- [Sentence Transformers](https://www.sbert.net) — `all-MiniLM-L6-v2` embeddings
- [Google Gemini API](https://ai.google.dev) — answer generation
- [NumPy](https://numpy.org) / [Pandas](https://pandas.pydata.org) — data handling

---

## Roadmap

- [ ] FAISS vector database for faster retrieval
- [ ] Cached embeddings to skip re-encoding on reload
- [ ] Hybrid retrieval scoring (combine all three methods)
- [ ] Conversation memory for multi-turn Q&A
- [ ] Citation highlighting in the source PDF
- [ ] Multi-document support
- [ ] Docker deployment

---

## Learning Outcomes

This project covers the full RAG stack end-to-end:

- PDF parsing and text extraction
- NLP preprocessing pipeline
- Text chunking strategies
- Classical information retrieval (BoW, TF-IDF)
- Semantic search with sentence embeddings
- Prompt engineering for grounded answers
- LLM integration with Gemini
- Streamlit application development

---

## Author

**Sadia Halima** — NLP Internship Mini Project

---

## License

This project is intended for educational purposes as part of an internship assignment.
