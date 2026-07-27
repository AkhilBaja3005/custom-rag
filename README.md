# 🚀 Local 10,000-Page PDF RAG Application (MPS Accelerated)

A production-grade, local-first Retrieval-Augmented Generation (RAG) system built for high-throughput processing of massive PDF documents (up to 10,000 pages) on Apple Silicon macOS (`device="mps"`). Designed under strict **1.5 GB peak RAM limit** using streaming ingestion, local vector storage, and cross-encoder re-ranking.

---

## ⚡ Technical Highlights

- **Streaming Batch Ingestion**: Processes PDFs in configurable page batches (default 50 pages) using PyMuPDF (`fitz`) and `pymupdf4llm` to prevent loading large files into system memory.
- **Multi-Modal Metadata Extraction**: Extracts digital text, structured Markdown tables, AcroForms/annotations (`page.widgets()`, `page.annots()`), and visual assets.
- **LLM Vision Pass**: Detects embedded diagrams/CAD drawings (> 200x200 px) and offloads textual summaries to `gemini-3.1-flash-lite` via `google-genai` SDK.
- **Local Embedding & Storage**: Uses `SentenceTransformer("BAAI/bge-small-en-v1.5")` on Apple Silicon Metal Performance Shaders (`mps`) and stores embeddings in local on-disk Qdrant collection (`./qdrant_db`).
- **Two-Stage Retrieval & Re-ranking**: Fast top-15 vector retrieval followed by cross-encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on `mps` down to top-5 chunks.
- **Zero-Extrapolation Generation**: Synthesizes responses strictly using retrieved context with inline page citations `[Source: Page X]` via `gemini-3.1-flash-lite`.
- **Streamlit Web UI**: Includes live batch ingestion progress, chat interface with expandable context accordions, and memory-safe "New Chat" functionality.
- **Automated 10k-Page Benchmark**: Built-in benchmark engine (`build_and_benchmark.py`) that generates a real 10,000-page dataset from arXiv (`cat:cs.AI`), executes stress testing with `psutil` peak RAM logging, and verifies citation accuracy.

---

## 📁 Repository Structure

```text
├── config.py                 # System configuration, paths, and hardware device settings
├── ingest.py                 # Multi-parser streaming ingestion engine
├── query.py                  # Vector retrieval, cross-encoder re-ranking & synthesis
├── app.py                    # Streamlit web user interface
├── build_and_benchmark.py    # arXiv dataset builder, memory profiler & evaluation suite
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── qdrant_db/                # Local on-disk vector database storage (generated)
```

---

## 🛠️ Requirements & Setup

### Prerequisites
- macOS on Apple Silicon (M1/M2/M3/M4) with PyTorch Metal (MPS) support.
- Python 3.10+
- Gemini API Key set in environment variable: `GEMINI_API_KEY`

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Custom RAG"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Gemini API Key:**
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```

---

## 🏃 Running the Application

### 1. Launch the Streamlit Web UI
```bash
streamlit run app.py
```
- Upload your PDF file.
- Monitor real-time streaming ingestion progress.
- Ask questions and view answers along with expandable context sources and page numbers.

### 2. Run the 10,000-Page Benchmark & Stress Test
```bash
python build_and_benchmark.py
```
This script will:
1. Download computer science research papers from arXiv API (`cat:cs.AI`).
2. Stitch them into a 10,000-page master PDF (`arxiv_10000_pages_master.pdf`).
3. Execute `ingest.py` while logging peak RAM usage via `psutil` (asserting < 1.5 GB).
4. Run sample query evaluation and verify `[Source: Page X]` citation correctness.

---

## ⚙️ Configuration Options (`config.py`)

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `DEVICE` | `"mps"` (fallback `"cpu"`) | Hardware acceleration device for embeddings & cross-encoder |
| `BATCH_SIZE` | `50` pages | Streaming ingestion batch size for RAM control |
| `CHUNK_MIN_WORDS` | `400` | Minimum word count for sliding window text chunks |
| `CHUNK_MAX_WORDS` | `500` | Maximum word count for sliding window text chunks |
| `TOP_K_VECTOR` | `15` | Number of candidate chunks retrieved from vector search |
| `TOP_K_RERANK` | `5` | Final top chunks selected after cross-encoder re-ranking |
| `QDRANT_PATH` | `./qdrant_db` | On-disk storage path for Qdrant vector database |

---

## 📜 License
MIT License
