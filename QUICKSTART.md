# Quick Start Guide - Structured RAG Application

## What Changed?

The application has been refactored to use **structured initialization on startup** instead of loading documents at import time. This is much better for:

- ✅ Error handling and recovery
- ✅ Clear visibility into initialization process
- ✅ Ability to start API even if document loading fails
- ✅ Better debugging and troubleshooting
- ✅ Cleaner code organization

## New Architecture

```
App.py (RAGSystem)
    ↓
    initialize() on startup
    ↓
    Steps 1-8 (Loading, Chunking, Embeddings, etc)
    ↓
api.py (FastAPI)
    ↓
    Endpoints use rag_system instance
```

## How to Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create `.env` file:

```env
GROQ_API_KEY=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
# Optional:
# PDF_FILE_PATH=./documents/your_file.pdf
```

### 3. Configure embeddings (local) and LLM (Groq)

By default the app uses local Sentence-Transformers for embeddings. You can change the model via `HF_EMBED_MODEL` in `.env` (default `all-MiniLM-L6-v2`).

```bash
# Example .env entries
# HF_EMBED_MODEL=all-MiniLM-L6-v2
```

### 4. Start the API

```bash
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

### 5. Watch the Startup Logs

You'll see detailed initialization output:

```
============================================================
Starting RAG System Initialization
============================================================
STEP 1: Loading documents...
✓ Successfully loaded 6 documents

STEP 2: Creating text chunks...
✓ Created 45 chunks from documents

STEP 3: Initializing embeddings model...
   Using embedding model: all-MiniLM-L6-v2
✓ Embeddings model initialized

STEP 4: Setting up vector store...
   Found existing FAISS index...
✓ Loaded existing vector store

STEP 5: Creating retriever...
✓ Retriever created (k=3)

STEP 6: Initializing LLM...
   Using LLM model: llama-3.1-8b-instant
✓ LLM initialized (Groq)

STEP 7: Creating QA prompt template...
✓ QA prompt template created

STEP 8: Building RAG chain...
✓ RAG chain built successfully

============================================================
RAG System Initialization Complete! ✓
============================================================
```

### 6. Test the API

**Health Check:**

```bash
curl http://127.0.0.1:8000/
```

**System Status:**

```bash
curl http://127.0.0.1:8000/status
```

**Ask a Question:**

```bash
curl -X POST http://127.0.0.1:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main technical skills mentioned in the document?"}'
```

**Or use the test script:**

```bash
python test_api.py
```

## File Structure

```
RAG-Design-QA/
├── App.py                  # RAGSystem class (document loading, initialization)
├── api.py                  # FastAPI application (endpoints)
├── test_api.py             # Test script
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
│
├── documents/              # PDF files (required)
│   ├── document1.pdf
│   ├── document2.pdf
│   └── document3.pdf
│
├── faiss_index/            # Vector store (cached)
│   └── index.faiss
│
└── chroma_db/              # Chroma database (legacy)
```

## Key Files

### App.py - RAG System

```python
class RAGSystem:
    def initialize()      # 8-step initialization
    def get_rag_chain()   # Get the RAG chain
    def get_vectorstore() # Get the vector store

rag_system = RAGSystem()  # Global instance
```

### api.py - FastAPI

```python
@app.get("/")       # Health check
@app.get("/status") # RAG system status
@app.post("/qa")    # Question answering
```

## Initialization Steps

The `RAGSystem.initialize()` method performs 8 steps:

| Step | What                      | Output                |
| ---- | ------------------------- | --------------------- |
| 1    | Load documents (PDF/JSON) | 6 documents           |
| 2    | Split into chunks         | 45 chunks             |
| 3    | Initialize embeddings     | nomic-embed-text      |
| 4    | Create vector store       | FAISS index           |
| 5    | Create retriever          | k=3 similarity search |
| 6    | Initialize LLM            | Groq (llama-3.1-8b)   |
| 7    | Create prompt template    | System + human prompt |
| 8    | Build RAG chain           | Ready for queries     |

## API Endpoints

### GET `/` - Health Check

```json
{
  "status": "ok",
  "message": "Document RAG QA API is running.",
  "groq": "configured",
  "local_embeddings": "available",
  "rag_system": "initialized"
}
```

### GET `/status` - RAG System Status

```json
{
  "status": "ready",
  "message": "RAG system is ready for queries",
  "initialized": true,
  "documents_count": 6,
  "chunks_count": 45
}
```

### POST `/qa` - Ask a Question

**Request:**

```json
{
  "question": "What are the minimum loan requirements?"
}
```

**Response:**

```json
{
  "question": "What are the minimum loan requirements?",
  "answer": "Based on the Commercial Loan Policy 2024, minimum qualifications include...",
  "sources": ["Commercial Loan Policy 2024"]
}
```

## Common Commands

### Start API with reload (development)

```bash
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

### Start API without reload (production)

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### Test with Python script

```bash
python test_api.py
```

### View API documentation

Open browser: `http://127.0.0.1:8000/docs` (Swagger UI)

### Rebuild vector store

```bash
# Delete the cached index
rmdir /s faiss_index

# Restart the API - it will rebuild
python -m uvicorn api:app --reload
```

## Configuration

### Using Documents

**Option 1: Single PDF (via environment variable)**

```env
PDF_FILE_PATH=./documents/banking_policy.pdf
```

**Option 2: Multiple PDFs (automatic discovery)**

```bash
mkdir documents
# Place your PDFs in documents/
# API will automatically load all *.pdf files
```

> **Note:** At least one PDF must be present for the RAG system to initialize successfully.

### Customizing Parameters

In `App.py`, modify the `RAGSystem.initialize()` method:

```python
# Change chunk size
chunk_size=2000  # default: 1000

# Change number of retrieved chunks
search_kwargs={"k": 5}  # default: k=3

# Change embedding model
embed_model = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")

# Change LLM temperature
temperature=0.5  # default: 0.1 (more deterministic)
```

## Troubleshooting

### "RAG system not initialized"

- Check server logs for errors during startup
- Verify GROQ_API_KEY is set in .env
- Ensure Ollama is running: `ollama serve`
- Try restarting the API

### Slow first startup

- First run creates FAISS index from scratch (~30-60 seconds)
- Subsequent starts are much faster (~5-15 seconds)
- To keep the index: don't delete `faiss_index/` directory

### Port already in use

```bash
python -m uvicorn api:app --port 8001
```

### Module import errors

```bash
cd c:\codebase\ai-learning\RAG-Design-QA
pip install -r requirements.txt
python -m uvicorn api:app --reload
```

## Documentation

For detailed information, see:

- `ARCHITECTURE.md` - Complete architecture overview
- `PDF_SETUP_GUIDE.md` - Guide for PDF document loading
- `TROUBLESHOOTING_422.md` - Guide for API request format issues

## Testing Workflow

1. **Check API is running:**

   ```bash
   curl http://127.0.0.1:8000/
   ```

2. **Check RAG system status:**

   ```bash
   curl http://127.0.0.1:8000/status
   ```

3. **Test with the script:**

   ```bash
   python test_api.py
   ```

4. **Manual test:**
   ```bash
   curl -X POST http://127.0.0.1:8000/qa \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the minimum loan requirements?"}'
   ```

## Next Steps

- Add more documents to `documents/` folder
- Customize the system prompt in `App.py`
- Configure chunk size for your use case
- Set up monitoring and logging
- Deploy to production (AWS, GCP, Azure, etc)

---

**Need help?** Check the detailed documentation files or review the server logs during startup.
