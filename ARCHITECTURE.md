# RAG Application Architecture - Structured Initialization

## Overview

The application has been refactored to use a structured initialization pattern that handles document loading, chunking, embeddings, and RAG chain setup on application startup rather than at module import time.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Application                    │
│                      (api.py)                            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ├─ Startup Event (lifespan)
                       │  └─ rag_system.initialize()
                       │
                       ├─ GET  /         (Health Check)
                       ├─ GET  /status   (RAG System Status)
                       └─ POST /qa       (Question Answering)
                       │
┌──────────────────────┴──────────────────────────────────┐
│                  RAG System (App.py)                     │
│                  ┌───────────────────┐                  │
│                  │   RAGSystem        │                  │
│                  ├───────────────────┤                  │
│                  │ • initialize()    │                  │
│                  │ • get_rag_chain() │                  │
│                  │ • get_vectorstore │                  │
│                  │ • get_retriever() │                  │
│                  └───────────────────┘                  │
└──────────────────────────────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
 Documents        Embeddings          LLM
 (PDF only)      (Local embeddings)           (Groq)
    │                  │                  │
    ▼                  ▼                  ▼
 Load PDFs      Create FAISS       Chat Groq
    │           Vector Store           │
    └──────────────────┬─────────────────┘
                       │
                       ▼
                   RAG Chain
              (Retrieval + Generation)
```

## Initialization Flow

The application follows an 8-step initialization process:

### Step 1: Load Documents

- Checks `PDF_FILE_PATH` environment variable for a single PDF
- Scans `documents/` directory for multiple PDFs
- Requires at least one PDF to be present for initialization to succeed
- Returns a list of `Document` objects with metadata

### Step 2: Create Text Chunks

- Uses `RecursiveCharacterTextSplitter`
- Chunk size: 1000 characters
- Overlap: 200 characters
- Splits on natural boundaries: paragraphs, sentences, words

### Step 3: Initialize Embeddings Model

- Uses local Sentence-Transformers embeddings by default (configurable)
- Model: `all-MiniLM-L6-v2` (configurable via `HF_EMBED_MODEL`)
- Converts text to dense vector representations

### Step 4: Set Up Vector Store (FAISS)

- **If index exists on disk:**
  - Loads existing FAISS index (faster startup)
  - Reuses previously computed embeddings
- **If index doesn't exist:**
  - Creates new FAISS index from document chunks
  - Saves to `faiss_index/` directory for future use

### Step 5: Create Retriever

- Wraps FAISS vector store for similarity search
- Configured to retrieve k=3 most relevant chunks per query

### Step 6: Initialize LLM (Groq)

- Model: `llama-3.1-8b-instant`
- Temperature: 0.1 (more deterministic)
- API key: From `GROQ_API_KEY` environment variable

### Step 7: Create QA Prompt Template

- System prompt instructs the assistant as a general document expert
- Provides guidelines for using context effectively
- Prompt template structure: system message + human question + context

### Step 8: Build RAG Chain

- Combines retriever + prompt + LLM + output parser
- Uses `RunnableParallel` for efficiency
- Returns both answer and source documents

## Code Structure

### App.py - RAG System Implementation

```python
class RAGSystem:
    """Main RAG System class for document processing, embedding, and QA chain."""

    def __init__(self):
        # Initialize all components as None
        self.documents = None
        self.chunks = None
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.qa_prompt = None
        self.rag_chain = None
        self.is_initialized = False

    def initialize(self):
        """Initialize the RAG system with all components."""
        # 8-step initialization process
        # Returns True if successful, False otherwise

    def get_rag_chain(self):
        """Get the RAG chain if initialized."""

    def get_vectorstore(self):
        """Get the vector store if initialized."""

    def get_retriever(self):
        """Get the retriever if initialized."""

# Create global instance
rag_system = RAGSystem()
```

### api.py - FastAPI Application

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup: rag_system.initialize()
    # Shutdown: Cleanup

    yield

app = FastAPI(lifespan=lifespan)

# Endpoints:
@app.get("/")            # Health check
@app.get("/status")      # RAG system status
@app.post("/qa")         # Question answering
```

## Key Features

### 1. Explicit Initialization

- Document processing happens on app startup, not on import
- Startup logs show exactly which steps are executing
- Easy to debug initialization issues

### 2. Error Handling

- Each initialization step is wrapped in try-except
- Detailed error logging with stack traces
- App can start even if RAG system fails to initialize
- Endpoints return 503 error if RAG system not ready

### 3. State Management

- Global `rag_system` instance accessible from API endpoints
- `is_initialized` flag indicates system readiness
- Prevents operations on uninitialized components

### 4. Flexible Document Loading

```bash
# Option 1: Single PDF via environment variable
PDF_FILE_PATH=./documents/document.pdf

# Option 2: Multiple PDFs in directory
documents/
├── policy1.pdf
├── policy2.pdf
└── guidelines.pdf

# Option 3: JSON fallback
data.json
```

### 5. Vector Store Persistence

- FAISS index saved to disk after first run
- Subsequent starts load existing index (10x faster)
- Delete `faiss_index/` directory to rebuild from scratch

## Endpoints

### 1. GET `/` - Health Check

Returns API and model connectivity/configuration status.

```json
{
  "status": "ok",
  "message": "Document RAG QA API is running.",
  "groq": "configured",
  "local_embeddings": "available",
  "rag_system": "initialized"
}
```

### 2. GET `/status` - RAG System Status

Returns detailed initialization status.

```json
{
  "status": "ready",
  "message": "RAG system is ready for queries",
  "initialized": true,
  "documents_count": 6,
  "chunks_count": 45
}
```

### 3. POST `/qa` - Question Answering

Asks a question and gets an answer with sources.

**Request:**

```json
{
  "question": "What are the main technical skills mentioned?"
}
```

**Response:**

```json
{
  "question": "What are the main technical skills mentioned?",
  "answer": "Based on the document content..."
}
```

## Configuration

### Environment Variables

Create a `.env` file with:

```env
# Required
GROQ_API_KEY=your_api_key_here

# Optional - Document Loading
PDF_FILE_PATH=./documents/your_document.pdf

# Optional - Embeddings Configuration (local)
HF_EMBED_MODEL=all-MiniLM-L6-v2

# Required for LLM
GROQ_API_KEY=your_api_key_here

# Optional - If you prefer remote embeddings (OpenAI), set `OPENAI_API_KEY`
# OPENAI_API_KEY=your_openai_api_key_here
```

## Startup Process

### 1. Start the Application

```bash
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

### 2. Watch the Initialization Logs

The application will print detailed initialization progress:

```
============================================================
Starting RAG System Initialization
============================================================
STEP 1: Loading documents...
✓ Successfully loaded 6 documents

STEP 2: Creating text chunks...
✓ Created 45 chunks from documents

STEP 3: Initializing embeddings model...
   Using embedding model: nomic-embed-text
✓ Embeddings model initialized

STEP 4: Setting up vector store...
   Found existing FAISS index at: faiss_index/index.faiss
   Loading from disk...
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

### 3. Check API Status

```bash
curl http://127.0.0.1:8000/
```

## Development Workflow

### Adding a New Document

1. Place PDF in `documents/` folder
2. Delete `faiss_index/` directory (or modify existing)
3. Restart the application
4. RAG system will rebuild the vector store with new documents

### Debugging Initialization

1. Check server logs for error messages
2. Verify environment variables are set correctly
3. Ensure Ollama is running (if using Ollama)
4. Test with health endpoint: `curl http://127.0.0.1:8000/`

### Testing the API

```bash
# Use the provided test script
python test_api.py

# Or use cURL
curl -X POST http://127.0.0.1:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main technical skills mentioned?"}'
```

## Troubleshooting

### Issue: "RAG system not initialized"

- Check server startup logs for errors
- Verify GROQ_API_KEY is set
- Ensure Ollama is running (for embeddings)
- Check that documents exist

### Issue: Slow startup

- FAISS index is being created from scratch
- First run takes longer due to document processing
- Subsequent runs load cached index (much faster)

### Issue: "Port already in use"

```bash
# Use a different port
python -m uvicorn api:app --port 8001
```

### Issue: Module import errors

```bash
# Make sure you're in the correct directory
cd c:\codebase\ai-learning\RAG-Design-QA

# Verify Python environment
python -m pip install -r requirements.txt
```

## Performance Metrics

- **Startup time (first run):** ~30-60 seconds (includes FAISS index creation)
- **Startup time (subsequent):** ~5-15 seconds (loads cached index)
- **Query response time:** ~2-5 seconds (depends on LLM model and document size)
- **Memory usage:** ~500MB-1GB (FAISS index + embeddings + LLM)

## Best Practices

1. **Use FAISS Index Caching**
   - Keep the `faiss_index/` directory
   - Deletes it only when rebuilding with new documents

2. **Monitor Logs**
   - Check startup logs for initialization progress
   - Watch for warnings about missing documents

3. **Test Before Production**
   - Use the test script: `python test_api.py`
   - Verify documents are loading correctly
   - Check Ollama and Groq connectivity

4. **Environment Variables**
   - Never commit API keys to version control
   - Use `.env` file and keep it in `.gitignore`
   - Use appropriate values for chunk size and overlap

## Future Enhancements

1. **Dynamic Document Reload**
   - Add `/reload` endpoint to reload documents without restart
   - Support adding/removing documents at runtime

2. **Multiple Vector Stores**
   - Support for different embedding models
   - Hybrid search (keyword + semantic)

3. **Caching**
   - Cache frequently asked questions
   - Reduce redundant LLM calls

4. **Monitoring**
   - Track query performance metrics
   - Log query patterns and effectiveness

5. **Word Document Support**
   - Add `python-docx` for DOCX files
   - Extend document loader to support more formats
