# Banking RAG QA System - Complete Setup & Implementation Guide

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Installation Steps](#installation-steps)
5. [Configuration](#configuration)
6. [Running the Application](#running-the-application)
7. [How the System Works](#how-the-system-works)
8. [API Documentation](#api-documentation)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

This is a **Retrieval-Augmented Generation (RAG)** system designed to answer questions about banking policies, loans, and financial products using a local LLM (via Ollama) and vector embeddings.

### Key Features:

- ✅ Local LLM processing (no cloud API calls)
- ✅ Vector-based semantic search using FAISS
- ✅ Document chunking with overlap for better context
- ✅ FastAPI REST endpoints for easy integration
- ✅ Real-time Ollama health checks
- ✅ Source tracking (returns which documents were used)

### Tech Stack:

- **LangChain**: LLM orchestration framework
- **FAISS**: Vector similarity search
- **Ollama**: Local LLM inference engine
- **FastAPI**: REST API framework
- **LangChain-Ollama**: Integration layer

---

## Architecture

### System Flow Diagram:

```
User Question
    ↓
[Vector Embedding] (Ollama: nomic-embed-text)
    ↓
[FAISS Vector Store]
    ↓
[Retrieve Top 3 Similar Chunks]
    ↓
[Format Context with Sources]
    ↓
[LLM with Prompt Template] (Ollama: gemma:2b)
    ↓
[Parse & Return Answer + Sources]
```

### Component Breakdown:

1. **Data Layer** (`data.json`)
   - 6 banking policy documents
   - Each document contains policy details, rates, and requirements

2. **Embedding Layer** (`App.py`)
   - Loads documents and splits into chunks (1000 chars, 200 char overlap)
   - Creates vector embeddings using nomic-embed-text model
   - Stores embeddings in FAISS index (persisted on disk)

3. **LLM Layer** (`App.py`)
   - Ollama running locally on port 11434
   - Model: gemma:2b (2B parameter model)
   - Temperature: 0.7 (for balanced creativity)

4. **API Layer** (`api.py`)
   - FastAPI server on port 8000
   - 2 endpoints: health check + QA endpoint
   - Returns answer with source documents

---

## Prerequisites

### System Requirements:

- Windows 10/11, macOS, or Linux
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space (for models and indexes)
- Python 3.8 or higher

### Required Software:

- Python 3.8+
- Ollama (from https://ollama.ai/)
- pip (Python package manager)

### Ollama Models Required:

- `nomic-embed-text` (for embeddings, ~274MB)
- `gemma:2b` (for LLM responses, ~1.6GB)

---

## Installation Steps

### Step 1: Install Ollama

1. Download from https://ollama.ai/
2. Install and launch Ollama
3. Verify it's running: Open terminal and run:
   ```powershell
   curl http://localhost:11434/api/tags
   ```
   You should get a JSON response if Ollama is running.

### Step 2: Pull Required Models

In your terminal/PowerShell:

```powershell
# Pull embedding model
ollama pull nomic-embed-text

# Pull LLM model
ollama pull gemma:2b
```

This may take 5-10 minutes depending on your internet speed.

### Step 3: Verify Models

```powershell
ollama list
```

You should see both `nomic-embed-text` and `gemma:2b` in the list.

### Step 4: Clone/Setup Project

Navigate to your project directory:

```powershell
cd c:\codebase\ai-learning\RAG-Design-QA
```

### Step 5: Create Python Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\Activate.ps1

# Or on macOS/Linux:
# source venv/bin/activate
```

### Step 6: Install Python Dependencies

```powershell
pip install -r requirements.txt
```

**Dependencies Explained:**

- `langchain`: Core RAG framework
- `langchain-ollama`: Ollama integration
- `langchain-community`: Community vectorstores
- `langchain-text-splitters`: Text chunking utilities
- `faiss-cpu`: Vector similarity search
- `python-dotenv`: Environment variable management
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `chromadb`: Alternative vector store (included)
- `tiktoken`: Token counting for OpenAI models
- `pypdf`: PDF document loading

### Step 7: Verify Installation

```powershell
python -c "import langchain; import faiss; import fastapi; print('All imports successful!')"
```

---

## Configuration

### Step 1: Create `.env` File

Create a file named `.env` in the project root directory:

```
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=gemma:2b

# Model Parameters
OLLAMA_TEMPERATURE=0.7

# Optional: API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### Step 2: Understand Configuration Options

| Variable             | Default                | Description                       |
| -------------------- | ---------------------- | --------------------------------- |
| `OLLAMA_BASE_URL`    | http://localhost:11434 | Ollama server URL                 |
| `OLLAMA_EMBED_MODEL` | nomic-embed-text       | Embedding model name              |
| `OLLAMA_CHAT_MODEL`  | gemma:2b               | Chat/LLM model name               |
| `OLLAMA_TEMPERATURE` | 0.7                    | LLM response randomness (0.0-1.0) |

**Temperature Explanation:**

- 0.0 = Deterministic (same answer every time)
- 0.5 = Balanced
- 1.0 = Creative/Varied answers

---

## Running the Application

### Option 1: Run with API Server

#### Terminal 1 - Verify Ollama is Running:

```powershell
# In PowerShell, Ollama should be running in system tray
# Verify connectivity:
curl http://localhost:11434/api/tags
```

#### Terminal 2 - Run the Application:

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the API server
python -m uvicorn api:app --host localhost --port 8000 --reload
```

Expected output:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Started server process [12345]
```

### Option 2: Quick Test with Python Script

Create `test_rag.py`:

```python
from App import rag_chain

# Test question
question = "What are the requirements for a commercial loan?"

# Run the RAG chain
result = rag_chain.invoke(question)

print("Question:", question)
print("\nAnswer:", result["answer"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"  - {doc.metadata.get('title', 'Unknown')}")
```

Run it:

```powershell
python test_rag.py
```

---

## How the System Works

### Phase 1: Initialization (App.py startup)

**Step 1.1: Load Documents**

```python
with DATA_PATH.open("r", encoding="utf-8") as f:
    financial_documents = json.load(f)
```

- Loads 6 banking policy documents from `data.json`
- Each document has `title` and `content` fields

**Step 1.2: Create Document Objects**

```python
documents = [
    Document(
        page_content=doc["content"],
        metadata={
            "title": doc["title"],
            "source": doc["title"],
            "category": "banking_policy",
        },
    )
    for doc in financial_documents
]
```

- Wraps raw documents in LangChain `Document` objects
- Adds metadata (title, source, category)

**Step 1.3: Split into Chunks**

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,         # Max 1000 characters per chunk
    chunk_overlap=200,       # 200 char overlap between chunks
    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
)
chunks = text_splitter.split_documents(documents)
```

- Splits large documents into smaller chunks
- Overlap ensures context is preserved at chunk boundaries
- Separators prioritize splitting at natural boundaries

**Step 1.4: Create Embeddings**

```python
embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
```

- Initializes embedding model (nomic-embed-text)
- Converts text chunks into 768-dimensional vectors

**Step 1.5: Create/Load Vector Store**

```python
if faiss_index_path.exists():
    # Load existing index from disk
    faiss_vectorstore = FAISS.load_local(str(BASE_DIR / "faiss_index"), ...)
else:
    # Create new index from documents
    faiss_vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    faiss_vectorstore.save_local(str(BASE_DIR / "faiss_index"))
```

- Uses FAISS (Facebook AI Similarity Search) for fast vector lookups
- Persists index to disk for reuse (no need to recompute on restart)

**Step 1.6: Create Retriever**

```python
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})
```

- Wraps FAISS in a retriever that fetches top-3 similar documents

**Step 1.7: Initialize LLM**

```python
llm = ChatOllama(
    model=os.getenv("OLLAMA_CHAT_MODEL", "gemma:2b"),
    temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)
```

- Initializes the gemma:2b model for text generation

### Phase 2: Question Processing (API Request)

**Step 2.1: User Sends Question**

```http
POST /qa HTTP/1.1
Content-Type: application/json

{"question": "What are the interest rates for commercial loans?"}
```

**Step 2.2: Retrieve Similar Documents**

```python
retrieved_docs = faiss_vectorstore.similarity_search(request.question, k=3)
```

- Converts question to embedding
- Finds 3 most similar document chunks
- Example: Question about "commercial loan interest rates" matches "Commercial Loan Policy 2024"

**Step 2.3: Format Context**

```python
def format_docs(docs: List[Document]) -> str:
    formatted_chunks = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown Source")
        formatted_chunks.append(f"Source: {title}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted_chunks)
```

- Concatenates chunks with source headers
- Creates readable context for the LLM

**Step 2.4: Create Prompt with Context**

```python
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a knowledgeable banking assistant..."),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
])
```

- System prompt gives LLM instructions
- Human prompt includes question + retrieved context

**Step 2.5: Generate Answer**

```python
result = rag_chain.invoke(request.question)
```

- LLM processes: System prompt + retrieved context + question
- Generates answer based only on provided context

**Step 2.6: Extract and Return Sources**

```python
sources = [
    doc.metadata.get("title", "Unknown Source")
    for doc in result.get("source_documents", [])
]
return {
    "question": request.question,
    "answer": result["answer"],
    "sources": unique_sources,
}
```

- Extracts source document titles
- Removes duplicates
- Returns structured response

---

## API Documentation

### Health Check Endpoint

**Request:**

```http
GET /
```

**Response (Success):**

```json
{
  "status": "ok",
  "message": "Banking RAG QA API is running.",
  "ollama": "connected"
}
```

**Response (Ollama Offline):**

```json
{
  "status": "ok",
  "message": "Banking RAG QA API is running.",
  "ollama": "disconnected",
  "warning": "Ollama not accessible at http://localhost:11434"
}
```

### Question-Answer Endpoint

**Request:**

```http
POST /qa
Content-Type: application/json

{
    "question": "What is the minimum credit score for a commercial loan?"
}
```

**Response (Success):**

```json
{
  "question": "What is the minimum credit score for a commercial loan?",
  "answer": "According to the Commercial Loan Policy 2024, the minimum personal credit score required for a commercial loan is 680 or higher. Additionally, you'll need a business credit score (Paydex) of 75 or higher.",
  "sources": ["Commercial Loan Policy 2024"]
}
```

**Response (Error - Ollama Not Running):**

```json
{
  "detail": "Error connecting to LLM model. Make sure Ollama is running on http://localhost:11434"
}
```

**Response (Error - Processing Failure):**

```json
{
  "detail": "Error processing question: [error details]"
}
```

### Status Codes

| Code | Meaning                 |
| ---- | ----------------------- |
| 200  | Successful response     |
| 503  | Ollama connection error |
| 500  | Internal server error   |

---

## Example Usage

### Using cURL

```powershell
# Health check
curl http://localhost:8000/

# Ask a question
curl -X POST http://localhost:8000/qa `
  -H "Content-Type: application/json" `
  -d '{"question": "What loans are available?"}'
```

### Using Python

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/qa",
    json={"question": "What are FHA loan requirements?"}
)

result = response.json()
print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Sources: {', '.join(result['sources'])}")
```

### Using FastAPI Swagger UI

1. Ensure API is running
2. Open: http://localhost:8000/docs
3. Use the interactive Swagger interface to test endpoints
4. Or use: http://localhost:8000/redoc for ReDoc documentation

---

## File Structure and Descriptions

```
RAG-Design-QA/
├── App.py                    # Core RAG chain logic
├── api.py                    # FastAPI endpoints
├── data.json                 # Banking policy documents
├── requirements.txt          # Python dependencies
├── .env                      # Configuration (create yourself)
├── faiss_index/              # FAISS vector store (auto-created)
│   ├── index.faiss           # Vector embeddings index
│   └── [uuid]/               # Index metadata
├── chroma_db/                # Alternative vector store (not used)
│   └── chroma.sqlite3
└── venv/                     # Virtual environment (auto-created)
```

---

## Troubleshooting

### Issue 1: "Connection refused: http://localhost:11434"

**Cause:** Ollama is not running

**Solution:**

```powershell
# On Windows, Ollama should be in system tray
# Check if running; if not, restart Ollama application
# Verify:
curl http://localhost:11434/api/tags
```

### Issue 2: Models not found (404 error)

**Cause:** Required models not downloaded

**Solution:**

```powershell
# Pull models
ollama pull nomic-embed-text
ollama pull gemma:2b

# Verify
ollama list
```

### Issue 3: "ModuleNotFoundError: No module named 'langchain'"

**Cause:** Dependencies not installed

**Solution:**

```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue 4: FAISS index not found

**Cause:** First run or index was deleted

**Solution:**

- First run: App.py will automatically create the index
- Just run the application, it will generate `faiss_index/` folder

```powershell
python -c "from App import faiss_vectorstore; print('Index created successfully')"
```

### Issue 5: Slow responses

**Cause:** Model loading or network issues

**Solution:**

1. Increase Ollama memory: Check Ollama settings
2. Use faster model: `mistral:7b` instead of `gemma:2b`
3. Check system RAM and free disk space

### Issue 6: "Port 8000 already in use"

**Cause:** Another application is using port 8000

**Solution:**

```powershell
# Use a different port
python -m uvicorn api:app --host 0.0.0.0 --port 8080

# Or kill the process on port 8000
netstat -ano | findstr :8000
taskkill /PID [PID] /F
```

### Issue 7: FAISS load error with "allow_dangerous_deserialization"

**Cause:** Security warning in newer versions

**Solution:** This is already handled in the code (line 45 of App.py). Just ensure your `requirements.txt` has current versions.

---

## Performance Optimization Tips

### 1. Adjust Chunk Size

**Current:** 1000 chars with 200 char overlap

- Smaller chunks = faster search, less context
- Larger chunks = slower search, more context
- Edit in `App.py` line 27-31

### 2. Change Model

**Current:** gemma:2b (2 billion parameters, ~6 seconds per response)

- Faster: `mistral:7b` (7B, ~8 seconds)
- Faster: `neural-chat:7b` (more specialized)
- Slower but better: `mistral:large` (34B, ~30 seconds)

To change:

```bash
ollama pull mistral:7b
# Then update .env: OLLAMA_CHAT_MODEL=mistral:7b
```

### 3. Adjust Temperature

**Current:** 0.7 (balanced)

- Lower (0.1-0.3) = More precise, less creative
- Higher (0.8-1.0) = More creative, less consistent

### 4. Reduce Retrieved Documents

**Current:** Top 3 documents
Edit `App.py` line 56:

```python
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})  # Reduce from 3 to 2
```

---

## Next Steps & Extensions

1. **Add More Data:**
   - Add more policy documents to `data.json`
   - Rebuild vector store

2. **Fine-tune Prompting:**
   - Edit system prompt in `App.py` (lines 59-72)
   - Test different instruction styles

3. **Add Authentication:**
   - Add API key validation to `api.py`
   - Token-based request limiting

4. **Database Integration:**
   - Store conversation history
   - Track usage metrics

5. **Web Interface:**
   - Create frontend with React/Vue
   - Real-time response streaming

---

## References

- LangChain Docs: https://python.langchain.com/
- Ollama: https://ollama.ai/
- FAISS: https://github.com/facebookresearch/faiss
- FastAPI: https://fastapi.tiangolo.com/

---

## Support

For issues or questions:

1. Check the Troubleshooting section
2. Review logs: Check console output for error messages
3. Verify all prerequisites are installed
4. Test each component individually

---

**Last Updated:** April 2024
**Version:** 1.0
