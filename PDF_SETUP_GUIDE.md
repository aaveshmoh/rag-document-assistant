# PDF Document Loading Guide

The RAG application now **requires PDF documents** for the RAG system to function. Here's how to set it up:

## Loading Methods (Priority Order)

### 1. **Single PDF via Environment Variable** (Recommended)

Set the `PDF_FILE_PATH` environment variable to load a specific PDF:

```bash
# In .env file
PDF_FILE_PATH=./documents/sample.pdf
```

Then run the application:

```bash
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

### 2. **Multiple PDFs from documents/ Directory**

Place your PDF files in a `documents/` directory in your project root:

```
RAG-Design-QA/
├── App.py
├── api.py
├── documents/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── resume.pdf
├── .env
└── ...
```

The application will automatically load all PDF files from the `documents/` folder.

> **Important:** At least one PDF must be present in either location for the RAG system to initialize successfully.

## Setup Steps

### Option A: Using Environment Variable (Single PDF)

1. **Prepare your PDF**
   - Ensure you have a PDF file (e.g., `./documents/document.pdf`)

2. **Update .env**

   ```
   GROQ_API_KEY=your_api_key_here
   PDF_FILE_PATH=./documents/document.pdf
   ```

3. **Run the application**
   ```bash
   python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
   ```

### Option B: Using documents/ Directory (Multiple PDFs)

1. **Create documents directory**

   ```bash
   mkdir documents
   ```

2. **Add your PDF files**
   - Place all PDF files in the `documents/` folder
   - Files are processed in alphabetical order
   - At least one PDF must be present

3. **Update .env**

   ```
   GROQ_API_KEY=your_api_key_here
   # PDF_FILE_PATH is optional when using documents/ directory
   ```

4. **Run the application**
   ```bash
   python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
   ```

## How It Works

- **PDF Extraction**: Uses `PyPDFLoader` to extract text from PDFs page by page
- **Metadata**: Each page gets metadata including:
  - `title`: PDF filename (without extension)
  - `source`: Full PDF filename
  - `category`: "pdf_document"
- **Chunking**: Documents are split into chunks (1000 chars, 200 char overlap)
- **Embeddings**: Uses Ollama embeddings (or configured embedder)
- **Vector Store**: Stores in FAISS index for fast retrieval

## Supported File Types

- ✅ PDF files (\*.pdf) - Required
- 📝 Word/DOCX support can be added (optional, requires `python-docx`)

## Logging

Check the application logs to see which files are being loaded:

```
INFO: Found 2 PDF file(s) in documents/
INFO: Loading PDF: document.pdf
INFO: Successfully loaded 45 pages from document.pdf
INFO: Loading PDF: investment_guide.pdf
INFO: Successfully loaded 32 pages from investment_guide.pdf
```

## Adding Word/DOCX Support (Optional)

To add support for Word documents, update `requirements.txt`:

```
python-docx
```

Then modify the `load_documents()` function in `App.py` to include:

```python
from langchain_community.document_loaders import Docx2txtLoader

# In load_documents() function:
docx_files = list(pdf_dir.glob("*.docx"))
for docx_file in docx_files:
    loader = Docx2txtLoader(str(docx_file))
    # ... similar processing as PDFs
```

## Troubleshooting

### No PDF files found - RAG system won't initialize

- **Error**: "No PDF files found. Please add PDFs..."
- **Solution**:
  1. Create `documents/` folder in the project root
  2. Add at least one PDF file to the folder
  3. Or set `PDF_FILE_PATH` environment variable
  4. Restart the application

### PDF not loading

- Check the file path in `PDF_FILE_PATH`
- Ensure the PDF file exists and is readable
- Check application logs for error messages

### FAISS index needs rebuild

If you change documents, you may want to delete the existing FAISS index:

```bash
rm -r faiss_index/
```

Then re-run the application to rebuild the index with new documents.

### Permission issues

Ensure the application has read permissions for the PDF files:

```bash
# On Linux/Mac
chmod 644 documents/*.pdf

# On Windows (via PowerShell)
Get-ChildItem documents/*.pdf | ForEach-Object { icacls $_.FullName /grant:r "$($env:USERNAME):(F)" }
```

## Example API Usage

Once running, query the RAG API:

```bash
curl -X POST "http://localhost:8000/qa" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main technical skills mentioned in the document?"}'
```

Response:

```json
{
  "question": "What are the main technical skills mentioned in the document?",
  "answer": "Based on the document content...",
  "sources": ["document.pdf", "resume.pdf"]
}
```
