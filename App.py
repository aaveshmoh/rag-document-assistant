import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List
import logging

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import Iterable
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
import json
import requests
from requests import RequestException


class LocalEmbeddings:
    """Minimal embeddings wrapper using Sentence-Transformers.

    Provides `embed_documents` and `embed_query` methods compatible with
    LangChain's embeddings interface used by FAISS.from_documents.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._backend = None
        self._model_name = model_name
        if SentenceTransformer is not None:
            # Prefer local sentence-transformers
            self._backend = "local"
            self.model = SentenceTransformer(model_name)
        else:
            # Attempt to fall back to OpenAI embeddings if available
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                try:
                    from langchain_openai import OpenAIEmbeddings

                    self._backend = "openai"
                    # Use a reasonable default if model_name looks like an HF model
                    openai_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
                    self.openai = OpenAIEmbeddings(model=openai_model)
                except Exception:
                    raise ImportError(
                        "OpenAI embeddings requested via OPENAI_API_KEY but `langchain-openai` is not installed."
                    )
            else:
                raise ImportError(
                    "sentence-transformers is required for LocalEmbeddings — install it with `pip install sentence-transformers` "
                    "or set OPENAI_API_KEY and install `langchain-openai` to use OpenAI embeddings as a fallback."
                )

    def embed_documents(self, texts: Iterable[str]) -> list:
        if self._backend == "local":
            arr = self.model.encode(list(texts), convert_to_numpy=True)
            return [list(map(float, v)) for v in arr]
        else:
            return self.openai.embed_documents(list(texts))

    def embed_query(self, text: str) -> list:
        if self._backend == "local":
            v = self.model.encode([text], convert_to_numpy=True)[0]
            return list(map(float, v))
        else:
            return self.openai.embed_query(text)

    # Make the object callable so LangChain/FAISS can call it directly
    def __call__(self, text: str) -> list:
        return self.embed_query(text)


class GroqEmbeddings:
    """Embeddings via Groq HTTP API.

    Expects environment variable `GROQ_API_KEY` to be set. The default
    endpoint is https://api.groq.ai/v1/embeddings but can be overridden by
    `GROQ_API_URL`.
    """

    def __init__(self, model: str = "embed-1"):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for GroqEmbeddings")
        self.model = model
        self.url = os.getenv("GROQ_API_URL", "https://api.groq.ai/v1/embeddings")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, payload: dict) -> dict:
        try:
            resp = requests.post(self.url, headers=self.headers, data=json.dumps(payload), timeout=15)
            resp.raise_for_status()
            return resp.json()
        except RequestException as e:
            raise ConnectionError(f"Groq embeddings request failed: {e}")

    def embed_documents(self, texts: Iterable[str]) -> list:
        payload = {"model": self.model, "input": list(texts)}
        data = self._post(payload)
        # Expecting response like: {"data": [{"embedding": [...]}, ...]}
        items = data.get("data") or []
        return [item.get("embedding") for item in items]

    def embed_query(self, text: str) -> list:
        payload = {"model": self.model, "input": [text]}
        data = self._post(payload)
        items = data.get("data") or []
        if not items:
            return []
        return items[0].get("embedding")

    # Callable interface for LangChain compatibility
    def __call__(self, text: str) -> list:
        return self.embed_query(text)
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

def load_documents():
    """Load documents from PDF files only.
    
    Priority:
    1. PDF_FILE_PATH environment variable (single PDF)
    2. documents/ directory (multiple PDFs)
    
    Returns empty list if no PDFs found.
    """
    documents = []
    
    # Check for single PDF from environment variable
    pdf_path = os.getenv("PDF_FILE_PATH")
    if pdf_path:
        pdf_file = Path(pdf_path)
        logger.info(f"file path: {pdf_file.name}")
        if pdf_file.exists():
            logger.info(f"Loading PDF from environment variable: {pdf_file.name}")
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                for doc in documents:
                    doc.metadata["title"] = pdf_file.stem
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["category"] = "pdf_document"
                logger.info(f"Successfully loaded {len(documents)} pages from {pdf_file.name}")
                return documents
            except Exception as e:
                logger.error(f"Error loading PDF from PDF_FILE_PATH: {e}")
    
    # Check for PDF files in the documents directory
    pdf_dir = BASE_DIR / "documents"
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if pdf_files:
            logger.info(f"Found {len(pdf_files)} PDF file(s) in documents/")
            for pdf_file in pdf_files:
                logger.info(f"Loading PDF: {pdf_file.name}")
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    pdf_docs = loader.load()
                    # Add metadata with document title
                    for doc in pdf_docs:
                        doc.metadata["title"] = pdf_file.stem
                        doc.metadata["source"] = pdf_file.name
                        doc.metadata["category"] = "pdf_document"
                    documents.extend(pdf_docs)
                    logger.info(f"Successfully loaded {len(pdf_docs)} pages from {pdf_file.name}")
                except Exception as e:
                    logger.error(f"Error loading PDF {pdf_file.name}: {e}")
    
    if not documents:
        logger.warning("No PDF files found. Please add PDFs to:")
        logger.warning(f"  1. Set PDF_FILE_PATH environment variable, or")
        logger.warning(f"  2. Place PDFs in {pdf_dir} directory")
    
    return documents

def format_docs(docs: List[Document]) -> str:
    """Format documents for the context in the prompt."""
    formatted_chunks = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown Source")
        formatted_chunks.append(f"Source: {title}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted_chunks)

class RAGSystem:
    """Main RAG System class for document processing, embedding, and QA chain."""
    
    def __init__(self):
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
        """Initialize the RAG system with documents, embeddings, and LLM."""
        logger.info("=" * 60)
        logger.info("Starting RAG System Initialization")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load Documents
            logger.info("STEP 1: Loading documents...")
            self.documents = load_documents()
            if not self.documents:
                logger.error("No documents loaded!")
                return False
            logger.info(f"✓ Successfully loaded {len(self.documents)} documents")
            
            # Step 2: Split Documents into Chunks
            logger.info("\nSTEP 2: Creating text chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            )
            self.chunks = text_splitter.split_documents(self.documents)
            logger.info(f"✓ Created {len(self.chunks)} chunks from documents")
            
            # Step 3: Initialize Embeddings
            logger.info("\nSTEP 3: Initializing embeddings model...")
            # Choose embedding backend: prefer Groq if requested, else local, else OpenAI fallback
            use_groq = os.getenv("USE_GROQ_EMBED", "").lower() in ("1", "true", "yes")
            groq_model = os.getenv("GROQ_EMBED_MODEL")
            hf_model = os.getenv("HF_EMBED_MODEL", "all-MiniLM-L6-v2")

            if use_groq or (groq_model and os.getenv("GROQ_API_KEY")):
                groq_model = groq_model or os.getenv("GROQ_EMBED_MODEL", "embed-1")
                logger.info(f"   Using Groq embeddings model: {groq_model}")
                try:
                    self.embeddings = GroqEmbeddings(model=groq_model)
                except Exception as e:
                    logger.error("Failed to initialize Groq embeddings: %s", str(e))
                    raise
            else:
                logger.info(f"   Using embedding model: {hf_model} (sentence-transformers)")
                try:
                    self.embeddings = LocalEmbeddings(model_name=hf_model)
                except Exception as e:
                    logger.error("Failed to initialize local embeddings: %s", str(e))
                    raise
            logger.info("✓ Embeddings model initialized")
            
            # Step 4: Load or Create Vector Store
            logger.info("\nSTEP 4: Setting up vector store...")
            faiss_index_path = BASE_DIR / "faiss_index" / "index.faiss"
            
            if faiss_index_path.exists():
                logger.info(f"   Found existing FAISS index at: {faiss_index_path}")
                logger.info("   Loading from disk...")
                self.vectorstore = FAISS.load_local(
                    str(BASE_DIR / "faiss_index"),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                # Verify embedding dimension matches FAISS index dimension
                try:
                    index_obj = getattr(self.vectorstore, "index", None)
                    if index_obj is None:
                        index_obj = getattr(self.vectorstore, "_faiss_index", None)
                    index_dim = getattr(index_obj, "d", None)
                except Exception:
                    index_dim = None

                try:
                    sample_emb = self.embeddings.embed_query("test")
                    emb_dim = len(sample_emb) if sample_emb else None
                except Exception:
                    emb_dim = None

                if index_dim and emb_dim and index_dim != emb_dim:
                    logger.warning(
                        "Detected FAISS index dimension (%s) != embedding dim (%s). Rebuilding index...",
                        index_dim,
                        emb_dim,
                    )
                    # remove stale index and rebuild
                    import shutil

                    try:
                        shutil.rmtree(str(BASE_DIR / "faiss_index"))
                        logger.info("Removed stale faiss_index directory")
                    except Exception as e:
                        logger.error("Failed to remove faiss_index: %s", str(e))
                    # create new index below (fall through to else branch)
                    self.vectorstore = None
                else:
                    logger.info(f"✓ Loaded existing vector store")
            else:
                logger.info("   No existing index found")
                logger.info("   Creating new FAISS index from documents...")
                self.vectorstore = FAISS.from_documents(
                    documents=self.chunks,
                    embedding=self.embeddings
                )
                logger.info("   Saving index to disk...")
                self.vectorstore.save_local(str(BASE_DIR / "faiss_index"))
                logger.info(f"✓ Created and saved new vector store at: {faiss_index_path}")
            
            # Step 5: Create Retriever
            logger.info("\nSTEP 5: Creating retriever...")
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            logger.info("✓ Retriever created (k=3)")
            
            # Step 6: Initialize LLM
            logger.info("\nSTEP 6: Initializing LLM...")
            llm_model = "llama-3.1-8b-instant"
            logger.info(f"   Using LLM model: {llm_model}")
            self.llm = ChatGroq(
                model=llm_model,
                temperature=0.1,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            logger.info("✓ LLM initialized (Groq)")
            
            # Step 7: Create QA Prompt
            logger.info("\nSTEP 7: Creating QA prompt template...")
            self.qa_prompt = ChatPromptTemplate.from_messages([
            (
            "system",
            (
            "You are a helpful assistant. Use the retrieved PDF context to "
            "answer user questions accurately and comprehensively.\n"
            "\nGuidelines:\n"
            "1. Use ONLY the provided PDF context. If the answer is not in the context, say you don't know.\n"
            "2. Be precise and include specific details, numbers, and facts when available.\n"
            "3. Be professional and helpful.\n"
            "4. When possible, mention which PDF document or section you used for the information."
           ),
                ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}\n\nAnswer:",
    ),
])
            logger.info("✓ QA prompt template created")
            
            # Step 8: Build RAG Chain
            logger.info("\nSTEP 8: Building RAG chain...")
            self.rag_chain = RunnableParallel(
                answer=(
                    {
                        "context": self.retriever | RunnableLambda(format_docs),
                        "question": RunnablePassthrough(),
                    }
                    | self.qa_prompt
                    | self.llm
                    | StrOutputParser()
                ),
                source_documents=self.retriever,
            )
            logger.info("✓ RAG chain built successfully")
            
            self.is_initialized = True
            logger.info("\n" + "=" * 60)
            logger.info("RAG System Initialization Complete! ✓")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Error during RAG System initialization: {e}", exc_info=True)
            self.is_initialized = False
            return False
    
    def get_rag_chain(self):
        """Get the RAG chain if initialized."""
        if not self.is_initialized:
            raise RuntimeError("RAG System not initialized. Call initialize() first.")
        return self.rag_chain
    
    def get_vectorstore(self):
        """Get the vector store if initialized."""
        if not self.is_initialized:
            raise RuntimeError("RAG System not initialized. Call initialize() first.")
        return self.vectorstore
    
    def get_retriever(self):
        """Get the retriever if initialized."""
        if not self.is_initialized:
            raise RuntimeError("RAG System not initialized. Call initialize() first.")
        return self.retriever

# Create global RAG system instance
rag_system = RAGSystem()


if __name__ == "__main__":
    # This file should not be run directly.
    # Use: python -m uvicorn api:app --reload --host localhost --port 8000
    print("Please run the API using: python -m uvicorn api:app --reload --host localhost --port 8000")
