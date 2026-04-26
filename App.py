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
from langchain_ollama import OllamaEmbeddings, ChatOllama
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
            embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            logger.info(f"   Using embedding model: {embed_model}")
            self.embeddings = OllamaEmbeddings(model=embed_model)
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
