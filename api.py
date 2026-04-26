import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List
from App import rag_system, format_docs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Define Pydantic models
class QARequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    question: str
    answer: str

class InitializationStatus(BaseModel):
    status: str
    message: str
    initialized: bool
    documents_count: int = 0
    chunks_count: int = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    logger.info("FastAPI startup event triggered")
    success = rag_system.initialize()
    if not success:
        logger.error("Failed to initialize RAG system during startup")
        # The app will start but RAG endpoints will return errors
    else:
        logger.info("RAG system initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("FastAPI shutdown event triggered")
    logger.info("Cleaning up RAG system")

app = FastAPI(
    title="Document RAG QA API",
    description="RAG-based Q&A API for document analysis and information retrieval",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", tags=["Health"])
def health():
    """Health check endpoint."""
    import os
    # Simple health check: verify GROQ API key presence and RAG system state
    groq_key = os.getenv("GROQ_API_KEY")
    try:
        import sentence_transformers as _st  # type: ignore
        local_embed = True
    except Exception:
        local_embed = False

    status = {
        "status": "ok",
        "message": "Document RAG QA API is running.",
        "groq": "configured" if groq_key else "missing",
        "local_embeddings": "available" if local_embed else "missing",
        "rag_system": "initialized" if rag_system.is_initialized else "not_initialized",
    }
    return status

@app.get("/status", response_model=InitializationStatus, tags=["Status"])
def system_status():
    """Get RAG system initialization status."""
    return {
        "status": "ready" if rag_system.is_initialized else "not_ready",
        "message": "RAG system is ready for queries" if rag_system.is_initialized else "RAG system not initialized",
        "initialized": rag_system.is_initialized,
        "documents_count": len(rag_system.documents) if rag_system.documents else 0,
        "chunks_count": len(rag_system.chunks) if rag_system.chunks else 0,
    }

@app.post("/qa", response_model=QAResponse, tags=["QA"])
def answer_question(request: QARequest):
    """Answer a question using the RAG system."""
    try:
        # Check if RAG system is initialized
        if not rag_system.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="RAG system not initialized. Please check server logs and restart."
            )
        
        logger.info("Received /qa request")
        logger.info("Question: %s", request.question)
        
        # Get vectorstore for similarity search
        vectorstore = rag_system.get_vectorstore()
        retrieved_docs = vectorstore.similarity_search(request.question, k=3)
        logger.info("Retrieved %d document chunks", len(retrieved_docs))
        for index, doc in enumerate(retrieved_docs, start=1):
            title = doc.metadata.get("title", "Unknown Source")
            logger.info("  chunk %d source=%s len=%d", index, title, len(doc.page_content))
            logger.info("  content preview: %s", doc.page_content[:200].replace("\n", " "))

        # Format the context like the RAG chain does
        formatted_context = format_docs(retrieved_docs)
        logger.info("Formatted context length: %d", len(formatted_context))
        logger.info("Formatted context preview: %s", formatted_context[:500].replace("\n", " "))

        # Get RAG chain and invoke
        logger.info("Running RAG chain for incoming question")
        rag_chain = rag_system.get_rag_chain()
        result = rag_chain.invoke(request.question)
        logger.info("RAG chain invoke complete")

        # Extract unique sources
        sources = [
            doc.metadata.get("title", "Unknown Source")
            for doc in result.get("source_documents", [])
        ]
        unique_sources = list(dict.fromkeys(sources))

        logger.info("Returning answer")
        logger.debug("Sources: %s", unique_sources)

        return {
            "question": request.question,
            "answer": result["answer"],
        }
    except HTTPException:
        raise
    except ConnectionError as e:
        logger.error("Connection error: %s", str(e))
        raise HTTPException(
            status_code=503,
            detail=f"Error connecting to LLM model: {str(e)}"
        )
    except Exception as e:
        logger.error("Error processing question: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)