import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from App import rag_chain, faiss_vectorstore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Banking RAG QA API")

class QARequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]

@app.get("/")
def health():
    return {"status": "ok", "message": "Banking RAG QA API is running."}

@app.post("/qa", response_model=QAResponse)
def answer_question(request: QARequest):
    logger.info("Received /qa request")
    logger.info("Question: %s", request.question)

    retrieved_docs = faiss_vectorstore.similarity_search(request.question, k=3)
    logger.info("Retrieved %d document chunks", len(retrieved_docs))
    for index, doc in enumerate(retrieved_docs, start=1):
        title = doc.metadata.get("title", "Unknown Source")
        logger.info("  chunk %d source=%s len=%d", index, title, len(doc.page_content))
        logger.info("  content preview: %s", doc.page_content[:200].replace("\n", " "))

    # Format the context like the RAG chain does
    from App import format_docs
    formatted_context = format_docs(retrieved_docs)
    logger.info("Formatted context length: %d", len(formatted_context))
    logger.info("Formatted context preview: %s", formatted_context[:500].replace("\n", " "))

    logger.info("Running RAG chain for incoming question")
    result = rag_chain.invoke(request.question)
    logger.info("RAG chain invoke complete")

    sources = [
        doc.metadata.get("title", "Unknown Source")
        for doc in result.get("source_documents", [])
    ]
    unique_sources = list(dict.fromkeys(sources))

    logger.info("Returning answer with %d unique source titles", len(unique_sources))
    logger.debug("Sources: %s", unique_sources)

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": unique_sources,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)