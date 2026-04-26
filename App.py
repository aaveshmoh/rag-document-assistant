import json
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_groq import ChatGroq

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.json"

with DATA_PATH.open("r", encoding="utf-8") as f:
    financial_documents = json.load(f)

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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
)

chunks = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

# Load FAISS vector store from disk if it exists, otherwise create from documents
faiss_index_path = BASE_DIR / "faiss_index" / "index.faiss"
if faiss_index_path.exists():
    faiss_vectorstore = FAISS.load_local(str(BASE_DIR / "faiss_index"), embeddings, allow_dangerous_deserialization=True)
else:
    faiss_vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    faiss_vectorstore.save_local(str(BASE_DIR / "faiss_index"))

faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})

""" llm = ChatOllama(
    model=os.getenv("OLLAMA_CHAT_MODEL", "gemma:2b"),
    temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
) """

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # or mixtral-8x7b-32768 / gemma2-9b-it
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a knowledgeable banking assistant. Use the retrieved context to "
                "answer user questions about banking policies, loans, and financial products.\n"
                "\nGuidelines:\n"
                "1. Use ONLY the provided context. If the answer is not in the context, say you don't know.\n"
                "2. Be precise and include numbers, rates, and requirements when available.\n"
                "3. Be professional and helpful.\n"
                "4. When possible, mention which policy document you used."
            ),
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}\n\nAnswer:",
        ),
    ]
)

def format_docs(docs: List[Document]) -> str:
    formatted_chunks = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown Source")
        formatted_chunks.append(f"Source: {title}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted_chunks)

def make_rag_chain(retriever):
    return RunnableParallel(
        answer=(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | qa_prompt
            | llm
            | StrOutputParser()
        ),
        source_documents=retriever,
    )

rag_chain = make_rag_chain(faiss_retriever)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("App:app", host="localhost", port=8000, reload=False)
