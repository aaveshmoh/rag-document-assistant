import os
import json
from dotenv import load_dotenv
from typing import List, Dict
import time

# LangChain / OpenAI imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

print("✅ Libraries imported successfully!")

# Load financial documents from JSON file
with open("data.json", "r") as f:
    financial_documents = json.load(f)


print(f"✅ Created {len(financial_documents)} banking policy documents")
print("\nDocuments:")
for i, doc in enumerate(financial_documents, 1):
    print(f"  {i}. {doc['title']}")


documents = [
    Document(
        page_content=doc['content'],
        metadata={
            'title': doc['title'],
            'source': doc['title'],
            'category': 'banking_policy'
        }
    )
    for doc in financial_documents
]

print(f"✅ Created {len(documents)} LangChain documents")    

print(documents[0].page_content)


# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # Characters per chunk
    chunk_overlap=200,        # Overlap to preserve context
    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)

print(f"✅ Split into {len(chunks)} chunks")
print(f"\nExample chunk:")
print(f"Title: {chunks[0].metadata['title']}")
print(f"Content preview: {chunks[0].page_content[:200]}...")
print(f"Length: {len(chunks[0].page_content)} characters")

for chunk in chunks:
    print(f"Length: {len(chunk.page_content)} characters")



# Local embeddings (Ollama running on localhost)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"   # best embedding model in Ollama
)

# Test embeddings
test_text = "What are the requirements for a business loan?"
test_embedding = embeddings.embed_query(test_text)

print("✅ Local Embeddings model initialized")
print(f"\nTest embedding:")
print(f"  Query: '{test_text}'")
print(f"  Vector dimensions: {len(test_embedding)}")
print(f"  First 5 values: {test_embedding[:5]}")    



print("Creating FAISS vector store...")
start_time = time.time()

# Create FAISS vector store
faiss_vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

faiss_time = time.time() - start_time

print(f"✅ FAISS vector store created in {faiss_time:.2f} seconds")
print(f"   Total vectors: {len(chunks)}")

# Save FAISS index (optional - for reuse)
faiss_vectorstore.save_local("faiss_index")
print("✅ FAISS index saved to disk")


print("Creating ChromaDB vector store...")
start_time = time.time()

# Create ChromaDB vector store
chroma_vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=r"./chroma_db",  # Persistent storage
    collection_name="banking_policies"
)

chroma_time = time.time() - start_time

print(f"✅ ChromaDB vector store created in {chroma_time:.2f} seconds")
print(f"   Total vectors: {len(chunks)}")
print(f"   Persisted to: ./chroma_db")

# Compare creation times
print(f"\n⏱️  Performance Comparison:")
print(f"   FAISS:    {faiss_time:.2f}s")
print(f"   ChromaDB: {chroma_time:.2f}s")


# Test query
query = "What credit score is needed for a commercial loan?"

print(f"🔍 Query: '{query}'")
print("\n" + "="*80)

# FAISS search
print("\n📊 FAISS Results:")
print("="*80)
start_time = time.time()
faiss_results = faiss_vectorstore.similarity_search(query, k=3)
faiss_search_time = time.time() - start_time

for i, doc in enumerate(faiss_results, 1):
    print(f"\nResult {i}:")
    print(f"Source: {doc.metadata['title']}")
    print(f"Content: {doc.page_content[:200]}...")

print(f"\n⏱️  FAISS search time: {faiss_search_time*1000:.2f}ms")


# ChromaDB search
print("\n" + "="*80)
print("\n📊 ChromaDB Results:")
print("="*80)
start_time = time.time()
chroma_results = chroma_vectorstore.similarity_search(query, k=3)
chroma_search_time = time.time() - start_time

for i, doc in enumerate(chroma_results, 1):
    print(f"\nResult {i}:")
    print(f"Source: {doc.metadata['title']}")
    print(f"Content: {doc.page_content[:200]}...")

print(f"\n⏱️  ChromaDB search time: {chroma_search_time*1000:.2f}ms")



# Performance comparison
print("\n" + "="*80)
print("\n⚡ Performance Comparison:")
print(f"   FAISS:    {faiss_search_time*1000:.2f}ms")
print(f"   ChromaDB: {chroma_search_time*1000:.2f}ms")
if faiss_search_time < chroma_search_time:
    speedup = chroma_search_time / faiss_search_time
    print(f"   FAISS is {speedup:.1f}x faster ⚡")
else:
    speedup = faiss_search_time / chroma_search_time
    print(f"   ChromaDB is {speedup:.1f}x faster ⚡")




llm = ChatOllama(
    model="tinyllama:1.1b",
    temperature=0.7,
    base_url="http://localhost:11434"   # optional (default is localhost)
)

# Test
response = llm.invoke("What is the capital of India?")
print(response.content)


# Custom prompt template (LCEL style)
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
    """Format retrieved documents into a string for the prompt."""
    formatted_chunks = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown Source")
        formatted_chunks.append(f"Source: {title}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted_chunks)


# Create retrievers
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})
chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 3})

def make_rag_chain(retriever):
    """Create a simple RAG chain using LCEL (prompt | llm)."""
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
        # We'll also expose the raw source documents for inspection
        source_documents=retriever,
    )

# RAG chains for FAISS and ChromaDB
faiss_rag_chain = make_rag_chain(faiss_retriever)
chroma_rag_chain = make_rag_chain(chroma_retriever)

print("✅ FAISS & ChromaDB RAG chains created using LCEL (prompt | llm)")



# Test question
question = "What credit score is required for a commercial loan and what documentation is needed?"

print(f"\n{'='*80}")
print(f"💬 Question: {question}")
print(f"{'='*80}")

# Get answer from LCEL RAG chain (FAISS)
result = faiss_rag_chain.invoke(question)

print("\n🤖 Answer:")
print(result["answer"])

print(f"\n{'='*80}")
print("📚 Source Documents:")
print(f"{'='*80}")
for i, doc in enumerate(result["source_documents"], 1):
    print(f"\n{i}. {doc.metadata.get('title', 'Unknown Source')}")
    print(f"   Excerpt: {doc.page_content[:150]}...")


# Test question
question = "What did I ask you?"

print(f"\n{'='*80}")
print(f"💬 Question: {question}")
print(f"{'='*80}")

# Get answer from LCEL RAG chain (FAISS)
result = faiss_rag_chain.invoke(question)

print("\n🤖 Answer:")
print(result["answer"])

print(f"\n{'='*80}")
print("📚 Source Documents:")
print(f"{'='*80}")
for i, doc in enumerate(result["source_documents"], 1):
    print(f"\n{i}. {doc.metadata.get('title', 'Unknown Source')}")
    print(f"   Excerpt: {doc.page_content[:150]}...")



chat_history: List = []  # Holds HumanMessage / AIMessage objects

conv_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful banking assistant. Use the retrieved context and the "
                "conversation history to answer the user's question. If the answer is not "
                "in the context, say you don't know."
            ),
        ),
        (
            "system",
            "Conversation so far:\n{chat_history}",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}\n\nAnswer:",
        ),
    ]
)

def format_chat_history(history: List) -> str:
    """Turn a list of messages into a readable transcript."""
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)


# Conversational RAG chain using Chroma retriever
conv_chain = (
    {
        "question": RunnablePassthrough(),
        "chat_history": RunnableLambda(lambda _: format_chat_history(chat_history)),
        "context": chroma_retriever | RunnableLambda(format_docs),
    }
    | conv_prompt
    | llm
    | StrOutputParser()
)

print("✅ Conversational RAG chain created using LCEL (prompt | llm)")



# Multi-turn conversation demo (Conversational RAG)
conversation = [
    "What are the requirements for a business loan?",
    "What about interest rates?",
    "How long does the approval process take?",
    "What types of collateral are accepted?"
]

print(f"\n{'='*80}")
print("💬 CONVERSATIONAL RAG DEMONSTRATION (LCEL)")
print(f"{'='*80}\n")


for i, question in enumerate(conversation, 1):
    print(f"\n{'='*80}")
    print(f"Turn {i}")
    print(f"{'='*80}")
    print(f"\n👤 User: {question}")

    # Get response from conversational chain
    answer = conv_chain.invoke(question)

    print(f"\n🤖 Assistant: {answer}")

    # Retrieve and show sources (current question)
    print("\n📚 Sources: ", end="")
    source_docs = chroma_vectorstore.similarity_search(question, k=3)
    sources = [doc.metadata.get("title", "Unknown Source") for doc in source_docs]
    print(", ".join(sorted(set(sources))))

    # Update history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    time.sleep(1)  # Brief pause between turns

    question = "what conversation did we have until now?"
    answer = conv_chain.invoke(question)
    print(answer)