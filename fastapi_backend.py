import os
import json
from typing import List, Optional, Generator

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# ---------- Setup ----------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

FAISS_INDEX_DIR = "faiss_index_merged_updated"
EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"


# ---------- Models ----------
class ChatRequest(BaseModel):
    query: str
    k: Optional[int] = 30


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


# ---------- RAG Core ----------
def load_resources():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


vector_store = load_resources()


def retrieve_context(query: str, k: int = 30):
    docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
    return [
        {
            "text": doc.page_content or "",
            "source": doc.metadata.get("source_file", doc.metadata.get("source", "Aryma Labs Repository"))
            or "Aryma Labs Repository",
        }
        for doc, _ in docs_with_scores
    ]


def generate_answer(query: str, context) -> str:
    context_str = "\n\n".join(c["text"] for c in context)
    system_instruction = (
        "You are a causality agent. Answer using the provided context and explain all the answers in as much detail as possible "
        "Do not cite, name, or mention any sources, filenames, or references in your response. "
        "Think using the context and answer the questions using your intelligence. "
        "Explain causal concepts clearly and precisely from the context only. "
        "Just give the answer directly; there is no need to say 'The context says that ...' or 'According to the context'."
    )
    prompt = f"""
Context:
{context_str}

Question:
{query}

Answer (using only the context above; do not name any sources):
"""
    model = genai.GenerativeModel(
        GENERATION_MODEL,
        system_instruction=system_instruction,
    )
    response = model.generate_content(prompt)
    return response.text or ""


def stream_answer(query: str, context) -> Generator[str, None, None]:
    """Stream the answer token by token."""
    context_str = "\n\n".join(c["text"] for c in context)
    system_instruction = (
        "You are a causality agent. Answer using the provided context and explain all the answers in as much detail as possible. "
        "Do not cite, name, or mention any sources, filenames, or references in your response. "
        "Think using the context and answer the questions using your intelligence. "
        "Explain causal concepts clearly and precisely from the context only. "
        "Just give the answer directly; there is no need to say 'The context says that ...' or 'According to the context'."
    )
    prompt = f"""
Context:
{context_str}

Question:
{query}

Answer (using only the context above; do not name any sources):
"""
    model = genai.GenerativeModel(
        GENERATION_MODEL,
        system_instruction=system_instruction,
    )
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


def order_sources(context) -> List[str]:
    unique_sources = list({c["source"] for c in context})
    # Aryma Labs Repository first, then the rest
    ordered = ["Aryma Labs Repository"] + [
        s for s in unique_sources if s != "Aryma Labs Repository"
    ]
    # De-duplicate while preserving order
    seen = set()
    result: List[str] = []
    for src in ordered:
        if src not in seen:
            seen.add(src)
            result.append(src)
    return result


# ---------- FastAPI App ----------
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.post("/chat", response_model=ChatResponse)
# def chat(request: ChatRequest) -> ChatResponse:
#     """
#     RAG chatbot endpoint (non-streaming).
#     """
#     context = retrieve_context(request.query, k=request.k or 30)
#     answer = generate_answer(request.query, context)
#     sources = order_sources(context)
#     return ChatResponse(answer=answer, sources=sources)


@app.post("/chat")
def chat_stream(request: ChatRequest):
    """
    Streaming RAG chatbot endpoint using Server-Sent Events.

    Sends events in the format:
    - data: {"type": "token", "content": "..."} for each token
    - data: {"type": "sources", "content": [...]} at the end
    - data: {"type": "done"} when complete
    """
    context = retrieve_context(request.query, k=request.k or 30)
    sources = order_sources(context)

    def event_generator():
        # Stream the answer tokens
        for token in stream_answer(request.query, context):
            data = json.dumps({"type": "token", "content": token})
            yield f"data: {data}\n\n"

        # Send sources at the end
        sources_data = json.dumps({"type": "sources", "content": sources})
        yield f"data: {sources_data}\n\n"

        # Signal completion
        done_data = json.dumps({"type": "done"})
        yield f"data: {done_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# Optional root health check
@app.get("/")
def read_root():
    return {"status": "ok", "message": "RAG Chatbot API is running"}
