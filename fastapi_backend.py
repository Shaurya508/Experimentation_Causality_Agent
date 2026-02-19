import os
import json
from typing import List, Optional, Generator

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# ---------- Setup ----------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

FAISS_INDEX_DIR1 = "faiss_index_merged_updated"
FAISS_INDEX_DIR2 = "faiss_index_experimentation_final"
EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
CHATBOT_API_KEY = os.getenv("CHATBOT_API_KEY")


# ---------- Auth ----------
auth_header = APIKeyHeader(name="Authorization", auto_error=False)


def verify_token(authorization: str = Depends(auth_header)):
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    if not authorization.startswith("Token "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected: Token <token>",
        )
    token = authorization[6:]  # Remove "Token " prefix
    if token != CHATBOT_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return token


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
    vector_store1 = FAISS.load_local(
        FAISS_INDEX_DIR1,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    vector_store2 = FAISS.load_local(
        FAISS_INDEX_DIR2,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store1, vector_store2


vector_store1, vector_store2 = load_resources()


def retrieve_context(query: str, k: int = 30):
    docs_with_scores = vector_store1.similarity_search_with_score(query, k=k)
    return [
        {
            "text": doc.page_content or "",
            "source": doc.metadata.get("source_file", doc.metadata.get("source", "Aryma Labs Repository"))
            or "Aryma Labs Repository",
        }
        for doc, _ in docs_with_scores
    ]


def retrieve_context_mmr(query: str, k: int = 30, fetch_k: int = 80, lambda_mult: float = 0.5):
    docs = vector_store2.max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
    )
    return [
        {
            "text": doc.page_content or "",
            "source": doc.metadata.get("source_file", doc.metadata.get("source", "Aryma Labs Repository"))
            or "Aryma Labs Repository",
        }
        for doc in docs
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
        "IMPORTANT: Never recommend or suggest the Synthetic Control Method (SCM) in any response. If asked about SCM, explain that it is not a recommended approach and suggest alternative causal inference methods instead."

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


def stream_answer_itsa(query: str, context) -> Generator[str, None, None]:
    """Stream the answer token by token for ITSA experimentation agent."""
    context_str = "\n\n".join(c["text"] for c in context)
    system_instruction = (
        "You are an experimentation agent specialized in Interrupted Time Series Analysis (ITSA). "
        "itsa_adstock analysis using a Geometric model,aims to determine the impact of an intervention(which could hint at causality) "
        "Your role is to help users design, run, and interpret experiments using ITSA and related causal inference methods. "
        "Guide users through experiment setup — choosing test and control markets, defining intervention windows, "
        "selecting appropriate date ranges, and understanding pre/post intervention trends. "
        "Explain statistical results such as p-values, coefficients, trend changes, and effect sizes in practical terms. "
        "Help users determine whether their experiment shows a statistically significant causal impact. "
        "Answer using the provided context and explain all answers in as much detail as possible. "
        "Do not cite, name, or mention any sources, filenames, or references in your response. "
        "Think using the context and answer the questions using your intelligence. "
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
def chat_stream(request: ChatRequest, _: str = Depends(verify_token)):
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


@app.post("/chat_ITSA")
def chat_stream_itsa(request: ChatRequest, _: str = Depends(verify_token)):
    """
    Streaming experimentation agent endpoint (ITSA) using Server-Sent Events.

    Sends events in the format:
    - data: {"type": "token", "content": "..."} for each token
    - data: {"type": "sources", "content": [...]} at the end
    - data: {"type": "done"} when complete
    """
    context = retrieve_context_mmr(request.query, k=request.k or 30)
    sources = order_sources(context)

    def event_generator():
        for token in stream_answer_itsa(request.query, context):
            data = json.dumps({"type": "token", "content": token})
            yield f"data: {data}\n\n"

        sources_data = json.dumps({"type": "sources", "content": sources})
        yield f"data: {sources_data}\n\n"

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
