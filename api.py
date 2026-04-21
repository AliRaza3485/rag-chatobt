# api.py

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from src.config import validate_config, DATA_FOLDER, TOP_K_RESULTS
from src.document_loader import load_all_and_chunk
from src.vector_store import should_rebuild, create_vector_store, load_vector_store, get_retriever
from src.rag_chain import create_rag_chain, ask_question, clear_history
from src.schemas import (
    QuestionRequest,
    AnswerResponse,
    HealthResponse,
    ClearHistoryRequest,
    ClearHistoryResponse,
)

# Global variables — sirf ek baar initialize honge
rag_chain = None
retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Server start hone pe yeh run hoga.
    RAG system initialize karo — sirf ek baar.
    Yeh 'lifespan' pattern hai — industry standard.
    """
    global rag_chain, retriever

    print("Server starting — RAG system initialize ho raha hai...")

    # Config validate karo
    validate_config()

    # Vector store load ya banao
    if should_rebuild(DATA_FOLDER):
        chunks = load_all_and_chunk(DATA_FOLDER)
        vector_store = create_vector_store(chunks)
    else:
        vector_store = load_vector_store()

    # Retriever aur chain banao
    retriever = get_retriever(vector_store, k=TOP_K_RESULTS)
    rag_chain = create_rag_chain(retriever)

    print("RAG system ready!")

    # Server chalta rahe
    yield

    # Server band hone pe cleanup
    print("Server shutting down...")


# FastAPI app banao
app = FastAPI(
    title="RAG Chatbot API",
    description="Industry level RAG chatbot — LangChain + ChromaDB + Groq",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware add karo
# Yeh allow karta hai ke frontend (React etc) is API ko call kar sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
async def health_check():
    """
    Server chal raha hai ya nahi — check karo.
    Production mein monitoring tools yeh endpoint check karte hain.
    """
    return HealthResponse(
        status="healthy",
        message="RAG Chatbot API is running!"
    )


@app.post(
    "/chat",
    response_model=AnswerResponse,
    tags=["Chat"],
    status_code=status.HTTP_200_OK,
)
async def chat(request: QuestionRequest):
    """
    Main chat endpoint.
    Question bhejo — answer + sources wapas aao.
    """
    # RAG system ready hai ya nahi check karo
    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system abhi initialize nahi hua — thoda wait karo",
        )

    try:
        # Question puchho
        result = ask_question(
            rag_chain,
            request.question,
            request.session_id,
        )

        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=request.session_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}",
        )


@app.post(
    "/clear-history",
    response_model=ClearHistoryResponse,
    tags=["Chat"],
)
async def clear_chat_history(request: ClearHistoryRequest):
    """
    Session history clear karo.
    Nai conversation shuru karne ke liye.
    """
    clear_history(request.session_id)

    return ClearHistoryResponse(
        message="History clear ho gayi!",
        session_id=request.session_id,
    )


@app.get(
    "/sessions/{session_id}/exists",
    tags=["Chat"],
)
async def check_session(session_id: str):
    """
    Session exist karta hai ya nahi check karo.
    """
    from src.rag_chain import session_store
    exists = session_id in session_store

    return {
        "session_id": session_id,
        "exists": exists,
    }