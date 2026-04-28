# api.py

import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
    Server start hone pe run hoga.
    RAG system sirf ek baar initialize hoga.
    Yeh lifespan pattern hai — industry standard.
    """
    global rag_chain, retriever

    print("Server starting — RAG system initialize ho raha hai...")

    validate_config()

    if should_rebuild(DATA_FOLDER):
        chunks = load_all_and_chunk(DATA_FOLDER)
        vector_store = create_vector_store(chunks)
    else:
        vector_store = load_vector_store()

    retriever = get_retriever(vector_store, k=TOP_K_RESULTS)
    rag_chain = create_rag_chain(retriever)

    print("RAG system ready!")
    yield
    print("Server shutting down...")


# FastAPI app banao
app = FastAPI(
    title="RAG Chatbot API",
    description="Industry level RAG chatbot — LangChain + ChromaDB + Groq",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
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
    Server chal raha hai ya nahi check karo.
    """
    return HealthResponse(
        status="healthy" if rag_chain is not None else "initializing",
        message="RAG Chatbot API is running!" if rag_chain is not None else "Still initializing..."
    )


@app.post(
    "/upload",
    tags=["Documents"],
)
async def upload_document(file: UploadFile = File(...)):
    """
    Document upload karo data/ folder mein.
    Supported: PDF, TXT, DOCX, CSV, MD
    """
    allowed = [".pdf", ".txt", ".docx", ".csv", ".md"]
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {allowed}"
        )

    file_path = os.path.join(DATA_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "message": f"{file.filename} upload ho gaya!",
        "filename": file.filename,
    }


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
    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system abhi initialize nahi hua — thoda wait karo",
        )

    try:
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
    Session exist karta hai ya nahi.
    """
    from src.rag_chain import session_store
    exists = session_id in session_store

    return {
        "session_id": session_id,
        "exists": exists,
    }


# Frontend serve karo — sabse aakhir mein
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")