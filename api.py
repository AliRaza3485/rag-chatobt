# api.py

from genericpath import exists

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
       It will run after the server starts.
       It will initilize the rag only for once.
       This is 'lifespan' pattern - Industry standard.
    """  
    global rag_chain, retriever

    print("Server starting  RAG system is getting initialized...")
    # Config the Validate
    validate_config()
    # Load or create the vector store
    if should_rebuild(DATA_FOLDER):
        chunks = load_all_and_chunk(DATA_FOLDER)
        vector_store = create_vector_store(chunks)
    else:
        vector_store = load_vector_store()
    # Create the retriever and RAG chain
    retriever = get_retriever(vector_store, k=TOP_K_RESULTS)   
    rag_chain = create_rag_chain(retriever) 
    print("RAG system is ready!")
    # Server will keep running 
    yield
    # Cleanup when server is shutting down
    print("Server is shutting down...")


# # FastAPI app banao
app = FastAPI(
    title="RAG Chatbot API",
    description="Industry level RAG chatbot  Langchain + ChromaDB + Groq",
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
    Check weather the server is up and RAG system is initialized.
    """
    return HealthResponse(
        status="Healthy" if rag_chain is not None else "Initializing",
        message="RAG Chatbot API is running!" if rag_chain is not None else "RAG system is still initializing, please wait..."
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
    Send question and get answer + sources back.
    """
    # Check if RAG system is ready or not
    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system is still initializing - please wait for a moment",
        )
    try:
        # Ask the question to RAG chain
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
            detail=f"An error occurred while processing your request: {str(e)}",
        )

@app.post(
        "/clear-history",
        response_model=ClearHistoryResponse,
        tags=["Chat"],
)
async def clear_chat_history(request: ClearHistoryRequest):
    """
    Clear the chat history for a session.

    """
    clear_history(request.session_id)
    return ClearHistoryResponse(
        message="History is cleared ",
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

