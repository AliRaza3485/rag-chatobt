# src/vector_store.py

import os
import hashlib
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL, DATA_FOLDER


# Global variable — model ek baar load ho
_embedding_model = None


def get_embedding_model():
    """
    Embedding model load karo — sirf pehli baar.
    Baad mein same model reuse hoga — reload nahi hoga.
    Yeh Singleton Pattern hai — interview mein zaroor poochha jaata hai!
    """
    global _embedding_model

    # Agar pehle se loaded hai toh wahi return karo
    if _embedding_model is not None:
        return _embedding_model

    # Pehli baar load karo
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    _embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Embedding model loaded!")
    return _embedding_model


def get_data_hash(data_folder: str) -> str:
    """
    Data folder ki files ka hash banao.
    Agar koi file badli ya nai aayi — hash change ho jaayega.
    Yeh batata hai ke vector store rebuild karna chahiye ya nahi.
    """
    hash_data = []

    for filename in sorted(os.listdir(data_folder)):
        filepath = os.path.join(data_folder, filename)
        extension = os.path.splitext(filename)[1].lower()

        if extension in [".txt", ".pdf"]:
            # File ka size aur last modified time lo
            stat = os.stat(filepath)
            hash_data.append(f"{filename}:{stat.st_size}:{stat.st_mtime}")

    # Sab info ko ek hash mein convert karo
    combined = "|".join(hash_data)
    return hashlib.md5(combined.encode()).hexdigest()


def save_hash(hash_value: str):
    """
    Hash ko disk pe save karo — next run mein compare karenge.
    """
    hash_file = os.path.join(CHROMA_DB_PATH, "data_hash.json")
    with open(hash_file, "w") as f:
        json.dump({"hash": hash_value}, f)


def load_saved_hash() -> str:
    """
    Pehle se saved hash lo.
    Agar file nahi hai toh empty string return karo.
    """
    hash_file = os.path.join(CHROMA_DB_PATH, "data_hash.json")
    if not os.path.exists(hash_file):
        return ""
    with open(hash_file, "r") as f:
        data = json.load(f)
    return data.get("hash", "")


def should_rebuild(data_folder: str) -> bool:
    """
    Vector store rebuild karna chahiye ya nahi — yeh decide karo.
    Rebuild karo agar:
    1. chroma_db folder exist nahi karta
    2. Data files badli hain pehle se
    """
    # Vector store exist nahi karta
    if not os.path.exists(CHROMA_DB_PATH):
        print("Vector store nahi mila — naya banega.")
        return True

    # Data files check karo
    current_hash = get_data_hash(data_folder)
    saved_hash = load_saved_hash()

    if current_hash != saved_hash:
        print("Data files badli hain — vector store rebuild hoga.")
        return True

    print("Data same hai — existing vector store use hoga.")
    return False


def create_vector_store(chunks):
    """
    Naya vector store banao aur hash save karo.
    """
    print("Creating vector store...")
    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    # Hash save karo — agli baar compare karenge
    current_hash = get_data_hash(DATA_FOLDER)
    save_hash(current_hash)

    print(f"Vector store created! {len(chunks)} chunks stored.")
    return vector_store


def load_vector_store():
    """
    Existing vector store load karo.
    """
    print("Loading existing vector store...")
    embeddings = get_embedding_model()

    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
    )

    print("Vector store loaded!")
    return vector_store


def get_retriever(vector_store, k=3):
    """
    Retriever banao.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    print("Retriever ready!")
    return retriever