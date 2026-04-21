# src/config.py

from dotenv import load_dotenv
import os

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Paths
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DATA_FOLDER = os.getenv("DATA_FOLDER", "./data")

# Models
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval settings
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))


def validate_config():
    """
    Config validate at the start of the application for missing or invalid settings
    """
    errors = []

    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY missing — .env file check karo")

    if not os.path.exists(DATA_FOLDER):
        errors.append(f"Data folder nahi mila: {DATA_FOLDER}")

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append("CHUNK_OVERLAP, CHUNK_SIZE se bada nahi ho sakta")

    if errors:
        print("\n CONFIG ERRORS:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Config validation failed — upar errors fix karo")

    print("Config validated!")