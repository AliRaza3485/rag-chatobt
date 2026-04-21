# src/document_loader.py

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


# Supported formats aur unke loaders — dictionary mein rakho
# Naya format add karna ho toh sirf yahan ek line add karo
SUPPORTED_LOADERS = {
    ".txt":  lambda path: TextLoader(path, encoding="utf-8"),
    ".pdf":  lambda path: PyPDFLoader(path),
    ".docx": lambda path: Docx2txtLoader(path),
    ".csv":  lambda path: CSVLoader(path, encoding="utf-8"),
    ".md":   lambda path: UnstructuredMarkdownLoader(path),
    ".xlsx": lambda path: UnstructuredExcelLoader(path),
}


def load_document(file_path: str):
    """
    File path se document load karo.
    Extension ke basis pe sahi loader use hoga automatically.
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension not in SUPPORTED_LOADERS:
        raise ValueError(
            f"Unsupported file type: {extension}\n"
            f"Supported formats: {list(SUPPORTED_LOADERS.keys())}"
        )

    loader = SUPPORTED_LOADERS[extension](file_path)
    return loader.load()


def load_all_documents(data_folder: str):
    """
    Poore data folder ki saari supported files load karo.
    """
    all_documents = []
    skipped = []

    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        extension = os.path.splitext(filename)[1].lower()

        if extension in SUPPORTED_LOADERS:
            try:
                print(f"Loading: {filename}")
                docs = load_document(file_path)
                all_documents.extend(docs)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                skipped.append(filename)
        else:
            skipped.append(filename)

    if skipped:
        print(f"Skipped files: {skipped}")

    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Documents ko chunks mein todo.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def load_and_chunk(file_path: str, chunk_size=500, chunk_overlap=50):
    """
    Single file — load + chunk.
    """
    documents = load_document(file_path)
    return chunk_documents(documents, chunk_size, chunk_overlap)


def load_all_and_chunk(data_folder: str, chunk_size=500, chunk_overlap=50):
    """
    Poora folder — load + chunk.
    """
    documents = load_all_documents(data_folder)
    return chunk_documents(documents, chunk_size, chunk_overlap)