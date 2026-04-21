# main.py

import os
from src.config import validate_config, DATA_FOLDER, TOP_K_RESULTS
from src.document_loader import load_all_and_chunk
from src.vector_store import should_rebuild, create_vector_store, load_vector_store, get_retriever
from src.rag_chain import create_rag_chain, ask_question, clear_history


def initialize_vector_store():
    if should_rebuild(DATA_FOLDER):
        chunks = load_all_and_chunk(DATA_FOLDER)
        vector_store = create_vector_store(chunks)
    else:
        vector_store = load_vector_store()
    return vector_store


def print_answer(result: dict):
    """
    Answer aur sources print karo — clean format mein.
    """
    print(f"\nBot: {result['answer']}")

    # Sources dikhao agar hain
    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print(f"  - {source}")

    print("-" * 50)


def main():
    print("=" * 50)
    print("RAG Chatbot Starting...")
    print("=" * 50)

    try:
        validate_config()
    except ValueError:
        return

    vector_store = initialize_vector_store()
    retriever = get_retriever(vector_store, k=TOP_K_RESULTS)
    rag_chain = create_rag_chain(retriever)

    session_id = "user_session_1"

    print("\nCommands:")
    print("  'exit'  — chatbot band karo")
    print("  'clear' — history clear karo")
    print("=" * 50)

    while True:
        try:
            question = input("\nAap: ").strip()

            if not question:
                continue

            if question.lower() == "exit":
                print("Goodbye!")
                break

            if question.lower() == "clear":
                clear_history(session_id)
                print("Nai conversation shuru!")
                continue

            print(f"\nQuestion: {question}")
            print("Thinking...")

            result = ask_question(rag_chain, question, session_id)
            print_answer(result)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

        except Exception as e:
            print(f"\nError: {e}")
            print("Dobara try karo.")


if __name__ == "__main__":
    main()