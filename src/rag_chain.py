# src/rag_chain.py

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from src.config import GROQ_API_KEY, LLM_MODEL
import json
import os

# Global session store — RAM cache
session_store = {}

# History folder — disk pe save hogi
HISTORY_FOLDER = "./chat_histories"


def get_llm():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.2,
    )
    return llm


def save_history_to_disk(session_id: str, history: ChatMessageHistory):
    """
    History ko disk pe save karo.
    Server restart ke baad bhi history milegi.
    """
    # Folder banao agar nahi hai
    os.makedirs(HISTORY_FOLDER, exist_ok=True)

    filepath = os.path.join(HISTORY_FOLDER, f"{session_id}.json")

    # Messages ko simple dict mein convert karo
    messages = []
    for msg in history.messages:
        messages.append({
            "type": msg.type,
            "content": msg.content,
        })

    # JSON file mein save karo
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)


def load_history_from_disk(session_id: str) -> ChatMessageHistory:
    """
    Disk se history load karo.
    Agar file nahi hai toh khali history return karo.
    """
    filepath = os.path.join(HISTORY_FOLDER, f"{session_id}.json")
    history = ChatMessageHistory()

    # File exist nahi karti — nai history
    if not os.path.exists(filepath):
        return history

    # File se messages load karo
    with open(filepath, "r", encoding="utf-8") as f:
        messages = json.load(f)

    # Messages history mein add karo
    for msg in messages:
        if msg["type"] == "human":
            history.add_user_message(msg["content"])
        else:
            history.add_ai_message(msg["content"])

    return history


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Session history lo.
    Pehle RAM check karo — nahi hai toh disk se load karo.
    """
    if session_id not in session_store:
        # Disk se load karo
        session_store[session_id] = load_history_from_disk(session_id)

    return session_store[session_id]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_sources(docs) -> list:
    sources = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        filename = source.replace("\\", "/").split("/")[-1]

        if filename not in seen:
            seen.add(filename)
            page = doc.metadata.get("page")
            if page is not None:
                sources.append(f"{filename} (page {page + 1})")
            else:
                sources.append(filename)

    return sources


def reformulate_question(llm, question: str, chat_history) -> str:
    if not chat_history:
        return question

    history_text = ""
    for msg in chat_history:
        if msg.type == "human":
            history_text += f"Human: {msg.content}\n"
        else:
            history_text += f"Assistant: {msg.content}\n"

    reformulate_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Given the conversation history and a follow-up question,
rewrite the follow-up question to be a standalone question.
If already standalone, return as-is.
Return ONLY the reformulated question — no explanation.

Conversation history:
{history}"""
        ),
        ("human", "{question}"),
    ])

    chain = reformulate_prompt | llm | StrOutputParser()
    return chain.invoke({"history": history_text, "question": question})


def get_answer_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful assistant. Answer the question based ONLY
on the context below. If the answer is not in the context,
say 'I don't know based on the provided documents.'
Do not make up any information.

Context:
{context}"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])


def create_rag_chain(retriever):
    llm = get_llm()
    answer_prompt = get_answer_prompt()
    answer_chain = answer_prompt | llm | StrOutputParser()

    return {
        "retriever": retriever,
        "llm": llm,
        "answer_chain": answer_chain,
    }


def ask_question(rag_chain, question: str, session_id: str = "default") -> dict:
    retriever = rag_chain["retriever"]
    llm = rag_chain["llm"]
    answer_chain = rag_chain["answer_chain"]

    # Step 1: History lo — RAM ya disk se
    history = get_session_history(session_id)

    # Step 2: Question reformulate karo
    reformulated_q = reformulate_question(llm, question, history.messages)

    # Step 3: Chunks dhundo
    docs = retriever.invoke(reformulated_q)
    context = format_docs(docs)

    # Step 4: Sources nikalo
    sources = get_sources(docs)

    # Step 5: Answer generate karo
    answer = answer_chain.invoke({
        "context": context,
        "chat_history": history.messages,
        "question": question,
    })

    # Step 6: History update karo — RAM mein
    history.add_user_message(question)
    history.add_ai_message(answer)

    # Step 7: Disk pe save karo — persist karo
    save_history_to_disk(session_id, history)

    # Step 8: Answer + sources return karo
    return {
        "answer": answer,
        "sources": sources,
    }


def clear_history(session_id: str = "default"):
    """
    RAM aur disk dono se history delete karo.
    """
    # RAM se delete karo
    if session_id in session_store:
        del session_store[session_id]

    # Disk se delete karo
    filepath = os.path.join(HISTORY_FOLDER, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)

    print(f"History cleared: {session_id}")