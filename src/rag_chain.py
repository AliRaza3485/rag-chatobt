# src/rag_chain.py

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from src.config import GROQ_API_KEY, LLM_MODEL

# Global session store
session_store = {}


def get_llm():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.2,
    )
    return llm


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_sources(docs) -> list:
    """
    Retrieved chunks se unique source files nikalo.
    Duplicate sources remove karo.
    """
    sources = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        # Sirf filename rakho — poora path nahi
        filename = source.replace("\\", "/").split("/")[-1]

        if filename not in seen:
            seen.add(filename)
            # Page number bhi nikalo agar PDF hai
            page = doc.metadata.get("page")
            if page is not None:
                sources.append(f"{filename} (page {page + 1})")
            else:
                sources.append(filename)

    return sources


def reformulate_question(llm, question: str, chat_history) -> str:
    """
    History ke basis pe question reformulate karo.
    """
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
    """
    Question puchho — answer + sources return karo.
    Ab dict return karta hai — answer aur sources dono.
    """
    retriever = rag_chain["retriever"]
    llm = rag_chain["llm"]
    answer_chain = rag_chain["answer_chain"]

    # Step 1: History lo
    history = get_session_history(session_id)

    # Step 2: Question reformulate karo
    reformulated_q = reformulate_question(llm, question, history.messages)

    # Step 3: Relevant chunks dhundo
    docs = retriever.invoke(reformulated_q)
    context = format_docs(docs)

    # Step 4: Sources nikalo — kahan se aaya answer
    sources = get_sources(docs)

    # Step 5: Answer generate karo
    answer = answer_chain.invoke({
        "context": context,
        "chat_history": history.messages,
        "question": question,
    })

    # Step 6: History update karo
    history.add_user_message(question)
    history.add_ai_message(answer)

    # Step 7: Answer aur sources dono return karo
    return {
        "answer": answer,
        "sources": sources,
    }


def clear_history(session_id: str = "default"):
    if session_id in session_store:
        del session_store[session_id]
        print("History cleared!")