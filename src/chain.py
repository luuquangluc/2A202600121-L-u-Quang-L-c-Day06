from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import (
    ENABLE_RERANKER,
    LLM_PROVIDER,
    LLM_MODEL,
    GEMINI_LLM_MODEL,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
)
from src.retriever import get_retriever
from src.reranker import CrossEncoderReranker


def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def _get_llm():
    provider = (LLM_PROVIDER or "openai").lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL, temperature=LLM_TEMPERATURE)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def _build_context(query: str, retriever, reranker: CrossEncoderReranker | None) -> str:
    docs = retriever.invoke(query)
    if reranker is not None:
        docs = reranker.rerank(query, docs)
    return format_docs(docs)


def build_rag_chain():
    """Assemble the full RAG chain: retriever → prompt → LLM → output."""
    retriever = get_retriever()
    reranker = CrossEncoderReranker() if ENABLE_RERANKER else None
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": lambda q: _build_context(q, retriever, reranker),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
