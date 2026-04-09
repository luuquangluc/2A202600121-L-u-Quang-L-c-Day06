from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import LLM_PROVIDER, LLM_MODEL, GEMINI_LLM_MODEL, LLM_TEMPERATURE, SYSTEM_PROMPT
from src.retriever import get_retriever


def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def _get_llm():
    provider = (LLM_PROVIDER or "openai").lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL, temperature=LLM_TEMPERATURE)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def build_rag_chain():
    """Assemble the full RAG chain: retriever → prompt → LLM → output."""
    retriever = get_retriever()
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
