# app/services/llm_service.py
from langchain_openai import ChatOpenAI
from app.config import settings

def get_llm():
    """
    Returns a LangChain ChatOpenAI model for use in RetrievalQA.
    OPENAI_API_KEY must be set in the environment.
    """
    llm = ChatOpenAI(
        model=settings.OPENAI_CHAT_MODEL,
        temperature=0.5,
        # you can optionally add:
        # max_completion_tokens=256,
    )
    return llm
