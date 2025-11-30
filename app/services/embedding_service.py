# app/services/embedding_service.py
from langchain_openai import OpenAIEmbeddings
from app.config import settings

def get_embedding_model():
    """
    Returns an embedding model for Chroma.
    """
    embedding = OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        # dimensions=1536,  # optional override for text-embedding-3 models
    )
    return embedding
