from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env file
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_CHAT_MODEL = "gpt-4.1-mini"
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

    VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", "chroma_db"))
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploaded_docs"))

settings = Settings()
