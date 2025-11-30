from pydantic import BaseModel
from typing import Optional

class IngestResponse(BaseModel):
    document_id: str
    message: str

class QueryRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    top_k: int = 4

class QueryResponse(BaseModel):
    answer: str
