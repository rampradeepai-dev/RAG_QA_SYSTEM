from pydantic import BaseModel
from typing import List, Optional


class IngestResponse(BaseModel):
    document_id: str
    message: str


class QueryRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    top_k: int = 4

class SourceDocument(BaseModel):
    document_id: Optional[str]
    page: Optional[int]
    snippet: Optional[str]

class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: List[SourceDocument]
    rerankedsources: List[SourceDocument]


class DocumentItem(BaseModel):
    document_id: str
    filename: str
