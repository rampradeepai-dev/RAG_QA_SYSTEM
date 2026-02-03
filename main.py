from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid

from app.models import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
    DocumentItem,
)

from app.services.rag_service import RAGService
from typing import List
import json


app = FastAPI(
    title="Document RAG Q&A Service",
    version="0.1.0",
    description="Backend service for document ingestion and RAG-based Q&A.",
)

# Allow local frontends / tools to call this API easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RAGService()
INDEX_DIR = Path("index_db")
INDEX_DIR.mkdir(exist_ok=True)
INDEX_PATH = INDEX_DIR / "index.json"


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/documents", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF files are supported."
            ),
        )

    try:
        # Save uploaded file
        doc_id = f"{uuid.uuid4()}"
        file_path = INDEX_DIR / f"{doc_id}.pdf"

        with file_path.open("wb") as f:
            content = await file.read()
            f.write(content)

        # Ingest into vector store
        stored_doc_id = rag_service.ingest_document(
            str(file_path),
            document_id=doc_id,
        )

        # update index.json with filename + doc_id
        index = load_index()
        index.append({"document_id": stored_doc_id, "filename": file.filename})
        save_index(index)

        file_path.unlink(missing_ok=True)

        return IngestResponse(
            document_id=stored_doc_id,
            message="Document ingested successfully.",
        )
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/query", response_model=QueryResponse)
async def query_rag(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty.",
        )

    try:
        result = rag_service.query(
            question=payload.question,
            document_id=payload.document_id,
            k=payload.top_k,
        )
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            confidence=result["confidence"],
            sources=result["sources"],
            rerankedsources=result["rerankedsources"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.get("/documents/index", response_model=List[DocumentItem])
def list_documents():
    index = load_index()
    return [DocumentItem(**item) for item in index]


def load_index() -> List[dict]:
    if INDEX_PATH.exists():
        with INDEX_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_index(entries: List[dict]) -> None:
    with INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
