# tests/test_api.py

import io
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from pathlib import Path

from main import app

client = TestClient(app)


# ------------------------------
#  Test: /health
# ------------------------------
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ------------------------------
#  Test: /documents (success)
# ------------------------------
@patch("main.rag_service")
def test_upload_document_success(mock_rag_service, tmp_path):
    # Mock RAG ingestion
    mock_rag_service.ingest_document.return_value = "mock-doc-id"

    # Fake PDF file
    fake_pdf = io.BytesIO(b"%PDF-1.4 Fake PDF content")
    files = {"file": ("test.pdf", fake_pdf, "application/pdf")}

    # Call API
    response = client.post("/documents", files=files)

    assert response.status_code == 200
    result = response.json()
    assert result["document_id"] == "mock-doc-id"
    assert "Document ingested successfully." in result["message"]


# ------------------------------
#  Test: /documents (invalid file type)
# ------------------------------
def test_upload_invalid_file_type():
    fake_txt = io.BytesIO(b"hello world")
    files = {"file": ("test.txt", fake_txt, "text/plain")}

    response = client.post("/documents", files=files)

    assert response.status_code == 400
    assert "Only PDF files are supported." in response.json()["detail"]


# ------------------------------
#  Test: /query (success)
# ------------------------------
@patch("main.rag_service")
def test_query_success(mock_rag_service):
    # Mock answer
    mock_rag_service.query.return_value = "This is a mocked answer."

    payload = {
        "question": "What is the content?",
        "document_id": "doc123",
        "top_k": 4
    }

    response = client.post("/query", json=payload)

    assert response.status_code == 200
    assert response.json()["answer"] == "This is a mocked answer."


# ------------------------------
#  Test: /query (empty question)
# ------------------------------
def test_query_empty_question():
    payload = {
        "question": "   ",
        "document_id": "doc123",
        "top_k": 4
    }

    response = client.post("/query", json=payload)

    assert response.status_code == 400
    assert "Question must not be empty." in response.json()["detail"]
