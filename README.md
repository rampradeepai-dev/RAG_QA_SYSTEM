# 📘 RAG_QA_SYSTEM — Document Question-Answering using FastAPI, LangChain & OpenAI

This project implements a complete **Retrieval-Augmented Generation (RAG)** workflow using:

- **FastAPI** for backend APIs  
- **LangChain** for retrieval + LLM orchestration  
- **Chroma** as the vector store  
- **OpenAI embeddings + LLM models**  
- **PyPDFLoader** for PDF ingestion  
- **Gradio UI** (optional) for interactive Q&A

Upload a PDF → it gets embedded → ask questions → the system retrieves relevant text and generates an answer grounded in the document.

---

## 🚀 Features

- Upload PDF documents and store them in a **persistent vector database**  
- Automatic text extraction, splitting, embedding, and chunk-level metadata tagging  
- Query documents using RAG pipeline  
- Filter search by `document_id`  
- FastAPI backend with clean REST endpoints:
  - `/documents` — Upload & ingest PDFs  
  - `/query` — Ask questions  
  - `/health` — Service health  
- Optional Gradio frontend for quick testing  
- Complete unit tests using `pytest`  
- Modular architecture (`services`, `models`, `config`, etc.)

---

## 📂 Project Structure

RAG_QA_SYSTEM/<br>
│── app/<br>
│ ├── main.py # FastAPI entrypoint<br>
│ ├── models.py # Request/response models<br>
│ ├── services/<br>
│ │ ├── rag_service.py # Core ingestion + RAG logic<br>
│ │ ├── llm_service.py # OpenAI LLM wrapper<br>
│ │ ├── embedding_service.py # Embedding model loader<br>
│ ├── gradio_frontend.py # Optional UI<br>
│ ├── config.py # Settings (API keys, paths)<br>
│<br>
│── tests/<br>
│ ├── test_api.py # API tests<br>
│ ├── test_rag_basic.py # RAG unit tests<br>
│<br>
│── .venv/ # Virtual environment (ignored)<br>
│── uploaded_docs/ # Stored PDFs<br>
│── chroma_db/ # Vector database<br>
│── requirements.txt<br>
│── .env<br>
│── README.md<br>

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/rampradeepai-dev/RAG_QA_SYSTEM.git
```

### 2. Create & activate virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your .env file

Create .env in the project root:
```bash
OPENAI_API_KEY=your_api_key_here
VECTOR_DB_DIR=chroma_db
```

### 5. Run FastAPI backend
In First Terminal
```bash
py main.py
```

### 6. Run Gradio frontend
In Second Terminal
```bash
py gradio_frontend.py
```

---

## 🧪 Running Tests
```bash
pytest -q
```

#### Tests include:
<li>Mocked ingestion</li>
<li>Mocked LLM responses</li>
<li>PDF upload via TestClient</li>
<li>Validation error checks</li>

---

## 🧠 How the RAG Pipeline Works

### **1. Ingestion**

- The uploaded PDF is saved inside the `uploaded_docs/` directory.  
- Text is extracted using **PyPDFLoader**.  
- The text is chunked using **RecursiveCharacterTextSplitter**.  
- Each chunk is converted into embeddings using **OpenAIEmbeddings**.  
- All embeddings are stored in a persistent **Chroma** vector database, tagged with a `document_id`.

---

### **2. Query**

- The system retrieves the **top-k most relevant chunks**, optionally filtered by `document_id`.  
- These chunks are passed alongside the question to the **RetrievalQA** chain.  
- The LLM generates a **grounded, context-aware answer** using only retrieved content.  
- The API returns the answer as structured JSON.

---

## 🛠 Technologies Used

| Component         | Technology            |
|------------------|------------------------|
| Backend API      | FastAPI               |
| LLM              | OpenAI Chat Models    |
| Embeddings       | OpenAI Embeddings     |
| Vector Database  | Chroma                |
| Document Loader  | PyPDFLoader           |
| RAG Framework    | LangChain             |
| UI               | Gradio (optional)     |
| Testing          | pytest                |

---

## 📸 Example Requests

### **Upload a Document**

```http
POST /documents
Content-Type: multipart/form-data
```

### **Query a Document**
```http
POST /query
{
  "question": "What is the summary of the first chapter?",
  "document_id": "f32e5b4c-1234-4daf-b211-f1a2e54398fa",
  "top_k": 4
}
```

## 📸 Example UI Screens
### **Upload Document**
<img width="1039" height="795" alt="RAG_QA_1" src="https://github.com/user-attachments/assets/932194e3-ec8b-4b7b-9af1-70a93254ee11" />
<br/>
<img width="1041" height="793" alt="RAG_QA_2" src="https://github.com/user-attachments/assets/33d8eed9-5d6e-44d1-86c1-008801a56c78" />
<br/>

### **Ask A Question**
<img width="1041" height="792" alt="RAG_QA_3" src="https://github.com/user-attachments/assets/a965dc8e-79a9-44dd-af2a-9e54bf8149ed" />


## 🤝 Contributing

Thanks for the opportunity.






