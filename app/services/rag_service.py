# app/services/rag_service.py
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

from app.config import settings
from app.services.llm_service import get_llm
from app.services.embedding_service import get_embedding_model


class RAGService:
    def __init__(self, persist_directory: Path | str = settings.VECTOR_DB_DIR):
        self.persist_directory = str(persist_directory)
        self.embedding_model = get_embedding_model()
        self.llm = get_llm()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
        )

    def _get_vectordb(self) -> Chroma:
        return Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )

    def ingest_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
    ) -> str:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        doc_id = document_id or Path(file_path).stem
        for d in docs:
            d.metadata["document_id"] = doc_id

        chunks = self.text_splitter.split_documents(docs)

        if not chunks:
            # Pages exist but no text (e.g. scanned image-only PDF)
            raise ValueError(
                "No text chunks created from this file."
                "This often happens with scanned/image-only PDFs "
                "without a text layer."
            )

        vectordb = self._get_vectordb()
        vectordb.add_documents(chunks)
        vectordb.persist()

        return doc_id

    def query(
        self,
        question: str,
        document_id: Optional[str] = None,
        k: int = 4,
    ) -> str:
        vectordb = self._get_vectordb()

        search_kwargs = {"k": k}
        if document_id:
            search_kwargs["filter"] = {"document_id": document_id}

        retriever = vectordb.as_retriever(search_kwargs=search_kwargs)

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
        )

        result = qa_chain.invoke({"query": question})
        return (
            result["result"]
            if isinstance(result, dict) and "result" in result
            else result
        )
