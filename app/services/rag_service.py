# app/services/rag_service.py
from pathlib import Path
from typing import Optional
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder

from app.config import settings
from app.services.llm_service import get_llm
from app.services.embedding_service import get_embedding_model

import tiktoken
import json

tokenizer = tiktoken.get_encoding("cl100k_base")

prompt = PromptTemplate(
    template="""
        You are a question-answering assistant.

        Answer the question using ONLY the provided context.
        If the answer is not supported by the context, say "I don't know".

        Return your response strictly in JSON format:
        {{
        "question": string,
        "answer": string,
        "confidence": number  // value between 0 and 1
        }}

        Context:
        {context}

        Question:
        {question}
        """,
    input_variables=["context", "question"],
)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

class RAGService:
    def __init__(self, persist_directory: Path | str = settings.VECTOR_DB_DIR):
        self.persist_directory = str(persist_directory)
        self.embedding_model = get_embedding_model()
        self.llm = get_llm()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=count_tokens,
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

        return doc_id

    def query(
        self,
        question: str,
        document_id: Optional[str] = None,
        k: int = 4,
    ) -> str:
        vectordb = self._get_vectordb()

        search_kwargs = {"k": 100}
        
        if document_id:
            search_kwargs["filter"] = {"document_id": document_id}

        # Retrieve
        docs = vectordb.similarity_search(question, **search_kwargs)

        if not docs:
            return {"answer": "I don't know", "confidence": 0.0}

        sources = []
        for doc in docs:
            sources.append({
                "document_id": doc.metadata.get("document_id"),
                "page": doc.metadata.get("page"),
                "snippet": doc.page_content[:300]  # limit for payload size
            })
        # Rerank
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        pairs = [(question, d.page_content) for d in docs]
        scores = reranker.predict(pairs)

        reranked_docs = [
            doc for _, doc in sorted(zip(scores, docs), reverse=True)
        ][:k]

        rerankedsources = []
        for doc in reranked_docs:
            rerankedsources.append({
                "document_id": doc.metadata.get("document_id"),
                "page": doc.metadata.get("page"),
                "snippet": doc.page_content[:300]  # limit for payload size
            })

        context = "\n\n".join(d.page_content for d in reranked_docs)

        chain = (
            {
                "context": RunnableLambda(lambda _: context),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)

        parsed = json.loads(answer)

        return {
            "question": question,
            "answer": parsed.get("answer"),
            "confidence": parsed.get("confidence"),
            "sources": sources,
            "rerankedsources": rerankedsources
        }
    


