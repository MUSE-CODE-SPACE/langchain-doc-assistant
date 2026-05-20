"""
Retrieval-augmented Question Answering chain.

This module builds an LCEL pipeline of the form::

    {context, question} -> prompt -> llm -> StrOutputParser

The retriever is backed by one of three vector store implementations, selected
via the ``VECTOR_STORE`` env var (default ``in_memory``):

* ``in_memory`` - numpy-based cosine similarity over chunk vectors. Uses
  ``FakeEmbeddings`` so no ML model is required; perfect for tests.
* ``chroma`` - persisted ``langchain_chroma.Chroma`` collection (optional
  ``[rag]`` extras).
* ``faiss`` - ``langchain_community.vectorstores.FAISS`` (optional ``[rag]``
  extras + ``faiss-cpu``).

When no LLM is configured, the chain falls back to an extractive keyword-based
answerer so the chain still returns useful results in tests and offline demos.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from app.tools.document_tools import document_store

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHROMA_DIR = "./chroma_db"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class QAResult:
    """Question answering result."""

    question: str
    answer: str
    confidence: float
    sources: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str | None = None


# ---------------------------------------------------------------------------
# Embedding + vector store selection
# ---------------------------------------------------------------------------
def _resolve_backend() -> str:
    """Return the configured vector backend name (lowercased)."""
    return (os.environ.get("VECTOR_STORE") or "in_memory").strip().lower()


def _resolve_embeddings(backend: str) -> Embeddings:
    """Construct an embeddings object appropriate for the backend."""
    # in_memory uses FakeEmbeddings so tests don't need to download models.
    if backend == "in_memory":
        from langchain_community.embeddings import FakeEmbeddings

        return FakeEmbeddings(size=64)

    # For real backends, prefer sentence-transformers when available.
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as exc:
        logger.warning(
            "HuggingFaceEmbeddings unavailable (%s); using FakeEmbeddings.", exc
        )
        from langchain_community.embeddings import FakeEmbeddings

        return FakeEmbeddings(size=64)


def _build_vector_store(
    documents: list[Document], embeddings: Embeddings, backend: str
) -> VectorStore | None:
    """Build a real (chroma/faiss) vector store, or None for in_memory."""
    if backend == "chroma":
        try:
            from langchain_chroma import Chroma  # type: ignore

            persist_dir = os.environ.get("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR)
            return Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_dir,
            )
        except Exception as exc:
            logger.warning("Chroma backend unavailable (%s); falling back.", exc)
            return None

    if backend == "faiss":
        try:
            from langchain_community.vectorstores import FAISS

            return FAISS.from_documents(documents=documents, embedding=embeddings)
        except Exception as exc:
            logger.warning("FAISS backend unavailable (%s); falling back.", exc)
            return None

    return None


# ---------------------------------------------------------------------------
# Lightweight in-memory retriever (numpy cosine similarity)
# ---------------------------------------------------------------------------
class _InMemoryRetriever:
    """Numpy-based cosine similarity retriever over chunk embeddings."""

    def __init__(self, embeddings: Embeddings, documents: list[Document]) -> None:
        import numpy as np

        self._np = np
        self.embeddings = embeddings
        self.documents = documents

        if documents:
            vectors = embeddings.embed_documents([d.page_content for d in documents])
            self._matrix = np.array(vectors, dtype="float32")
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = self._matrix / norms
        else:
            self._matrix = np.zeros((0, 0), dtype="float32")

    def get_relevant_documents(
        self, query: str, top_k: int = 4
    ) -> list[Document]:
        if len(self.documents) == 0:
            return []
        q_vec = self._np.array(self.embeddings.embed_query(query), dtype="float32")
        norm = self._np.linalg.norm(q_vec)
        if norm == 0:
            return list(self.documents[:top_k])
        q_vec = q_vec / norm
        scores = self._matrix @ q_vec
        order = self._np.argsort(-scores)[:top_k]
        return [self.documents[int(i)] for i in order]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
QA_SYSTEM = (
    "You are a careful document research assistant. Answer the user's question "
    "using ONLY the provided context. Cite the source filenames you used. If "
    "the context does not contain the answer, say you don't have enough "
    "information."
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\n"
            "Answer concisely and cite sources by filename.",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Main chain
# ---------------------------------------------------------------------------
class DocumentQAChain:
    """LCEL retrieval-augmented QA chain over the in-process document store."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm = llm
        self.backend = _resolve_backend()
        self.embeddings = _resolve_embeddings(self.backend)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 4,
    ) -> QAResult:
        """Run retrieval then synthesis."""
        documents = self._collect_documents(document_id)
        if not documents:
            return QAResult(
                question=question,
                answer="No documents have been uploaded yet.",
                confidence=0.0,
                sources=[],
            )

        retrieved = self._retrieve(question, documents, top_k=top_k)
        if not retrieved:
            return QAResult(
                question=question,
                answer="I couldn't find relevant content to answer this question.",
                confidence=0.1,
                sources=[],
            )

        sources = self._format_sources(retrieved)
        if self.llm is not None:
            answer_text = self._invoke_llm(question, retrieved)
            confidence = 0.85
        else:
            answer_text, confidence = self._extractive_answer(
                question, [d.page_content for d in retrieved]
            )

        return QAResult(
            question=question,
            answer=answer_text,
            confidence=confidence,
            sources=sources,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _collect_documents(self, document_id: str | None) -> list[Document]:
        """Build LangChain ``Document`` list from the in-memory store."""
        documents: list[Document] = []
        for chunk in document_store.chunks.values():
            if document_id and chunk.document_id != document_id:
                continue
            stored = document_store.get_document(chunk.document_id)
            filename = stored.metadata.filename if stored else "unknown"
            documents.append(
                Document(
                    page_content=chunk.content,
                    metadata={
                        "document_id": chunk.document_id,
                        "filename": filename,
                        "page": chunk.page_number,
                        "chunk_id": chunk.chunk_id,
                    },
                )
            )
        return documents

    def _retrieve(
        self, query: str, documents: list[Document], top_k: int
    ) -> list[Document]:
        backend = self.backend
        store = _build_vector_store(documents, self.embeddings, backend) if backend in {
            "chroma",
            "faiss",
        } else None

        if store is not None:
            return store.similarity_search(query, k=top_k)

        retriever = _InMemoryRetriever(self.embeddings, documents)
        return retriever.get_relevant_documents(query, top_k=top_k)

    def _format_sources(self, documents: list[Document]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        seen: set[str] = set()
        for doc in documents:
            key = f"{doc.metadata.get('document_id')}::{doc.metadata.get('chunk_id')}"
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "document": doc.metadata.get("filename", "unknown"),
                    "document_id": doc.metadata.get("document_id"),
                    "page": doc.metadata.get("page", 1),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "excerpt": (
                        doc.page_content[:200] + "..."
                        if len(doc.page_content) > 200
                        else doc.page_content
                    ),
                }
            )
        return sources

    def _invoke_llm(self, question: str, documents: list[Document]) -> str:
        """Run the LCEL pipe: context -> prompt -> llm -> parser."""

        def _format(_inp: dict[str, Any]) -> str:
            parts = []
            for doc in documents:
                fname = doc.metadata.get("filename", "unknown")
                page = doc.metadata.get("page", 1)
                parts.append(f"[Source: {fname}, page {page}]\n{doc.page_content}")
            return "\n\n---\n\n".join(parts)

        chain = (
            {
                "context": RunnableLambda(_format),
                "question": RunnablePassthrough(),
            }
            | QA_PROMPT
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(question)

    # ------------------------------------------------------------------
    # Keyword extractive fallback (no LLM available)
    # ------------------------------------------------------------------
    def _extractive_answer(
        self, question: str, contexts: list[str]
    ) -> tuple[str, float]:
        joined = "\n\n".join(contexts)
        keywords = [w.lower() for w in question.split() if len(w) > 3]
        sentences = [s.strip() for s in re.split(r"[.!?]+", joined) if len(s.strip()) > 15]

        scored: list[tuple[int, str]] = []
        for sentence in sentences:
            score = sum(1 for kw in keywords if kw in sentence.lower())
            if score > 0:
                scored.append((score, sentence))

        if not scored:
            return ("I couldn't find a specific answer in the documents.", 0.2)

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for _, s in scored[:2]]
        confidence = min(0.7, scored[0][0] / max(len(keywords), 1))
        return (". ".join(top) + ".", round(confidence, 2))


def create_qa_chain(llm: BaseChatModel | None = None) -> DocumentQAChain:
    """Factory function to create a QA chain."""
    return DocumentQAChain(llm=llm)
