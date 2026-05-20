"""
Document Tools - LangChain tools for document operations.

Each tool is a ``langchain_core.tools.BaseTool`` subclass with an explicit
``args_schema`` (pydantic) so the LCEL ``create_tool_calling_agent`` can wire
them up cleanly.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.processors.document_processor import (
    DocumentChunk,
    ProcessedDocument,
)


# ---------------------------------------------------------------------------
# In-memory document store (lightweight; the RAG chain handles vector search)
# ---------------------------------------------------------------------------
class DocumentStore:
    """Simple in-memory document store with keyword fallback search."""

    def __init__(self) -> None:
        self.documents: dict[str, ProcessedDocument] = {}
        self.chunks: dict[str, DocumentChunk] = {}

    def add_document(self, doc: ProcessedDocument) -> str:
        self.documents[doc.document_id] = doc
        for chunk in doc.chunks:
            self.chunks[chunk.chunk_id] = chunk
        return doc.document_id

    def get_document(self, doc_id: str) -> ProcessedDocument | None:
        return self.documents.get(doc_id)

    def search_chunks(
        self,
        query: str,
        top_k: int = 5,
        doc_id: str | None = None,
    ) -> list[DocumentChunk]:
        """Simple keyword scoring (used as fallback when no vector backend)."""
        query_terms = [t for t in query.lower().split() if t]
        scored: list[tuple[int, DocumentChunk]] = []

        for chunk in self.chunks.values():
            if doc_id and chunk.document_id != doc_id:
                continue
            content_lower = chunk.content.lower()
            score = sum(1 for term in query_terms if term in content_lower)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def list_documents(self) -> list[dict[str, Any]]:
        return [
            {
                "document_id": doc.document_id,
                "filename": doc.metadata.filename,
                "type": doc.metadata.doc_type.value,
                "word_count": doc.metadata.word_count,
                "chunk_count": len(doc.chunks),
            }
            for doc in self.documents.values()
        ]

    def delete_document(self, doc_id: str) -> bool:
        doc = self.documents.pop(doc_id, None)
        if doc is None:
            return False
        for chunk in doc.chunks:
            self.chunks.pop(chunk.chunk_id, None)
        return True

    def reset(self) -> None:
        self.documents.clear()
        self.chunks.clear()


# Process-wide store. Tests reset via ``document_store.reset()``.
document_store = DocumentStore()


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
class ListDocumentsInput(BaseModel):
    """Input for listing documents (no arguments)."""


class ListDocumentsTool(BaseTool):
    """List uploaded documents."""

    name: str = "list_documents"
    description: str = "List all uploaded documents available to the assistant."
    args_schema: type[BaseModel] = ListDocumentsInput

    def _run(self) -> str:
        docs = document_store.list_documents()
        if not docs:
            return json.dumps(
                {"documents": [], "message": "No documents uploaded yet."}, indent=2
            )
        return json.dumps({"documents": docs, "total_count": len(docs)}, indent=2)


class SearchDocumentsInput(BaseModel):
    """Input for document search."""

    query: str = Field(description="Search query to find relevant content")
    document_id: str | None = Field(
        default=None, description="Restrict search to this document ID"
    )
    top_k: int = Field(default=5, description="Maximum number of results to return")


class SearchDocumentsTool(BaseTool):
    """Keyword search across uploaded documents."""

    name: str = "search_documents"
    description: str = (
        "Search through uploaded documents to find relevant passages by keyword."
    )
    args_schema: type[BaseModel] = SearchDocumentsInput

    def _run(
        self,
        query: str,
        document_id: str | None = None,
        top_k: int = 5,
    ) -> str:
        chunks = document_store.search_chunks(query, top_k=top_k, doc_id=document_id)
        if not chunks:
            return json.dumps(
                {"results": [], "message": "No matching content found."}, indent=2
            )

        results = []
        for chunk in chunks:
            doc = document_store.get_document(chunk.document_id)
            snippet = (
                chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content
            )
            results.append(
                {
                    "content": snippet,
                    "document": doc.metadata.filename if doc else "Unknown",
                    "document_id": chunk.document_id,
                    "page": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                }
            )

        return json.dumps({"results": results, "count": len(results)}, indent=2)


class QueryDocumentInput(BaseModel):
    """Input for RAG-style question answering."""

    question: str = Field(description="Question to answer using the documents")
    document_id: str | None = Field(
        default=None, description="Restrict to a single document ID"
    )
    top_k: int = Field(default=4, description="How many chunks to retrieve")


class QueryDocumentTool(BaseTool):
    """Answer questions using retrieval-augmented generation."""

    name: str = "query_document"
    description: str = (
        "Answer a natural-language question using retrieval-augmented generation "
        "over the uploaded documents. Returns the answer plus the sources used."
    )
    args_schema: type[BaseModel] = QueryDocumentInput

    def _run(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 4,
    ) -> str:
        # Imported lazily to avoid a circular import (chain imports tools' store).
        from app.chains.qa_chain import DocumentQAChain

        chain = DocumentQAChain()
        result = chain.answer(
            question=question, document_id=document_id, top_k=top_k
        )
        return json.dumps(
            {
                "question": result.question,
                "answer": result.answer,
                "confidence": result.confidence,
                "sources": result.sources,
            },
            indent=2,
        )


class SummarizeDocumentInput(BaseModel):
    """Input for document summarization."""

    document_id: str = Field(description="ID of the document to summarize")
    max_length: int = Field(default=500, description="Approximate max summary length")


class SummarizeDocumentTool(BaseTool):
    """Extractive summarization of a document."""

    name: str = "summarize_document"
    description: str = "Generate an extractive summary of a single document."
    args_schema: type[BaseModel] = SummarizeDocumentInput

    def _run(self, document_id: str, max_length: int = 500) -> str:
        doc = document_store.get_document(document_id)
        if not doc:
            return json.dumps({"error": "Document not found"}, indent=2)

        content = doc.raw_content
        sentences = [s.strip() for s in re.split(r"[.!?]+", content) if len(s.strip()) > 20]

        word_freq: dict[str, int] = {}
        for sentence in sentences:
            for word in sentence.lower().split():
                if len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1

        scored = sorted(
            (
                (sum(word_freq.get(w.lower(), 0) for w in s.split()), s)
                for s in sentences
            ),
            key=lambda x: x[0],
            reverse=True,
        )

        summary_parts: list[str] = []
        current_length = 0
        for _, sentence in scored[:10]:
            if current_length + len(sentence) >= max_length:
                continue
            summary_parts.append(sentence)
            current_length += len(sentence)

        summary = ". ".join(summary_parts) + ("." if summary_parts else "")

        return json.dumps(
            {
                "document": doc.metadata.filename,
                "summary": summary,
                "word_count": doc.metadata.word_count,
                "original_length": len(content),
                "summary_length": len(summary),
            },
            indent=2,
        )


class ExtractEntitiesInput(BaseModel):
    """Input for entity extraction."""

    document_id: str = Field(description="ID of the document to scan")
    entity_types: list[str] | None = Field(
        default=None,
        description=(
            "Optional list to filter results to "
            "(persons, organizations, locations, dates, emails, urls, phone_numbers)"
        ),
    )


class ExtractEntitiesTool(BaseTool):
    """Extract named entities from a document via regex heuristics."""

    name: str = "extract_entities"
    description: str = (
        "Extract named entities (people, organizations, locations, dates, emails, URLs) "
        "from a document."
    )
    args_schema: type[BaseModel] = ExtractEntitiesInput

    def _run(
        self,
        document_id: str,
        entity_types: list[str] | None = None,
    ) -> str:
        doc = document_store.get_document(document_id)
        if not doc:
            return json.dumps({"error": "Document not found"}, indent=2)

        content = doc.raw_content
        entities: dict[str, list[str]] = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "emails": [],
            "urls": [],
            "phone_numbers": [],
        }

        entities["emails"] = sorted(
            set(re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", content))
        )
        entities["urls"] = sorted(
            set(re.findall(r"https?://[^\s<>\"{}|\\^`\[\]]+", content))
        )

        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",
            r"\d{1,2}/\d{1,2}/\d{2,4}",
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}",
        ]
        dates: list[str] = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, content))
        entities["dates"] = sorted(set(dates))

        entities["phone_numbers"] = sorted(
            set(
                re.findall(
                    r"[+]?[(]?[0-9]{1,3}[)]?[-\s.]?[0-9]{3}[-\s.]?[0-9]{4,6}", content
                )
            )
        )

        capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content)
        stop_words = {
            "The",
            "This",
            "That",
            "These",
            "Those",
            "Here",
            "There",
            "When",
            "Where",
            "What",
            "How",
            "Why",
        }
        for phrase in set(capitalized):
            if phrase in stop_words or len(phrase) < 3:
                continue
            if any(kw in phrase for kw in ["Inc", "Corp", "LLC", "Ltd", "Company", "Group"]):
                entities["organizations"].append(phrase)
            elif len(phrase.split()) == 2:
                entities["persons"].append(phrase)
            elif len(phrase.split()) == 1:
                entities["locations"].append(phrase)

        for key, values in entities.items():
            entities[key] = sorted(set(values))[:20]

        if entity_types:
            wanted = {t.lower() for t in entity_types}
            entities = {k: v for k, v in entities.items() if k in wanted}

        return json.dumps(
            {"document": doc.metadata.filename, "entities": entities}, indent=2
        )


class CompareDocumentsInput(BaseModel):
    """Input for comparing two documents."""

    document_id_1: str = Field(description="First document ID")
    document_id_2: str = Field(description="Second document ID")


class CompareDocumentsTool(BaseTool):
    """Compare two documents using term overlap as a similarity proxy."""

    name: str = "compare_documents"
    description: str = "Compare two documents to find common terms and differences."
    args_schema: type[BaseModel] = CompareDocumentsInput

    def _run(self, document_id_1: str, document_id_2: str) -> str:
        doc1 = document_store.get_document(document_id_1)
        doc2 = document_store.get_document(document_id_2)
        if not doc1 or not doc2:
            return json.dumps({"error": "One or both documents not found"}, indent=2)

        words1 = {w for w in doc1.raw_content.lower().split() if len(w) > 4}
        words2 = {w for w in doc2.raw_content.lower().split() if len(w) > 4}

        common = sorted(words1 & words2)[:20]
        only_in_1 = sorted(words1 - words2)[:10]
        only_in_2 = sorted(words2 - words1)[:10]
        similarity = len(words1 & words2) / max(len(words1 | words2), 1)

        return json.dumps(
            {
                "document_1": {
                    "filename": doc1.metadata.filename,
                    "word_count": doc1.metadata.word_count,
                },
                "document_2": {
                    "filename": doc2.metadata.filename,
                    "word_count": doc2.metadata.word_count,
                },
                "similarity_score": round(similarity, 3),
                "common_terms": common,
                "unique_to_doc1": only_in_1,
                "unique_to_doc2": only_in_2,
            },
            indent=2,
        )


def get_document_tools() -> list[BaseTool]:
    """Get all document tools."""
    return [
        ListDocumentsTool(),
        SearchDocumentsTool(),
        QueryDocumentTool(),
        SummarizeDocumentTool(),
        ExtractEntitiesTool(),
        CompareDocumentsTool(),
    ]
