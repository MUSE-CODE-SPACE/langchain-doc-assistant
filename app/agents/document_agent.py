"""
Document Agent - LangChain agent for document research.

Supports two execution modes:

1. LLM-backed mode (preferred): an Anthropic or OpenAI chat model wired up via
   LangChain's LCEL ``create_tool_calling_agent`` + ``AgentExecutor``. The agent
   can call the document tools (search, query, summarize, extract entities,
   compare). Selected automatically when an API key is present.
2. Keyword router fallback: zero-dependency rule-based responder that uses the
   tools directly. Used when no LLM credentials are available so the demo keeps
   working without API keys.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.chains.qa_chain import DocumentQAChain
from app.processors.document_processor import DocumentProcessorFactory
from app.tools.document_tools import (
    CompareDocumentsTool,
    ExtractEntitiesTool,
    SearchDocumentsTool,
    SummarizeDocumentTool,
    document_store,
    get_document_tools,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@dataclass
class DocumentContext:
    """Maintains document session context."""

    user_id: str = "default"
    loaded_documents: list[str] = field(default_factory=list)
    current_document_id: str | None = None
    language: str = "en"


def _resolve_llm() -> BaseChatModel | None:
    """Return a chat model based on env config, or ``None`` if unavailable.

    Selection logic:
    - ``LLM_PROVIDER`` env var, one of ``anthropic`` / ``openai`` / ``none``.
    - Otherwise auto-detect from ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``.
    - Returns ``None`` if no provider is usable; the caller falls back to the
      keyword router.
    """
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if provider == "none":
        return None

    if not provider:
        if has_anthropic:
            provider = "anthropic"
        elif has_openai:
            provider = "openai"
        else:
            return None

    try:
        if provider == "anthropic":
            if not has_anthropic:
                logger.warning("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is missing.")
                return None
            from langchain_anthropic import ChatAnthropic

            model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
            return ChatAnthropic(model=model, temperature=0.2)

        if provider == "openai":
            if not has_openai:
                logger.warning("LLM_PROVIDER=openai but OPENAI_API_KEY is missing.")
                return None
            from langchain_openai import ChatOpenAI

            model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
            return ChatOpenAI(model=model, temperature=0.2)
    except Exception as exc:
        logger.warning("Failed to construct LLM for provider %s: %s", provider, exc)
        return None

    logger.warning("Unknown LLM_PROVIDER=%s; falling back to keyword router.", provider)
    return None


class DocumentAssistantAgent:
    """Document research assistant with optional LLM-backed tool calling."""

    SYSTEM_PROMPT = """You are a thorough, citation-driven document research assistant.

Your capabilities:
1. **Document Processing**: Reason over PDF, DOCX, TXT, Markdown and CSV/XLSX uploads.
2. **Search**: Locate relevant passages across the user's documents.
3. **Question Answering (RAG)**: Answer questions grounded in document content.
4. **Summarization**: Produce concise summaries of individual documents.
5. **Entity Extraction**: Surface people, organizations, locations, dates, emails, URLs.
6. **Document Comparison**: Compare two documents and highlight overlap / differences.

Guidelines:
- Always cite the source document filename (and page when available).
- Prefer using the provided tools (search_documents, query_document, summarize_document,
  extract_entities, compare_documents, list_documents) over guessing from memory.
- If you don't have enough information to answer, say so explicitly.
- Suggest a follow-up action (e.g. "want me to summarize this section?") when useful.
- Format answers in Markdown when it improves readability."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        verbose: bool = False,
        history_window: int = 10,
    ) -> None:
        self.verbose = verbose
        self.context = DocumentContext()
        self.conversation_history: list[dict[str, str]] = []
        self._recent_turns: deque[tuple[str, str]] = deque(maxlen=history_window * 2)

        self.tools: list[BaseTool] = get_document_tools()
        self.llm: BaseChatModel | None = llm if llm is not None else _resolve_llm()
        self.qa_chain = DocumentQAChain(llm=self.llm)

        self._agent_executor: Any | None = None
        if self.llm is not None:
            self._agent_executor = self._build_agent_executor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def llm_enabled(self) -> bool:
        """``True`` if requests are answered by an LLM-backed agent."""
        return self._agent_executor is not None

    def upload_document(self, content: bytes, filename: str) -> dict[str, Any]:
        """Process a raw file payload and add it to the shared store."""
        try:
            doc = DocumentProcessorFactory.process_document(content, filename)
            document_store.add_document(doc)
            self.context.current_document_id = doc.document_id
            if doc.document_id not in self.context.loaded_documents:
                self.context.loaded_documents.append(doc.document_id)
            return {
                "success": True,
                "document_id": doc.document_id,
                "filename": doc.metadata.filename,
                "type": doc.metadata.doc_type.value,
                "word_count": doc.metadata.word_count,
                "chunk_count": len(doc.chunks),
                "message": (
                    f"Uploaded '{filename}' ({doc.metadata.word_count} words, "
                    f"{len(doc.chunks)} chunks)"
                ),
            }
        except Exception as exc:
            logger.exception("Document processing failed: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "message": f"Failed to process document: {exc}",
            }

    def get_documents(self) -> list[dict[str, Any]]:
        return document_store.list_documents()

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        ok = document_store.delete_document(doc_id)
        if ok:
            if self.context.current_document_id == doc_id:
                self.context.current_document_id = None
            self.context.loaded_documents = [
                d for d in self.context.loaded_documents if d != doc_id
            ]
            return {"success": True, "message": "Document deleted."}
        return {"success": False, "message": "Document not found."}

    def set_active_document(self, doc_id: str) -> dict[str, Any]:
        doc = document_store.get_document(doc_id)
        if not doc:
            return {"success": False, "message": "Document not found."}
        self.context.current_document_id = doc_id
        return {
            "success": True,
            "message": f"Active document set to '{doc.metadata.filename}'.",
        }

    def chat(self, user_message: str) -> str:
        """Process user message and generate a response."""
        timestamp = datetime.now(UTC).isoformat()
        self.conversation_history.append(
            {"role": "user", "content": user_message, "timestamp": timestamp}
        )
        self._recent_turns.append(("user", user_message))

        try:
            response = self._generate_response(user_message)
        except Exception as exc:
            logger.exception("Agent failed; falling back to keyword router: %s", exc)
            response = self._keyword_response(user_message)

        self.conversation_history.append(
            {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        self._recent_turns.append(("assistant", response))
        return response

    def query(self, question: str, document_id: str | None = None) -> dict[str, Any]:
        """Run a direct RAG query bypassing the conversational agent."""
        target = document_id or self.context.current_document_id
        result = self.qa_chain.answer(question=question, document_id=target)
        return {
            "question": result.question,
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": result.sources,
        }

    def reset(self) -> None:
        """Reset the agent's per-session state (keeps the shared store)."""
        self.context = DocumentContext()
        self.conversation_history = []
        self._recent_turns.clear()

    # ------------------------------------------------------------------
    # LLM-backed agent plumbing
    # ------------------------------------------------------------------
    def _build_agent_executor(self) -> Any:
        from langchain.agents import AgentExecutor, create_tool_calling_agent

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=6,
            handle_parsing_errors=True,
        )

    def _build_chat_history_messages(self) -> list[tuple[str, str]]:
        history = list(self._recent_turns)[:-1]
        return [
            ("human" if role == "user" else "ai", content) for role, content in history
        ]

    def _generate_response(self, message: str) -> str:
        if self._agent_executor is None:
            return self._keyword_response(message)

        result = self._agent_executor.invoke(
            {
                "input": message,
                "chat_history": self._build_chat_history_messages(),
            }
        )
        output = result.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for block in output:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(block))
            return "\n".join(parts).strip() or self._keyword_response(message)
        if isinstance(output, str) and output.strip():
            return output
        return self._keyword_response(message)

    # ------------------------------------------------------------------
    # Keyword router fallback
    # ------------------------------------------------------------------
    def _keyword_response(self, message: str) -> str:
        message_lower = message.lower()

        docs = document_store.list_documents()
        if not docs:
            return self._welcome_message()

        if any(kw in message_lower for kw in ("search", "find", "look for")):
            return self._handle_search(message)
        if any(kw in message_lower for kw in ("summarize", "summary", "brief")):
            return self._handle_summarize()
        if any(
            kw in message_lower
            for kw in ("extract", "entities", "names", "dates", "emails")
        ):
            return self._handle_extract()
        if any(kw in message_lower for kw in ("compare", "difference", "similar")):
            return self._handle_compare()
        if "?" in message or any(
            kw in message_lower for kw in ("what", "who", "when", "where", "why", "how")
        ):
            return self._handle_question(message)

        return self._handle_general(docs)

    def _handle_search(self, message: str) -> str:
        query = message
        for prefix in ("search for", "find", "look for", "search"):
            if prefix in message.lower():
                query = message.lower().split(prefix, 1)[-1].strip()
                break
        tool = SearchDocumentsTool()
        data = json.loads(
            tool._run(query=query, document_id=self.context.current_document_id)
        )
        results = data.get("results", [])
        if not results:
            return (
                "I couldn't find any matching content in the loaded documents. "
                "Try a different search term."
            )
        out = [f"**Search Results for '{query}':**\n"]
        for i, res in enumerate(results, 1):
            out.append(
                f"**{i}. {res['document']}** (page {res['page']})\n"
                f"{res['content']}\n"
            )
        out.append(f"\n*Found {data.get('count', len(results))} relevant sections.*")
        return "\n".join(out)

    def _handle_summarize(self) -> str:
        doc_id = self._resolve_doc_id()
        if not doc_id:
            return "No documents available to summarize."
        tool = SummarizeDocumentTool()
        data = json.loads(tool._run(document_id=doc_id))
        if "error" in data:
            return f"Error: {data['error']}"
        return (
            f"**Summary of '{data['document']}':**\n\n{data['summary']}\n\n"
            f"*Original: {data['original_length']} chars | "
            f"Summary: {data['summary_length']} chars*"
        )

    def _handle_extract(self) -> str:
        doc_id = self._resolve_doc_id()
        if not doc_id:
            return "No documents available for entity extraction."
        tool = ExtractEntitiesTool()
        data = json.loads(tool._run(document_id=doc_id))
        if "error" in data:
            return f"Error: {data['error']}"
        entities = data.get("entities", {})
        out = [f"**Entities Extracted from '{data['document']}':**\n"]
        for label in (
            "persons",
            "organizations",
            "locations",
            "dates",
            "emails",
            "urls",
            "phone_numbers",
        ):
            values = entities.get(label) or []
            if values:
                out.append(
                    f"**{label.replace('_', ' ').title()}:** {', '.join(values[:10])}\n"
                )
        return "\n".join(out)

    def _handle_compare(self) -> str:
        docs = document_store.list_documents()
        if len(docs) < 2:
            return "I need at least two documents to compare. Please upload more."
        tool = CompareDocumentsTool()
        data = json.loads(
            tool._run(
                document_id_1=docs[0]["document_id"],
                document_id_2=docs[1]["document_id"],
            )
        )
        out = [
            "**Document Comparison:**\n",
            f"**Document 1:** {data['document_1']['filename']} "
            f"({data['document_1']['word_count']} words)",
            f"**Document 2:** {data['document_2']['filename']} "
            f"({data['document_2']['word_count']} words)\n",
            f"**Similarity Score:** {data['similarity_score'] * 100:.1f}%\n",
        ]
        if data.get("common_terms"):
            out.append(f"**Common Terms:** {', '.join(data['common_terms'][:15])}")
        if data.get("unique_to_doc1"):
            out.append(f"**Unique to Doc 1:** {', '.join(data['unique_to_doc1'][:10])}")
        if data.get("unique_to_doc2"):
            out.append(f"**Unique to Doc 2:** {', '.join(data['unique_to_doc2'][:10])}")
        return "\n".join(out)

    def _handle_question(self, message: str) -> str:
        target = self.context.current_document_id
        result = self.qa_chain.answer(question=message, document_id=target)
        out = [f"**Answer:**\n{result.answer}\n"]
        if result.sources:
            out.append("**Sources:**")
            for source in result.sources[:3]:
                out.append(
                    f"- {source['document']} (page {source.get('page', 1)})"
                )
        out.append(f"\n*Confidence: {result.confidence * 100:.0f}%*")
        return "\n".join(out)

    def _handle_general(self, docs: list[dict[str, Any]]) -> str:
        out = [f"You have **{len(docs)} document(s)** loaded:\n"]
        for doc in docs[:5]:
            out.append(
                f"- **{doc['filename']}** ({doc['type']}, {doc['word_count']} words)"
            )
        out.append("\n**What would you like to do?**")
        out.append("- Ask a question about your documents")
        out.append("- Search for specific content")
        out.append("- Get a summary")
        out.append("- Extract entities")
        if len(docs) >= 2:
            out.append("- Compare two documents")
        return "\n".join(out)

    def _welcome_message(self) -> str:
        return (
            "Welcome! I'm your Document Assistant. I can help you:\n\n"
            "- **Upload documents** (PDF, Word, Text, Markdown, CSV)\n"
            "- **Search** through document content\n"
            "- **Answer questions** with citations (RAG)\n"
            "- **Summarize** document content\n"
            "- **Extract** entities (people, dates, organizations)\n"
            "- **Compare** two documents\n\n"
            "Upload a document to get started."
        )

    def _resolve_doc_id(self) -> str | None:
        if self.context.current_document_id:
            return self.context.current_document_id
        docs = document_store.list_documents()
        return docs[0]["document_id"] if docs else None


# Backwards compatible alias used by the legacy code & tests.
DocumentAgent = DocumentAssistantAgent


def create_document_agent(
    llm: BaseChatModel | None = None, verbose: bool = False
) -> DocumentAssistantAgent:
    """Factory function to create a document assistant agent."""
    return DocumentAssistantAgent(llm=llm, verbose=verbose)
