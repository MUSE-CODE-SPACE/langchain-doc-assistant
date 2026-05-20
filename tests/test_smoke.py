"""Smoke tests for the Document Assistant Flask app.

These tests exercise the app without requiring any LLM API keys or heavy ML
dependencies; the agent falls back to the keyword router and the QA chain uses
the in-memory vector backend with ``FakeEmbeddings``.
"""

from __future__ import annotations

import os

import pytest

# Make sure no real LLM provider is selected for tests and that the RAG chain
# uses the dependency-free in-memory backend.
os.environ.setdefault("LLM_PROVIDER", "none")
os.environ.setdefault("VECTOR_STORE", "in_memory")

from app.api import create_app
from app.tools.document_tools import document_store


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("VECTOR_STORE", "in_memory")
    # Ensure each test starts with a clean store (other smoke tests don't load
    # documents, but this keeps the tests order-independent).
    document_store.reset()
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


def test_app_starts() -> None:
    app = create_app()
    assert app is not None
    assert app.name


def test_health_endpoint(client) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "healthy"
    assert payload["service"] == "document-assistant"
    assert "timestamp" in payload
    # No agents have been created yet, so llm_enabled should default to False.
    assert payload["llm_enabled"] is False
    assert payload["vector_store"] == "in_memory"


def test_list_documents_empty(client) -> None:
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert "documents" in payload
    assert isinstance(payload["documents"], list)
    # Fresh app + no uploads => empty list is OK.
    assert payload["documents"] == []


def test_chat_keyword_fallback(client) -> None:
    resp = client.post(
        "/api/chat",
        json={"session_id": "smoke", "message": "What can you do?"},
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert "response" in payload
    assert isinstance(payload["response"], str)
    assert payload["response"].strip()
    # With LLM_PROVIDER=none the agent must operate in fallback mode.
    assert payload.get("llm_enabled") is False


def test_session_reset(client) -> None:
    # Prime a session, then reset it.
    client.post("/api/chat", json={"session_id": "reset-test", "message": "hi"})
    resp = client.post("/api/session/reset", json={"session_id": "reset-test"})
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "reset"
    assert payload["session_id"] == "reset-test"
