"""
Flask API for the Document Assistant.

Run for development with:

    python -m app.api

For production, prefer running through gunicorn:

    gunicorn --bind 0.0.0.0:5000 --workers 2 app.api:app
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask.wrappers import Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from app.agents.document_agent import DocumentAssistantAgent, create_document_agent
from app.tools.document_tools import document_store

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STATIC = _REPO_ROOT / "static"
_DEFAULT_TEMPLATES = _REPO_ROOT / "templates"

ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "md", "markdown", "csv", "xlsx", "html"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app() -> Flask:
    """Application factory."""
    app = Flask(
        __name__,
        static_folder=str(_DEFAULT_STATIC),
        template_folder=str(_DEFAULT_TEMPLATES),
    )
    CORS(app)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    agents: dict[str, DocumentAssistantAgent] = {}

    def get_agent(session_id: str) -> DocumentAssistantAgent:
        if session_id not in agents:
            agents[session_id] = create_document_agent()
        return agents[session_id]

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    @app.route("/")
    def index() -> Response | str:
        if (_DEFAULT_TEMPLATES / "index.html").exists():
            return render_template("index.html")
        return jsonify(
            {
                "service": "document-assistant",
                "message": "Document Assistant API. See /api/health and /api/documents.",
            }
        )

    @app.route("/static/<path:filename>")
    def serve_static(filename: str) -> Response:
        return send_from_directory(app.static_folder, filename)

    @app.route("/api/health", methods=["GET"])
    def health_check() -> Response:
        sample_agent = next(iter(agents.values()), None)
        llm_enabled = sample_agent.llm_enabled if sample_agent is not None else False
        return jsonify(
            {
                "status": "healthy",
                "service": "document-assistant",
                "timestamp": datetime.now(UTC).isoformat(),
                "active_sessions": len(agents),
                "documents_loaded": len(document_store.documents),
                "llm_enabled": llm_enabled,
                "vector_store": os.environ.get("VECTOR_STORE", "in_memory"),
            }
        )

    @app.route("/api/chat", methods=["POST"])
    def chat() -> tuple[Response, int] | Response:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "default")
        message = data.get("message", "")

        if not message:
            return jsonify({"error": "Message is required"}), 400

        try:
            agent = get_agent(session_id)
            response = agent.chat(message)
            return jsonify(
                {
                    "response": response,
                    "documents": agent.get_documents(),
                    "session_id": session_id,
                    "llm_enabled": agent.llm_enabled,
                }
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/documents", methods=["GET", "POST"])
    def documents() -> tuple[Response, int] | Response:
        if request.method == "GET":
            session_id = request.args.get("session_id", "default")
            agent = get_agent(session_id)
            return jsonify({"documents": agent.get_documents()})

        # POST = upload
        session_id = request.form.get("session_id", "default")
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400
        if not _allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "error": (
                            "File type not allowed. Supported: "
                            + ", ".join(sorted(ALLOWED_EXTENSIONS))
                        )
                    }
                ),
                400,
            )

        try:
            filename = secure_filename(file.filename)
            content = file.read()
            agent = get_agent(session_id)
            result = agent.upload_document(content, filename)
            status = 200 if result.get("success") else 500
            return jsonify(result), status
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/documents/<doc_id>", methods=["GET", "DELETE"])
    def document_detail(doc_id: str) -> tuple[Response, int] | Response:
        session_id = request.args.get("session_id", "default")
        agent = get_agent(session_id)

        if request.method == "DELETE":
            result = agent.delete_document(doc_id)
            status = 200 if result.get("success") else 404
            return jsonify(result), status

        doc = document_store.get_document(doc_id)
        if not doc:
            return jsonify({"error": "Document not found"}), 404
        return jsonify(
            {
                "document_id": doc.document_id,
                "filename": doc.metadata.filename,
                "type": doc.metadata.doc_type.value,
                "size_bytes": doc.metadata.size_bytes,
                "page_count": doc.metadata.page_count,
                "word_count": doc.metadata.word_count,
                "char_count": doc.metadata.char_count,
                "chunk_count": len(doc.chunks),
                "table_count": len(doc.tables),
                "created_at": doc.metadata.created_at.isoformat(),
            }
        )

    @app.route("/api/documents/<doc_id>/active", methods=["POST"])
    def set_active_document(doc_id: str) -> Response:
        session_id = (request.get_json(silent=True) or {}).get(
            "session_id", request.args.get("session_id", "default")
        )
        agent = get_agent(session_id)
        return jsonify(agent.set_active_document(doc_id))

    @app.route("/api/query", methods=["POST"])
    def query_documents() -> tuple[Response, int] | Response:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "default")
        question = data.get("question") or data.get("query")
        document_id = data.get("document_id") or data.get("doc_id")

        if not question:
            return jsonify({"error": "Question is required"}), 400

        try:
            agent = get_agent(session_id)
            result = agent.query(question=question, document_id=document_id)
            return jsonify({"session_id": session_id, **result})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/session/reset", methods=["POST"])
    def reset_session() -> Response:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "default")

        if session_id in agents:
            agents[session_id].reset()
            del agents[session_id]

        return jsonify({"status": "reset", "session_id": session_id})

    return app


app = create_app()


if __name__ == "__main__":
    # Development entry point only. Use gunicorn in production:
    # ``gunicorn --bind 0.0.0.0:5000 --workers 2 app.api:app``.
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
