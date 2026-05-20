# Smart Document Assistant

LangChain-powered intelligent document processing and Q&A system with RAG.

## Live Demo

[**View Demo**](https://yoon-k.github.io/langchain-doc-assistant/)

## Features

- **Multi-Format Processing**: PDF, DOCX, TXT, Markdown, CSV, XLSX, HTML support
- **Hybrid Search**: Vector similarity (Chroma / FAISS / in-memory) and keyword fallback
- **RAG Question Answering**: LCEL pipeline with source citations
- **Summarization**: Extractive document summarization
- **Entity Extraction**: People, organizations, dates, emails, URLs, phone numbers
- **Document Comparison**: Token-overlap similarity with shared/unique terms
- **Pluggable LLM**: Anthropic (default) or OpenAI, with keyword-router fallback when no keys are present
- **Switchable vector backend**: `VECTOR_STORE=in_memory|chroma|faiss`

## Architecture

```
langchain-doc-assistant/
├── app/
│   ├── agents/document_agent.py     # LCEL tool-calling agent + fallback
│   ├── chains/qa_chain.py           # LCEL RAG pipeline + vector backends
│   ├── processors/document_processor.py  # PDF / DOCX / TXT / MD / CSV / XLSX / HTML
│   ├── tools/document_tools.py      # BaseTool subclasses with args_schema
│   └── api.py                       # Flask app factory
├── tests/test_smoke.py              # Boots the app via factory
├── pyproject.toml                   # uv / pip install -e ".[dev,rag]"
├── Dockerfile, .dockerignore
├── Makefile
└── .github/workflows/ci.yml         # Ruff + pytest, no [rag] needed
```

## Installation

```bash
git clone https://github.com/MUSE-CODE-SPACE/langchain-doc-assistant.git
cd langchain-doc-assistant

# Base install (fast; no ML deps)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Optional full RAG stack (Chroma + FAISS + sentence-transformers)
pip install -e ".[dev,rag]"
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,rag]"
```

## Running

```bash
# Dev server (Flask)
python -m app.api
# or via Makefile
make dev

# Production (gunicorn)
gunicorn --bind 0.0.0.0:5000 --workers 2 app.api:app
```

### Docker

```bash
make docker-build      # builds doc-assistant:dev
make docker-run        # runs on PORT=5000
```

## Configuration

Copy `.env.example` to `.env` and fill in any values you need:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | auto-detect | `anthropic`, `openai`, or `none` |
| `ANTHROPIC_API_KEY` | — | Required for `anthropic` provider |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Override model |
| `OPENAI_API_KEY` | — | Required for `openai` provider |
| `OPENAI_MODEL` | `gpt-4o-mini` | Override model |
| `VECTOR_STORE` | `in_memory` | `in_memory`, `chroma`, `faiss` |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where Chroma writes |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `PORT` | `5000` | HTTP port |
| `FLASK_DEBUG` | `1` | Dev-server debug flag |

When neither `ANTHROPIC_API_KEY` nor `OPENAI_API_KEY` is set the assistant runs
its built-in keyword router and the RAG chain returns extractive answers - so
the demo and CI keep working with no external network calls.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Index / static demo |
| `/api/health` | GET | Service status + `llm_enabled` |
| `/api/chat` | POST | Agent chat: `{session_id, message}` |
| `/api/documents` | GET | List uploaded documents |
| `/api/documents` | POST | Upload (multipart form, field `file`) |
| `/api/documents/<doc_id>` | GET | Document metadata |
| `/api/documents/<doc_id>` | DELETE | Remove document |
| `/api/documents/<doc_id>/active` | POST | Set active document for the session |
| `/api/query` | POST | RAG query: `{question, document_id?}` |
| `/api/session/reset` | POST | Reset a session's conversation state |

## Tooling

```bash
make test         # pytest
make lint         # ruff check
make format       # ruff format + ruff check --fix
make clean        # remove venv + caches
```

CI (GitHub Actions) installs the base `[dev]` extras (no `[rag]`) and runs
`ruff check` + `pytest`.

## Tech Stack

- **LangChain 0.3** (`langchain`, `langchain-core`, `langchain-community`)
- **LLMs**: `langchain-anthropic`, `langchain-openai`
- **Vector stores**: in-memory cosine (numpy), `langchain-chroma`, FAISS (optional)
- **Embeddings**: `FakeEmbeddings` (default) or HuggingFace sentence-transformers
- **Doc parsing**: `pypdf`, `python-docx`, `openpyxl`, `beautifulsoup4`, `markdown`
- **Web**: Flask 3 + flask-cors, gunicorn (server extras)

## License

MIT License
