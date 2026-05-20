# Document Assistant developer Makefile.
# Run `make help` to list targets.

PYTHON ?= python3.12
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
PORT ?= 5000

.PHONY: help install dev test lint format docker-build docker-run clean

help:
	@echo "Available targets:"
	@echo "  install       Create venv and install runtime + dev dependencies"
	@echo "  dev           Run the Flask dev server on PORT=$(PORT)"
	@echo "  test          Run pytest"
	@echo "  lint          Run ruff (check)"
	@echo "  format        Run ruff format + ruff check --fix"
	@echo "  docker-build  Build the Docker image (tag: doc-assistant:dev)"
	@echo "  docker-run    Run the Docker image, mapping PORT=$(PORT)"
	@echo "  clean         Remove venv and build artifacts"

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/python
	$(PIP) install -e ".[dev,server]"

dev: install
	PORT=$(PORT) $(PY) -m app.api

test: install
	$(PY) -m pytest -v

lint: install
	$(VENV)/bin/ruff check .

format: install
	$(VENV)/bin/ruff format .
	$(VENV)/bin/ruff check --fix .

docker-build:
	docker build -t doc-assistant:dev .

docker-run: docker-build
	docker run --rm -p $(PORT):5000 \
		--env-file .env \
		doc-assistant:dev

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
