# Card Guide – common commands. Run from the project root: make <target>

# Read values from .env (if the file is missing, the defaults below apply).
-include .env
OLLAMA_BASE_URL ?= http://localhost:11434
OLLAMA_MODEL ?= qwen2.5:7b

.PHONY: help setup pull check run serve cards lint

help:
	@echo "make setup   Install dependencies and download the model"
	@echo "make pull    Download the model from .env"
	@echo "make check   Check that Ollama is running and the model is installed"
	@echo "make run     Start the chat"
	@echo "make serve   Start the Ollama server"
	@echo "make cards   Rebuild the model cards"
	@echo "make lint    Check and format the code with ruff"

setup:
	uv sync
	$(MAKE) pull

pull:
	OLLAMA_HOST=$(OLLAMA_BASE_URL) ollama pull $(OLLAMA_MODEL)

check:
	uv run python -m scripts.check_ollama

# checks Ollama itself before starting (see scripts/check_ollama.py)
run:
	uv run python -m scripts.main

serve:
	OLLAMA_HOST=$(OLLAMA_BASE_URL) ollama serve

cards:
	uv run python scripts/build_cards.py

lint:
	uvx ruff check --fix .
	uvx ruff format .
