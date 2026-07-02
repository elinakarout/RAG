PYTHON := python3

MAIN := src.main

FILES_DIR := src

install:
	uv sync

debug:
	uv run $(PYTHON) -m pdb -m $(MAIN)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

run:
	uv run $(PYTHON) -m $(MAIN)

lint:
	uv run flake8 $(FILES_DIR)
	uv run mypy $(FILES_DIR)

.PHONY: install run debug clean lint