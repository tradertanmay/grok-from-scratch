.PHONY: install run test lint format check eval loadtest

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PYTEST ?= $(VENV)/bin/pytest
RUFF ?= $(VENV)/bin/ruff

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -e ".[dev,mcp]"

run:
	$(PYTHON) -m grok_bot

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

check:
	$(RUFF) check .
	$(RUFF) format --check .
	$(PYTEST) -q

test:
	$(PYTEST) -q

eval:
	$(PYTHON) -m grok_bot.evals --fast

loadtest:
	$(PYTHON) -m grok_bot.loadtest --users 20
