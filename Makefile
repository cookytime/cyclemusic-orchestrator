VENV?=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

.PHONY: help setup env run capture upload run-module freeze lint clean stop oauth

help:
	@echo "Makefile targets:"
	@echo "  setup       Create virtualenv and install requirements"
	@echo "  env         Copy .env.example to .env"
	@echo "  run         Run full pipeline (scripts/run_pipeline.sh)"
	@echo "  capture     Run capture watcher (scripts/watch_analyze.sh)"
	@echo "  upload      Run upload watcher (scripts/watch_upload.sh)"
	@echo "  run-module  Run the main module via venv python"
	@echo "  stop        Stop all pipeline processes (uses sudo if available)"
	@echo "  oauth       One-time librespot OAuth login (prints auth URL)"
	@echo "  freeze      Freeze installed packages to requirements.txt"
	@echo "  lint        Run a basic lint (requires flake8)"
	@echo "  clean       Remove pyc/cache files"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

env:
	@test -f .env || cp .env.example .env && echo "Created .env from .env.example"

run:
	@./scripts/run_pipeline.sh

run-no-transfer:
	@LIBRESPOT_AUTO_TRANSFER=false ./scripts/run_pipeline.sh

capture:
	@./scripts/watch_analyze.sh

upload:
	@./scripts/watch_upload.sh

run-module:
	@$(PYTHON) -m cyclemusic.main

stop:
	@./scripts/stop_all.sh

oauth:
	@./scripts/librespot_oauth.sh

capture-debug:
	@$(PYTHON) scripts/capture_single.py single-test --seconds 10

freeze:
	@$(PIP) freeze > requirements.txt && echo "requirements.txt updated"

lint:
	@$(PIP) install --quiet flake8 || true
	@$(VENV)/bin/flake8 src || true

clean:
	@find . -type f -name "*.pyc" -delete || true
	@rm -rf __pycache__ .pytest_cache || true
