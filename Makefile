.PHONY: all format lint test tests test_watch integration_tests docker_tests help extended_tests \
	streamlit cli cli_db mcp_client app_and_client seed_properties db_properties

# Default target executed when no arguments are given to make.
all: help

# Python interpreter (override with `make PYTHON=python3` or your venv/conda python)
PYTHON ?= python

######################
# RUN APPS (DEMO)
######################

# Streamlit UI (MCP frontend + graph caller)
STREAMLIT_APP ?= src/mcp/app.py
STREAMLIT_HOST ?= 0.0.0.0
STREAMLIT_PORT ?= 8501

streamlit:
	$(LANGSMITH_ENV_PREFIX) $(PYTHON) -m streamlit run $(STREAMLIT_APP) --server.address $(STREAMLIT_HOST) --server.port $(STREAMLIT_PORT)

# CLI demo runner
CLI_APP ?= src/mcp/cli_agent.py
USER_ID ?= interview_user
CLI_ARGS ?=

cli:
	$(LANGSMITH_ENV_PREFIX) $(PYTHON) $(CLI_APP) --user-id $(USER_ID) $(CLI_ARGS)

# Most common interview/demo mode: skip DB/store dependencies
cli_db:
	$(LANGSMITH_ENV_PREFIX) $(PYTHON) $(CLI_APP) --disable-db --user-id $(USER_ID) $(CLI_ARGS)

# MCP demo client (spawns MCP server via stdio)
MCP_CLIENT_APP ?= src/mcp/mcpclient.py

mcp_client:
	$(LANGSMITH_ENV_PREFIX) $(PYTHON) $(MCP_CLIENT_APP)

seed_properties:
	$(LANGSMITH_ENV_PREFIX) $(PYTHON) -c "from src.agent.common.property_store import get_property_store; s=get_property_store(); rows=s.list_properties(); print(f'loaded={len(rows)} db_enabled={s.db_enabled}')"

db_properties:
	$(LANGSMITH_ENV_PREFIX) $(PYTHON) -c "import os,json,pymysql; from decimal import Decimal; from dotenv import load_dotenv; load_dotenv('.env'); conn=pymysql.connect(host=(os.getenv('DB_HOST','127.0.0.1').strip().strip('\\\"').strip(\"'\")),port=int((os.getenv('DB_PORT','3306').strip().strip('\\\"').strip(\"'\")) or '3306'),user=(os.getenv('DB_USER','').strip().strip('\\\"').strip(\"'\")),password=(os.getenv('DB_PASSWORD','').strip().strip('\\\"').strip(\"'\")),database=(os.getenv('DB_NAME','').strip().strip('\\\"').strip(\"'\")),charset='utf8mb4',autocommit=True,cursorclass=pymysql.cursors.DictCursor); cur=conn.cursor(); cur.execute('SELECT id,title,price,layout,area,city,region,district,bedrooms,features_json FROM property_listings ORDER BY price ASC,id ASC'); rows=cur.fetchall() or []; print(f'rows={len(rows)}'); [print(json.dumps(r, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, Decimal) else str(o))) for r in rows]; cur.close(); conn.close()"

# One-shot: start Streamlit + run MCP client once
# - Streamlit keeps running in foreground
# - MCP client runs once to verify it can pull Redis-backed state via the MCP server
app_and_client:
	@bash -c 'set -euo pipefail; \
		is_port_free() { \
			$(PYTHON) -c "import socket,sys; p=int(sys.argv[1]); s=socket.socket(); s.settimeout(0.2); r=s.connect_ex((\"127.0.0.1\", p)); s.close(); sys.exit(0 if r!=0 else 1)" "$$1"; \
		}; \
		log_file=".streamlit_app_and_client.log"; \
		port="$(STREAMLIT_PORT)"; \
		max_tries=20; \
		try=0; \
		while ! is_port_free "$$port"; do \
			echo "Port $$port is busy, trying next..."; \
			port=$$((port+1)); \
			try=$$((try+1)); \
			if [ "$$try" -ge "$$max_tries" ]; then \
				echo "No free port found in range starting at $(STREAMLIT_PORT)."; \
				exit 1; \
			fi; \
		done; \
			echo "[1/2] Starting Streamlit on port $$port..."; \
			$(LANGSMITH_ENV_PREFIX) $(PYTHON) -m streamlit run $(STREAMLIT_APP) --server.address $(STREAMLIT_HOST) --server.port "$$port" >"$$log_file" 2>&1 & \
			st_pid=$$!; \
		trap "echo \"Stopping Streamlit (pid=$$st_pid)\"; kill $$st_pid 2>/dev/null || true" INT TERM EXIT; \
		sleep 2; \
		if ! kill -0 "$$st_pid" 2>/dev/null; then \
			echo "Streamlit failed to start (pid $$st_pid exited)."; \
			echo "--- Streamlit log tail ($$log_file) ---"; \
			tail -n 80 "$$log_file" || true; \
			exit 1; \
		fi; \
		echo "[2/2] Running MCP client (spawns MCP stdio server)..."; \
		$(LANGSMITH_ENV_PREFIX) $(PYTHON) $(MCP_CLIENT_APP) || true; \
		echo "Streamlit is running at http://$(STREAMLIT_HOST):$$port (Ctrl+C to stop)"; \
		echo "Streamlit log: $$log_file"; \
		wait "$$st_pid"'

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

integration_tests:
	$(PYTHON) -m unittest discover -s tests/integration_tests -p "test_*.py"

test_watch:
	@echo "test_watch 已停用（pytest 已移除）。请手动重复执行 make test。"

test_profile:
	@echo "test_profile 已停用（pytest 已移除）。"

extended_tests:
	$(PYTHON) -m unittest discover -s tests/unit_tests -p "test_*.py"

# ---------------------------------------------------------------------------
# Environment bootstrap
# If a `.env` file exists in the project root it will be included so common
# variables (DB_*, OPENAI_*, STREAMLIT_PORT, etc.) are available to make.
# `.env` should contain simple KEY=VALUE lines (comments with # allowed).
# Variables may also be exported in the shell; Makefile bindings below use
# sensible defaults when values are not provided.
# ---------------------------------------------------------------------------

ifneq (,$(wildcard .env))
   include .env
endif

# Export all make variables to subprocess environment (python, streamlit, etc.)
# so values loaded from `.env` are visible at runtime.
.EXPORT_ALL_VARIABLES:

# Bind Make variables to environment (.env or exported env) with sensible defaults
PYTHON ?= $(shell echo $${PYTHON:-python})
STREAMLIT_APP ?= $(shell echo $${STREAMLIT_APP:-src/mcp/app.py})
STREAMLIT_HOST ?= $(shell echo $${STREAMLIT_HOST:-0.0.0.0})
STREAMLIT_PORT ?= $(shell echo $${STREAMLIT_PORT:-8501})

# LangSmith tracing switch (offline-safe)
# - DISABLE_LANGSMITH=true: force-disable tracing/upload for make targets
# - DISABLE_LANGSMITH=false: keep existing LangSmith behavior from env/config
DISABLE_LANGSMITH ?= $(shell echo $${DISABLE_LANGSMITH:-false})
ifeq ($(strip $(DISABLE_LANGSMITH)),true)
LANGSMITH_ENV_PREFIX = LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false LANGSMITH_API_KEY= LANGCHAIN_LANGSMITH_API_KEY= LANGCHAIN_API_KEY=
else
LANGSMITH_ENV_PREFIX =
endif

# CLI demo runner
CLI_APP ?= $(shell echo $${CLI_APP:-src/mcp/cli_agent.py})
USER_ID ?= $(shell echo $${USER_ID:-interview_user})
CLI_ARGS ?= $(shell echo $${CLI_ARGS:-})

# Database / persistence hints (may be empty if DISABLE_DB=true)
DB_USER ?= $(shell echo $${DB_USER:-})
DB_PASSWORD ?= $(shell echo $${DB_PASSWORD:-})
DB_HOST ?= $(shell echo $${DB_HOST:-127.0.0.1})
DB_PORT ?= $(shell echo $${DB_PORT:-3306})
DB_NAME ?= $(shell echo $${DB_NAME:-})

# Chroma / memory
CHROMA_PERSIST_DIR ?= $(shell echo $${CHROMA_PERSIST_DIR:-})
MEMORY_SQLITE_PATH ?= $(shell echo $${MEMORY_SQLITE_PATH:-})

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=src/
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=src
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	$(PYTHON) -m ruff check .
	[ "$(PYTHON_FILES)" = "" ] || $(PYTHON) -m ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || $(PYTHON) -m ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || $(PYTHON) -m mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && $(PYTHON) -m mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	ruff format $(PYTHON_FILES)
	ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	codespell --toml pyproject.toml

spell_fix:
	codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'streamlit                    - run Streamlit MCP frontend (default :8501)'
	@echo 'cli                          - run CLI agent (USER_ID=..., CLI_ARGS=...)'
	@echo 'cli_db                       - run CLI agent with --disable-db (demo-safe)'
	@echo 'mcp_client                   - run MCP client (spawns MCP stdio server)'
	@echo 'app_and_client               - start Streamlit + run MCP client once'
	@echo 'seed_properties              - init/seed property_listings into MySQL (fallback if DB disabled)'
	@echo 'db_properties                - query all rows from MySQL property_listings (for manual verification)'
	@echo 'DISABLE_LANGSMITH=true|false - toggle LangSmith tracing for make run targets'

