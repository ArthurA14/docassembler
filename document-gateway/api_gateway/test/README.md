# End-to-End Test Suite

This directory contains end-to-end tests for the document Gateway API. Tests run against a live API deployment to verify the full system works correctly.

## Tests Overview

| File | Purpose |
|------|---------------|
| `test_import.py` | Submit documents → poll status → verify imported. Tests duplicate detection (204), missing fields, non-existent file status (404), and database verification (requires `--live-backend`). |
| `test_document_assembly.py` | Generate documents with the `t_test` template, validate section structure (Introduction, Contexte, Descriptif, Conclusion) and content. Tests invalid/missing template IDs and non-existent job IDs. |
| `test_templates_prompts.py` | List and create templates and prompts (GET/PUT operations). |
| `test_system_endpoints.py` | Health check (`GET /health`), available files (`GET /files`), and imported file presence in the files list. |
| `conftest.py` | Shared fixtures, CLI options (`--live-backend`, `--poll-timeout`, `--poll-interval`), and helper functions for polling and database verification. |

## How to Run

### Prerequisites

Start the document app and ensure `API_BASE_URL` points to it (default: `http://localhost:5000`).

### Basic (no backend)

```bash
# All tests (skips those needing a live database/LLM)
uv run pytest api_gateway/test/

# Specific test file
uv run pytest api_gateway/test/test_import.py -v
```

### With live backend (recommended)

```bash
# Full suite including database verification
uv run pytest api_gateway/test/ --live-backend

# Custom timeouts for slow environments
uv run pytest api_gateway/test/ --live-backend --poll-timeout=600 --poll-interval=10

# By marker
uv run pytest api_gateway/test/ -m importation          # importation tests only
uv run pytest api_gateway/test/ -m document             # document tests only
uv run pytest api_gateway/test/ -m live_backend       # Database/LLM-dependent tests
```

### Test ordering

Tests automatically run in this order: validation tests → importation tests → document tests. This ensures documents are imported before document assembly runs.

## Test Data

- 5 sample documents about "RedLand War" (`test_documents/redland_war_*.txt`)
- Each test run generates fresh UUIDs for document IDs
- Mockoon configurations are in `mockoon_conf/` for mocking Qdrant and vLLM

