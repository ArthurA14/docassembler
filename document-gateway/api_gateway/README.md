# document assembly API

FastAPI-based REST API for asynchronous document importation and intelligent document assembly with citations and reference tracking.

## Overview

This API provides a comprehensive solution for:
- **Document importation**: Process and prepare documents for analysis
- **Template-Based document assembly**: Create structured documents using customizable templates
- **Citation Management**: Citation extraction and ranking with confidence scoring
- **Async Processing**: Background job processing for long-running operations

## Architecture

### Core Components

```
api_gateway/
├── main.py                  # FastAPI application entry point
├── main_popcorn_ner.py
├── dependencies.py
├── git_version.py
├── models/                  # Pydantic models for request/response validation
│   ├── ...
├── routes/                  # API route handlers
│   ├── import.py            # Document importation endpoints
│   ├── prompts.py           # Prompt management endpoints
│   ├── docassembler.py         # document assembly endpoints
│   ├── system.py            # System health & utility endpoints
│   └── templates.py         # Template management endpoints
├── utils/                   # Utility modules
│   ├── ...
├── doc/                     # API documentation
│   ├── openapi.yaml         # OpenAPI latest specification export
│   ├── openapi.legacy.yaml  # Legacy OpenAPI specification
│   └── feedback_api.yaml    # Future feedback API specification
└── test/                    # End to End test suite
    ├── __init__.py
    ├── conftest.py
    ├── test_import.py
    ├── test_document_assembly.py
    ├── test_system_endpoints.py
    ├── test_templates_prompts.py
    ├── mockoon_conf/         # Mockoon server configurations
    └── test_documents/       # Sample test documents
```


## API Endpoints

The API exposes the following route groups:

- **`/import`** — Document importation endpoints. Accepts a list of document references and enqueues a Celery task (`import_documents`) for asynchronous processing. Supports status polling via `/{coreDataId}/status` to track importation progress per document.

- **`/docassembler`** — document assembly endpoints. Initiates asynchronous document assembly via Celery (`generate_document` task) by specifying a template ID, document references, and optional context. The task is processed asynchronously, and the job status/result can be retrieved at `/documents/{jobId}` which returns generated sections with per-section citations and source references.

- **`/templates`** — Template management endpoints. Lists all available document templates and allows creating/upserting new YAML-based templates. Template content is validated against the `documentTemplate` schema, and the template name in the URL must match the `template_name` field inside the YAML payload.

- **`/prompts`** — Prompt management endpoints. Lists, retrieves, and creates prompt configurations used by the assembly engine. Prompts are YAML-based with version tracking and are validated against the `PromptDict` schema. Supports status polling for individual prompt retrieval.

- **`/system`** — System health and utility endpoints. Provides two operations: `GET /files` returns the list of available (imported) files in the database, and `POST /flush` clears all files, preprocessing, and document data from the system.

Detailed endpoint specifications are available in `doc/openapi.yaml`.