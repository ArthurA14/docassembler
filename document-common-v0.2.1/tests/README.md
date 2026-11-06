# document-common Tests

## Overview

This directory contains the test suite for the `document-common` package. Tests are organized to mirror the source code structure and use **pytest** as the testing framework.

## Running Tests

### Run all tests

```bash
cd document-common
uv run pytest
```

### Run with verbose output

```bash
uv run pytest -v
```

### Run with verbose output and show local variables on failure

```bash
uv run pytest -vv --showlocals
```


## Test Structure

```
tests/
├── conftest.py                  # Shared fixtures (mocks, config, etc.)
├── __init__.py
└── src/
    ├── components/
    │   └── base/
    │       ├── test_batched_operations.py    # Batch processing utilities
    │       ├── test_document_store.py        # Document store abstraction
    │       ├── test_embedder.py              # Text embedding components
    │       ├── test_assembly.py            # LLM response assembly
    │       ├── test_popcorn_ner.py           # ⚠️ DISABLED - component not finished
    │       ├── test_prompt.py                # Prompt template management
    │       ├── test_reranker_component.py    # Document reranking
    │       ├── test_retriever.py             # Document retrieval logic
    │       └── test_writer.py                # Document writing utilities
    ├── schemas/
    │   ├── test_chunk_entities_templates.py  # Chunk, Entity, document templates
    │   ├── test_citations.py                 # Citation models and pipelines
    │   ├── test_exceptions.py                # Custom exceptions
    │   ├── test_file.py                      # File model and enums
    │   └── test_document_and_api.py            # document and API models
    ├── services/
    │   ├── test_prompt_service.py            # Prompt CRUD operations
    │   └── test_template_service.py          # Template CRUD operations
    ├── storage/
    │   └── test_storage_base.py              # Storage layer
    └── utils/
        ├── test_reranker_utils.py            # Reranker utility functions
        └── config/
            ├── test_utils_config.py          # Config utilities
            └── test_utils_config_utils.py    # Config manager utilities
```

## What's Covered

### Components (`src/components/base/`)

| Module | Coverage |
|--------|----------|
| **Batched Operations** | Init with defaults/custom params, batch call with sync/async functions, kwargs handling, error handling, method batching, result validation |
| **Document Store** | Universal embedding retriever, documentDocumentStore filter/write/delete/count operations, DocumentStoreProvider factory with caching |
| **Embedder** | ApiEmbedder (single/multiple texts, dimension handling), TEIChonkieEmbeddings, EmbedderFactory, DocumentEmbedder, TextEmbedder |
| **assembly** | ResponseGenerator (structured/unstructured, log probs), AsyncResponseGenerator, Structuredassembly, Textassembly |
| **Popcorn NER** | ⚠️ **DISABLED** — Tests are commented out. The PopcornNer component is not yet finished. Tests will be re-enabled when the component is production-ready. |
| **Prompt** | PromptDict, PromptTemplate validation, PromptProvider singleton, YAML loading, prompt management with versions and overwrites |
| **Reranker** | RerankerAPIClient (config loading, explicit overrides, partial overrides, empty texts, success, timeout, truncation), DocumentReranker (enabled/disabled, min score, fallback, custom top_k) |
| **Retriever** | DocIdsRetriever, Retriever (init, rerank decision logic), DuplicateChecker, ParentFinder, DocumentVersionFilter |
| **Writer** | Document writing with default/custom policies, batched writes, empty document handling |

### Schemas (`src/schemas/`)

| Module | Coverage |
|--------|----------|
| **Chunk & Entities** | Chunk (minimal/full/type variants), Entity, NumLines, SectionContent (snake_case validation), SectionBlock, documentTemplate, PreprocessingTemplate |
| **Citations** | CitationLabel enum, Citation (span validation, confidence bounds, serialization), SupportingCitation, CitationStack (limit to 5, sorting), SourceMetadata, SourceReference, SourceReferences, ClassificationResult, PipelineMetrics, CitationPipelineResult |
| **Exceptions** | FilesNotimportedException, RequestTimeoutException (default/custom messages, UUID format) |
| **File** | FileType (mime detection, extension parsing, all enum values), FileStatus, File (minimal/full/model dump/validate, optional fields) |
| **document & API** | Section (minimal/full/UUID/status/citations), document (minimal/full/status), DocRef, importDocRef (plainText validation) |

### Services (`src/services/`)

| Module | Coverage |
|--------|----------|
| **PromptService** | Init with storage, upsert (with/without version, mode detection), get (existing/not found/missing fields), list, delete, exists, YAML initialization (create/skip/update, subfolders, error handling) |
| **TemplateService** | Init, upsert (raw and YAML with validation), get, delete, list, exists, YAML initialization (create/skip/update/error handling) |

### Storage & Utils

| Module | Coverage |
|--------|----------|
| **Storage** | Storage base class operations |
| **Utils** | Reranker utilities, config loading and manager utilities |

## Fixtures (conftest.py)

Shared fixtures available in all tests:

| Fixture | Description |
|---------|-------------|
| `_clear_config_cache` | Clears config singleton before each test (autouse) |
| `_clear_prompt_provider` | Clears PromptProvider singleton before each test (autouse) |
| `mock_logger` | Returns a MagicMock logger |
| `mock_openai_client` | Mocks OpenAI client for sync assembly tests |
| `mock_async_openai_client` | Mocks AsyncOpenAI client for async assembly tests |
| `mock_config` | Comprehensive mock config with llm, embedder, chunking, reranker, entity_extraction, popcorn_ner settings |
| `haystack_document` | Creates a simple Haystack Document |
| `mock_document_store` | Mocks Haystack DocumentStore |
| `mock_storage` | Mocks PromptStorage |
| `mock_template_storage` | Mocks TemplateStorage |

## Disabled Tests

The following test file is currently **disabled** (all tests commented out):

- `src/components/base/test_popcorn_ner.py` — The PopcornNer component is not yet finished. Tests will be re-enabled when the component reaches production readiness.