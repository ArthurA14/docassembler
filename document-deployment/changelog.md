# Changelog

All notable changes to this project will be documented in this file.

## [v3.1.0] - 2026-05-05

### New Features
*   **document-common Library:** Extracted all shared logic (schemas, components, services, storage, utilities) into a unified `document-common` library, eliminating code duplication across `document-gateway`, `document-importation-worker`, and `document-assembly-worker`.

### Refactoring
*   **Citation Extraction:** Streamlined Pydantic models in the citation extraction pipeline to reduce noisy API parameters and improve response payload size (embeddings, duplicates, etc...)

### Improvements
*   **Template & Prompt Validation:** Templates and prompts are now validated before being stored in Qdrant, rejecting malformed entries.
*   **Helm Charts:** Parameterized URLs in `values.yaml` for environment-specific deployments.

### Bug Fixes
*   Various bug fixes across the codebase.

---

## [v3.0.1] - 2026-04-15

### Documentation
*   **Infrastructure:** Added comprehensive Helm/Kubernetes deployment documentation to assist with environment setup and cluster configuration.

---

## [v3.0.0] - 2026-04-14

*Major release introducing a distributed task architecture with Celery and Redis, replacing the previous AsyncIO-based concurrency model.*

### Architecture
*   **Worker Split:** Refactored the codebase into two independent Celery workers — `document-importation-worker` and `document-assembly-worker` — for better separation of concerns and independent scaling.
*   **Celery Integration:** Replaced FastAPI AsyncIO concurrency with a distributed task queue (Celery + Redis), enabling true horizontal scaling and multi-user workload decoupling through workers.
*   **Redis Broker:** Introduced Redis as the Celery message broker and result backend, ensuring reliable task dispatching and result tracking across workers.

> For further details on the Celery + Redis architecture and integration guidelines, refer to the dedicated documentation from the root of this project: `additional-docs/README-celery-vs-async.md`.<br>

Or go to: http://documentation.<CLUSTER_IP>.nip.io/technos/celery-redis/README-celery-vs-async/

---

## [v2.0.1] - 2025-11-19

### Documentation
*   **Infrastructure:** Added comprehensive infrastructure deployment documentation to assist with environment setup.

---

## [v2.0.0] - 2025-11-18
*Major release introducing advanced citation handling, prompt versioning, and a complete infrastructure overhaul.*

### New Features
*   **Citation Engine:**
    *   Implemented a full Citation Extraction pipeline with strategy patterns.
    *   Added semantic source filtering and reranking capabilities.
    *   Introduced `CitationReference`, `documentResponse`, and `SectionResponse` models.
    *   Added global citation indexing to prevent marker overlap across paragraphs.
    *   Enabled "Reasoning Prompts" for better citation context.
*   **Prompt & Template Management:**
    *   **Versioning:** Added support for versioned prompts and templates with storage backend.
    *   **Hot-reloading:** Implemented APIs to upsert/delete templates and force reload without restarting.
    *   **YAML Init:** Support for initializing prompts via YAML configuration.
*   **document assembly:**
    *   **Multiple Sections:** Added support for generating documents with multiple nested sections.
    *   **Summary Extraction:** Added `extract_summary` and `extract_key_points` capabilities to the assembly queries.
    *   **Batching:** Introduced a `BatchProcessor` base class and optimized LLM batching with similarity thresholds.
*   **Data Pipeline:**
    *   **DocumentFactories:** Complete refactor of the preprocessing pipeline components (DocumentFactories) for better maintainability.
    *   **Validators:** Added validation for circular connections in processors.

### Infrastructure & Performance
*   **Vector Database:** **Migrated from ElasticSearch to Qdrant.**
    *   Updated all document stores, embeddings, and retrieval logic to support Qdrant.
*   **Package Management:** **Switched from Poetry to `uv`.**
    *   Significantly faster dependency resolution and project metadata management.
*   **Docker & DevOps:**
    *   Added `98proxy` configuration to Docker builds.
    *   Included static tokenizers directly in the repo to support offline/isolated environments.
    *   Optimized Dockerfile and removed unnecessary build configurations.

### Bug Fixes
*   **Response Models:** Hotfix for response models to correctly include nested section models.
*   **Tokenizer:** Fixed tokenizer paths and optimized loading speeds.
*   **Prompt Logic:** Fixed an issue where the operator "in" would occasionally fail in prompt templates.
*   **Logging:** Fixed document Manager log naming and improved verbosity for reranker debugging.

### Refactoring
*   **Clean Architecture:** Refactored pipeline into Stages and Builder patterns.
*   **Cleanup:** Removed obsolete citation processing modules, unused notebooks, and legacy "MVP mode" code.
*   **Naming:** Standardized nomenclature (renamed `writter` to `writer`, standardized `CitationLabel`).

---

## [v0.1.5] - 2025-09-10
*Previous stable release.*

*   Initial implementation of environment variable substitution in YAML.
*   Offline tokenizer support.
*   Basic Docker Compose configuration.
