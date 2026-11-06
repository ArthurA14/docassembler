# DocAssembler

DocAssembler is an asynchronous document-processing and assembly system. It imports source documents, indexes their content, and assembles structured documents using configurable templates and language models, with citations linking generated content to its sources.

## Features

- Document import, preprocessing, and indexing.
- Template-based document assembly.
- LLM-powered content generation with source citations.
- Asynchronous processing and task-status tracking.
- API-based management of documents, prompts, and templates.

## Architecture

A FastAPI gateway submits tasks to Redis. Dedicated Celery workers handle document importation and assembly.

| Component | Description |
|---|---|
| [`document-common-v0.2.1`](./document-common-v0.2.1/) | Shared components, schemas, configuration utilities, and storage integrations. |
| [`document-gateway`](./document-gateway/) | HTTP API and asynchronous task submission. |
| [`document-ingestion-worker`](./document-ingestion-worker/) | Source-document importation and preprocessing. |
| [`document-generation-worker`](./document-generation-worker/) | Template-driven document assembly and citation processing. |
| [`document-deployment`](./document-deployment/) | Docker Compose, Kubernetes, and Helm deployment configurations. |

The system integrates with Qdrant or Elasticsearch for document storage and retrieval, alongside configurable LLM, embedding, and parsing services.

## Getting Started

See the component READMEs for development instructions and the
[deployment documentation](./document-deployment/) for environment setup.

Deployment configurations contain example endpoints and placeholders. Configure the required services and credentials before running the system.
