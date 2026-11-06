# document assembly System

A production-grade, asynchronous document processing and document assembly pipeline built on FastAPI and Haystack. The system enables automated creation of structured documents using customizable templates with advanced citation management and source traceability.

## Architecture Overview

The document assembly System follows a modular, microservices-oriented architecture designed for high-throughput, asynchronous document processing. The architecture implements a layered approach where FastAPI-based API layer orchestrates document processing workflows through a central document Manager that coordinates multiple specialized subsystems.

```
API Layer (FastAPI)
    ↓
document Manager (Orchestration)
    ↓
Core Components
├── Document importation Pipeline
├── Template Management System
├── Prompt Management System
├── Citation Processing Pipeline
└── Async Job Management
```

### Key Components

**API Layer**: RESTful API providing endpoints for document importation, document assembly, template management, and system operations.

**document Manager**: Central orchestrator managing the complete document assembly pipeline, coordinating between all system components.

**Citation System**: Advanced multi-stage pipeline for citation extraction, ranking, validation, and confidence scoring with complete source traceability.

**Template Engine**: YAML-based template system supporting section-based document assembly with flexible prompting strategies.

## Technology Stack

- **API Framework**: FastAPI 0.115.11+ for high-performance async web services
- **Core Framework**: Haystack AI 2.11.2+ for document processing and vector operations
- **Document Processing**: Tika integration, Python-docx, multi-format support
- **Vector Storage**: Qdrant, Elasticsearch, or InMemory backends
- **LLM Integration**: OpenAI API, vLLM, Ollama compatibility
- **Async Processing**: Background job management with status tracking
- **Configuration**: YAML-based configuration system

## Quick Deployment

### Prerequisites

- Python 3.10-3.12
- Database backend (Qdrant recommended for production)
- LLM API endpoint (vLLM, OpenAI, or compatible)
- Network connectivity to DMZ environment

### Basic Setup

```bash
# Install dependencies
uv sync

# Configure database
# Edit config/configs/db/db_setup.yaml
database: qdrant
url: "http://your-qdrant-host:6333"

# Configure LLM
# Edit config/configs/llm/llm_config.yaml
api_url: "http://your-llm-endpoint/v1"
model: "your-model-name"

# Start API service
uvicorn api.main:app --host 0.0.0.0 --port 5000 --workers 4
```

### Production Deployment

```bash
# Using gunicorn for production
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000

# With load balancer configuration
nginx -c /etc/nginx/document-assembly.conf
```

## API Reference

The system exposes a RESTful API with the following core endpoints designed for asynchronous operation handling:

### Document importation

**POST /import**
- Initiates asynchronous document importation
- Accepts document references with metadata
- Returns 202 Accepted with importation job initiation
- Processing occurs in background workers

**GET /import/{coreDataId}/status**
- Polls importation status for specific documents
- Returns status: pending, processing, imported, failed
- Enables real-time monitoring of processing progress

### document assembly

**POST /generate-document**
- Initiates document assembly workflow
- Requires template ID, document references, optional context
- Returns 202 Accepted with job tracking ID
- Validates dependencies before job creation

**GET /documents/{jobId}**
- Retrieves complete document with sections and citations
- Includes per-section citation mapping and source references
- Provides real-time assembly status and progress
- Progressive delivery as sections complete

### Template Management

**GET /templates**: Lists available document templates
**PUT /templates/{template_name}**: Creates/updates templates
**DELETE /templates/{template_name}**: Removes templates

### System Operations

**GET /files**: Lists available imported files (experimental)
**POST /flush**: Clears all system data (irreversible)

## Configuration

System configuration is managed through YAML files in `config/configs/`:

### Database Configuration (db_setup.yaml)

```yaml
database: qdrant  # qdrant, elasticsearch, inmemory
url: "http://qdrant-host:6333"
indexes:
  sections: { name: "sections" }
  documents: { name: "documents" }
  files: { name: "files" }
  documents: { name: "documents" }
embed_chunks: true
```

### LLM Configuration (llm_config.yaml)

```yaml
api_url: "http://llm-endpoint/v1"
model: "model-name"
temperature: 0.70
max_output_tokens: 8192
timeout: 10000.0
top_p: 0.8
```

## Remote API Compatibility

The document assembly System is compatible with **any** OpenAI API endpoint provider, whether local or remote, provided adequate resources are available. This flexibility enables deployment across diverse infrastructure environments and LLM providers.

### Supported Endpoint Providers

**Cloud Providers:**
- OpenAI API (api.openai.com)
- Azure OpenAI Service
- AWS Bedrock (OpenAI-compatible endpoints)
- Google Vertex AI (with OpenAI compatibility layer)

**Self-Hosted Solutions:**
- vLLM server deployments
- Ollama local instances
- Any other custom OpenAI-compatible API servers

**Enterprise Solutions:**
- Private cloud deployments
- On-premises LLM infrastructure
- Hybrid cloud-edge architectures
- Specialized LLM providers (Anthropic, Cohere, etc.)

### Configuration Examples

**OpenAI Cloud API:**
```yaml
api_url: "https://api.openai.com/v1"
model: "gpt-4"
api_key: "sk-your-openai-api-key"
temperature: 0.70
max_output_tokens: 8192
timeout: 10000.0
```

**vLLM Self-Hosted:**
```yaml
api_url: "http://vllm-server:8000/v1"
model: "meta-llama/Llama-2-70b-chat-hf"
temperature: 0.70
max_output_tokens: 8192
timeout: 30000.0
```

**Ollama Local Instance:**
```yaml
api_url: "http://localhost:11434/v1"
model: "llama2"
temperature: 0.70
max_output_tokens: 8192
timeout: 60000.0
```

### Resource Considerations

**Network Connectivity:** Ensure stable network connection to remote endpoints, configure appropriate timeouts for long-running operations, and consider bandwidth requirements for large model interactions.

**Scalability Planning:** Remote API rate limits may constrain system throughput while local deployments offer unlimited usage but require infrastructure scaling.

## Core Workflows

### 1. Document importation Pipeline

1. Submit document references via POST /import
2. System validates and queues for processing
3. Background workers process documents asynchronously
4. Status polling available via GET endpoints
5. Documents indexed and made available for document assembly

### 2. document assembly Pipeline

1. Select appropriate template from available templates
2. Submit assembly request with document references
3. System validates document availability and generates job ID
4. Async processing of document sections with citation extraction
5. Real-time status updates and progress tracking
6. Final document delivery with complete source traceability

### 3. Template Management

1. Create YAML-based templates defining document structure
2. Configure prompting strategies per section
3. Set retrieval parameters and assembly methods
4. Templates validated and stored in database
5. Version control and management through API

## Integration Patterns

### Microservice Integration

The system is designed for integration into existing microservice architectures:

```python
# Example integration
import requests

# import documents
response = requests.post("http://doc-assembler-api/import", json={
    "docs": [{"file_id": "doc-123", "type": "document", "source": "file-manager"}]
})

# Generate document
job_response = requests.post("http://doc-assembler-api/generate-document", json={
    "templateId": "analysis-template",
    "docs": [{"file_id": "doc-123", "type": "document", "source": "file-manager"}],
    "context": {"subject": "Q4 Analysis", "persona": "analyst"}
})

# Poll for completion
document = requests.get(f"http://doc-assembler-api/documents/{job_response.json()['jobId']}")
```

### Async Job Management

For high-throughput scenarios:

```python
import asyncio
import aiohttp

async def generate_multiple_documents(templates, documents):
    async with aiohttp.ClientSession() as session:
        jobs = []
        for template_id in templates:
            payload = {
                "templateId": template_id,
                "docs": documents,
                "context": {"batch": "large_assembly"}
            }
            response = await session.post("/generate-document", json=payload)
            jobs.append(response.json()["jobId"])
        
        # Monitor all jobs
        while jobs:
            completed = []
            for job_id in jobs:
                status = await session.get(f"/documents/{job_id}")
                if status.json()["status"] == "completed":
                    completed.append(job_id)
            jobs = [j for j in jobs if j not in completed]
            await asyncio.sleep(1)  # Poll interval
```

## Performance Considerations

### Async Processing

The system implements asynchronous processing for:
- Document importation (non-blocking operations)
- document assembly (background job processing)
- Citation extraction and validation
- Template processing and management

### Scalability

Horizontal scaling supported through:
- Multiple worker processes (configurable via --workers flag)
- Database connection pooling
- Async I/O operations throughout the pipeline
- Stateless API design enabling load balancer distribution

### Resource Management

- Configurable worker counts for parallel request handling
- Database connection management and pooling
- Memory-efficient document processing
- Background job cleanup and resource reclamation

## Error Handling

### Standard HTTP Status Codes

- 200: Successful operation
- 202: Accepted (async operations initiated)
- 400: Bad Request (validation errors)
- 404: Resource not found
- 408: Request timeout (long-running operations)
- 409: Conflict (dependencies not ready)
- 500: Internal server error

### Error Response Format

```json
{
    "error": "Error description",
    "error_id": "unique-error-id",  # For timeout tracking
    "error_message": "Detailed technical message"
}
```

## Logging and Monitoring

The system provides structured logging with configurable levels:
- DEBUG: Detailed development information
- INFO: General operational messages
- WARNING: Non-critical issues
- ERROR: Error conditions requiring attention
- CRITICAL: System failures

### Request Tracing

All API requests and responses are logged with:
- Timestamps and correlation IDs
- Request payload sanitization
- Response status and timing information
- Error stack traces for debugging

## Security Considerations

### Network Security

- CORS configuration for allowed origins
- Network isolation through DMZ deployment
- Secure database connectivity patterns
- API endpoint access control integration

### Data Protection

- Input validation through Pydantic models
- UUID validation for resource identifiers
- Secure credential management for external services
- Audit logging for sensitive operations

## Documentation and Support

### Support Channels

- Development team: Contact for network access and infrastructure support
- DMZ environment assistance available
- Configuration troubleshooting and optimization guidance

## Version Information

Current Version: 3.1.0
Target Release: Production v3.1.0
Last Updated: 2026-04-29

For production deployment questions or infrastructure support, contact the development team for DMZ network configuration and optimization guidance.
