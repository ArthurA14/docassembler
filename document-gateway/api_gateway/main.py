import logging
import os
from contextlib import asynccontextmanager
from typing import Dict

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_gateway.git_version import current_git_version

# Import route modules
from api_gateway.routes.importation import router as import_router
from api_gateway.routes.prompts import router as prompts_router
from api_gateway.routes.docassembler import router as documents_router
from api_gateway.routes.system import router as system_router
from api_gateway.routes.templates import router as templates_router

# from document_common.utils.logger import Logger


# Logging level setup
def configure_logging():
    # Logger().set_level(level=os.getenv("LOG_LEVEL", "INFO"))
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(level=log_level)
    for name in [
        "elastic_transport.transport",
        "httpx",
        "haystack.components.builders.chat_prompt_builder",
    ]:
        logging.getLogger(name).setLevel(logging.CRITICAL)


configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # documentManager is not needed for document-gateway (sends tasks to Redis)
    # The actual processing happens in document-importation-worker
    os.makedirs("api_gateway/doc", exist_ok=True)
    try:
        with open("api_gateway/doc/openapi.yaml", "w") as f:
            yaml.safe_dump(app.openapi(), f, sort_keys=False)
    except PermissionError:
        pass  # Skip writing openapi spec if no write permissions
    yield


app = FastAPI(
    title="document Generator",
    version=current_git_version(),
    description="""
## Overview
Asynchronous document importation and intelligent document assembly service.

## Typical Workflow
1. import documents
2. Poll importation status
3. Generate a document
4. Poll document assembly status

## Key Features
- Document importation with status tracking
- Template-based document assembly
- Citation system with source traceability
- RESTful, job-oriented API
""",
    lifespan=lifespan,
)

# IMPORTANT: Update your CORS middleware to allow OPTIONS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://document.<CLUSTER_IP>.nip.io",  # frontend dns
    ],
    allow_credentials=True,
    allow_methods=["*"],  # This allows all methods INCLUDING OPTIONS
    allow_headers=["*"],  # This allows all headers
)

# @app.get("/")
# async def healthcheck() -> Dict[str, str] :
#     """
#     Root endpoint of the FastAPI application.
#     Returns:
#         - Dict[str, str]: A JSON response containing a welcome message.
#     """
#     return {
#         "service": "importation",
#         "status": "ok",
#         "version": "1.0.0",
#     }

@app.get("/health")
async def health() -> Dict[str, str]:
    """
    Health check endpoint for Docker health checks.
    """
    return {"status": "healthy"}


# Include all routers
app.include_router(import_router)
app.include_router(documents_router)
app.include_router(prompts_router)
app.include_router(system_router)
app.include_router(templates_router)
