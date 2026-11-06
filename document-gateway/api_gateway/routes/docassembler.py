# api_gatway/routes/docassembler.py

import logging

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Response, status
from document_common.components.writers.document_writer import DocumentWriter

from api_gateway.models import (
    GenerateDocumentRequest,
    GenerateDocumentResponse,
    DocumentResponse,
    SectionResponse,
)
from api_gateway.models.common import ErrorResponse
from celery_tasks.celery_app import celery_app

# from document_common.utils.logger import Logger

router = APIRouter(tags=["document"])

# logger = Logger().get_logger("api_gateway.routes.docassembler")
logger = logging.getLogger("api_gateway.routes.docassembler")


# -------------------------------
# Endpoints Implementation
# -------------------------------

# /generate-document


@router.post(
    "/generate-document",
    summary="Creates a document job",
    description="Initiates the document assembly process using a template, document references, and context.",
    response_description="document assembly job started.",
    tags=["document"],
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {
            "description": "document assembly job started.",
            "model": GenerateDocumentResponse,
        },
        204: {
            "description": "Identical task already running.",
        },
        500: {
            "description": "Internal server error during document creation.",
            "model": ErrorResponse,
        },
    },
)
def generate_document(
    payload: GenerateDocumentRequest,
    response: Response
) -> GenerateDocumentResponse:
    """
    Celery producer endpoint.

    Instead of directly awaiting documentManager.create_new_document(...),
    this endpoint enqueues a Celery task and returns immediately with a Celery task id.
    """
    logger.info(f"Generating document with request: {payload.model_dump_json()}")

    try:
        # payload_dict = payload.model_dump()
        # task_hash = generate_task_hash("docassembler", payload_dict)
        # existing_task_id = check_duplicate_task(redis_client, "docassembler", task_hash, payload_dict)

        # if existing_task_id:
        #     ar = AsyncResult(existing_task_id, app=celery_app)
        #     if ar.state in {"PENDING", "STARTED", "RETRY", "SUCCESS"}:
        #         return GeneratedocumentResponse(
        #             jobId=existing_task_id,
        #         )

        task = celery_app.send_task(
            "generate_document",
            args=[
                payload.templateId,
                [doc.coreDataId for doc in payload.docs],
                payload.context,
                payload.flush,
            ],
            queue="docassembler",
        )

        logger.info(f"[generate_document] jobId/task.id = {task.id}")

        # store_task_hash(
        #     redis_client,
        #     "docassembler",
        #     task_hash,
        #     task.id,
        #     payload_dict,
        #     int(os.getenv("REDIS_KEY_TTL_SECONDS", "432000")),
        # )

        # We expose the Celery task id as the external job id
        response.headers["Location"] = f"/documents/{task.id}"

        # 202
        return GenerateDocumentResponse(
            jobId=task.id,  # ~ job_id from DocAssemblyManager().create_new_document
            # taskId=task.id
        )

    # 500
    except Exception as e:
        logger.exception("Unexpected document creation error.")
        raise HTTPException(
            status_code=500, detail=ErrorResponse(error=str(e)).model_dump()
        )


# ------------------------------------

# /documents/{jobId}
# jobId=task.id

@router.get(
    "/documents/{jobId}",
    summary="Get document assembly job status and current result",
    description="Retrieves the status of a specific document assembly job, sections, citations, and source references.",
    response_description="Status of the document assembly job with separated citations and references.",
    tags=["document"],
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Status of the document assembly job with separated citations and references.",
            "model": DocumentResponse,
        },
        400: {
            "description": "Invalid request payload.",
            "model": ErrorResponse
        },
        404: {
            "description": "Job not found.",
            "model": ErrorResponse
        },
        409: {
            "description": "Cannot start document assembly because dependencies are not ready (e.g., files still being imported).",
            "model": ErrorResponse,
        },
        500: {
            "description": "Internal server error during document creation.",
            "model": ErrorResponse,
        },
    },
)
def get_document_task(
    jobId: str,
) -> DocumentResponse:
    """
    This endpoint now treats `jobId` as the Celery task id.

    Behavior:
    - PENDING / STARTED / RETRY
    - FAILURE / REVOKED
    - SUCCESS
    """
    ar = AsyncResult(jobId, app=celery_app)

    # -------------------------
    # Task not finished yet
    # -------------------------
    if ar.info: # The task exists, we check the task state
        if ar.state in {"PENDING"}:
            return DocumentResponse(
                jobId=jobId,
                status="pending",  # ar.state,
                createdAt=None,
                coreDataIds=[],
                sections=[],
                sources={},
            )
        if ar.state in {"STARTED", "RETRY"}:
            return DocumentResponse(
                jobId=jobId,
                status="processing",  # ar.state,
                createdAt=None,
                coreDataIds=[],
                sections=[],
                sources={},
            )

        # -------------------------
        # Task failed
        # -------------------------
        if ar.failed():
            error_text = str(ar.result)
            traceback_text = ar.traceback or ""

            logger.error(f"Celery task failed. state={ar.state}, result={ar.result}")
            logger.error(f"Celery traceback: {ar.traceback}")

            # 400
            if "ValueError" in traceback_text:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ErrorResponse(
                        error=error_text,
                    ).model_dump(),
                )

            # 409
            if "FilesNotImportedException" in traceback_text:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=ErrorResponse(
                        error="document assembly cannot start as all files are not imported.",
                    ).model_dump(),
                )

            # 500
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ErrorResponse(
                    error=error_text,
                ).model_dump(),
            )

    # If the task was not found in the celery, or was successful, we try the persistent back end:
    # -------------------------
    # Task succeeded
    # -------------------------
    document_writer = DocumentWriter()
    document = document_writer.get_document(job_id=jobId)

    if document is None:
        # We couldn't find the document in the celery tasks or in the back end
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error=f"document with id {jobId} not found.",
            ).model_dump()
        )

    # Convert sections to API response format (now includes per-section citations)
    sections_response = [
        SectionResponse.from_section(section)
        for section in (document.sections or [])
    ]

    # Log the whole document:
    logger.info(f"Job {jobId} completed successfully.")
    logger.info("Details of sections:")

    def build_document_text(section: SectionResponse, level: int = 1) -> str:
        document_text = ""
        if section.generated_content:
            document_text += f"{'#' * level} {section.title}\n"
            document_text += f"{section.generated_content}\n"
        if section.sections:
            sub_sections = sorted(section.sections, key=lambda x: x.order)
            for sub_s in sub_sections:
                document_text += build_document_text(section=sub_s, level=level + 1)
        return document_text

    sections = sorted(sections_response, key=lambda x: x.order)
    for section in sections:
        logger.info(build_document_text(section))

    # Build and return the refactored response with new per-section citations and root sources
    document_response = DocumentResponse(
        jobId=document.job_id,
        taskId=jobId,
        status=document.status,
        createdAt=document.created_at,
        coreDataIds=document.file_ids,
        sections=sections_response,
        sources=None,
    )
    
    # After sources are built, set references=None on sections to avoid duplication
    for section in document_response.sections:
        section.references = None
    
    return document_response
