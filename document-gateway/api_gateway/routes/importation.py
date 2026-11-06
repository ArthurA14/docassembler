# api_gateway/routes/import.py

import logging
import os

from celery.result import AsyncResult

# Import models for JSON encoding
from fastapi import APIRouter, HTTPException, status
from fastapi.encoders import jsonable_encoder
from document_common.components.writers.file_writer import FileWriter
from document_common.schemas import FileStatus

from api_gateway.models import (
    ImportAcceptedResponse,
    ImportationStatusResponse,
    ImportRequest,
    RequestTimeoutException,
)
from api_gateway.models.common import (
    ErrorResponse,
    TimeoutErrorResponse,
)
from api_gateway.utils.redis import redis_client
from celery_tasks.celery_app import celery_app

router = APIRouter(prefix="/import", tags=["import"])

# logger = Logger().get_logger("api_gateway.routes.importation")
logger = logging.getLogger("api_gateway.routes.importation")

# -------------------------------
# Endpoints Implementation
# -------------------------------

# /import


@router.post(
    "",
    summary="import documents",
    description="Accepts a list of document references to import and prepare for document assembly.",
    response_description="Documents are being imported.",
    tags=["import"],
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {
            "description": "Documents are being imported.",
            "model": ImportAcceptedResponse,
        },
        204: {
            "description": "Identical task already running.",
        },
        408: {
            "description": "Request timed out.",
            "model": TimeoutErrorResponse
        },
        500: {
            "description": "Internal server error during importation.",
            "model": ErrorResponse,
        },
    },
)
def import_documents(payload: ImportRequest) -> ImportAcceptedResponse:
    try:
        # payload_dict = {"docs": jsonable_encoder(payload.docs)}
        # task_hash = generate_task_hash("importation", payload_dict)
        # existing_task_id = check_duplicate_task(redis_client, "importation", task_hash, payload_dict)

        # if existing_task_id:
        #     ar = AsyncResult(existing_task_id, app=celery_app)
        #     if ar.state in {"PENDING", "STARTED", "RETRY", "SUCCESS"}:
        #         return Response(status_code=status.HTTP_204_NO_CONTENT)

        task = celery_app.send_task(
            "import_documents",
            args=[jsonable_encoder(payload.docs)],
            queue="importation",
        )

        logger.info(f"[import] jobId/task.id = {task.id}")

        # store_task_hash(
        #     redis_client,
        #     "importation",
        #     task_hash,
        #     task.id,
        #     payload_dict,
        #     int(os.getenv("REDIS_KEY_TTL_SECONDS", "432000")),
        # )

        # Store ONLY the mapping
        for doc in payload.docs:
            redis_client.setex(
                f"import_task:{doc.coreDataId}",
                int(os.getenv("REDIS_KEY_TTL_SECONDS", "432000")),
                task.id,
            )

        # 202
        return ImportAcceptedResponse(
            message="importation process has started for the provided documents."
        )

    # 408
    except RequestTimeoutException as e:
        raise HTTPException(
            status_code=408,
            detail=TimeoutErrorResponse(
                errorId=e.error_id, errorMessage=e.message
            ).model_dump(),
        )

    # 500
    except Exception as e:
        logger.exception("Unexpected importation error.")
        raise HTTPException(
            status_code=500, detail=ErrorResponse(error=str(e)).model_dump()
        )


# ------------------------------------

# /tasks/{task_id}/status

# @router.get(
#     "/tasks/{task_id}/status",
#     summary="Returns the Celery importation task status.",
#     description="Returns the current Celery status of an importation task.",
#     response_description="Current file status.",
#     tags=["import"],
#     responses={
#         404: {"description": "File not found."},
#     },

# )
# def get_import_task_status(task_id: str) :
#     ar = AsyncResult(task_id, app=celery_app)

#     return {
#          "task_id": task_id,
#          "state": ar.state,
#          "result": ar.result if ar.successful() else None
#     }


@router.get(
    "/{coreDataId}/status",
    summary="Returns the Celery importation task status.",
    description="Returns the current Celery status of an importation task.",
    response_description="Current file status.",
    tags=["import"],
    responses={
        404: {
            "description": "File not found."
        },
    },
)
def get_import_task_status(coreDataId: str) -> ImportationStatusResponse:
    task_id = redis_client.get(f"import_task:{coreDataId}")
    if not task_id:
        raise HTTPException(status_code=404, detail="File not found.")
    ar = AsyncResult(task_id, app=celery_app)

    if ar.info is not None:
        if ar.state in {"PENDING"}:
            return ImportationStatusResponse(
                coreDataId=coreDataId, status=FileStatus.PENDING
            )
        if ar.state in {"RETRY", "STARTED"}:
            return ImportationStatusResponse(
                coreDataId=coreDataId, status=FileStatus.PROCESSING
            )
        if ar.state in {"FAILURE"}:
            return ImportationStatusResponse(
                coreDataId=coreDataId, status=FileStatus.FAILED
            )
        if ar.state in {"SUCCESS"}:
            return ImportationStatusResponse(
                coreDataId=coreDataId, status=FileStatus.importED
            )
    else:
        file_writer = FileWriter()

        file_status = file_writer.get_file_status(file_id=coreDataId)
        if file_status:
            return ImportationStatusResponse(coreDataId=coreDataId, status=file_status)
        else:
            raise HTTPException(status_code=404, detail="File not found.")
