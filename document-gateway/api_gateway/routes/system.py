import logging

from fastapi import APIRouter, HTTPException
from document_common.components.writers.file_writer import FileWriter
from document_common.utils.clean_stores import flush_all
from document_common.utils.logger import Logger
# from rich.markup import escape

from api_gateway.models import AvailableFilesResponse, FlushSuccessResponse
from api_gateway.models.common import ErrorResponse

router = APIRouter(tags=["System"])


logger = Logger().get_logger("api_system")

@router.get(
    "/files",
    summary="Get available files in database",
    description="Returns available and imported files for document assembly. Raises an error if database is unavailable.",
    responses={
        500: {"description": "Unexpected error during files recuperation."}
    },
)
async def get_available_files() -> AvailableFilesResponse:
    """
    Returns a list of the available files in the `files` collection of the selected database.

    Returns:
        - 200: Files available in the database. Can be an empty list.
    """
    logging.info("HTTP POST /files - Request Recieved")
    try:
        file_writer = FileWriter()
        file_list = file_writer.list_imported_files()
        response = AvailableFilesResponse(files=file_list or [])
        return response
    except Exception as e:
        logging.critical(f"HTTP POST /files - Files check operation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during files recuperation: {str(e)}"
        )

@router.post(
    "/flush",
    summary="Flush system data",
    description="Attempts to flush files, preprocessing data and document data. Raises an error if the flush operation fails.",
    responses={
        500: {"model": ErrorResponse, "description": "Flush operation failed."}
    },
)
async def flush_data() -> FlushSuccessResponse:
    """
    Flushes files, preprocessing data and document data.

    Returns:
        - 200: If flush completed successfully.
        - 500: If flush failed due to an unexpected error.
    """
    try:
        # Call your internal flush function
        flush_all()

        return FlushSuccessResponse(message="Data flushed successfully.")

    # except Exception as e:
    #     logger.error(
    #         "Unexpected error during flush: %s",
    #         escape(str(e)),
    #         exc_info=True
    #     )
    except Exception as e:
        print("Unexpected error during flush:", repr(e), flush=True)

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=f"Unexpected error during flush: {str(e)}"
            ).model_dump()
        )
