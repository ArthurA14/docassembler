# celery_tasks/doc_assembler.py

import asyncio
from typing import Dict, List, Optional

from document_common.schemas.exceptions import FilesNotImportedException
from document_common.utils.logger import Logger

from celery_tasks.celery_app import celery_app
from src.document_assembly_manager import (
    DocAssemblyManager,
)

logger = Logger().get_logger(__name__)


@celery_app.task(
    name="generate_document",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def generate_document_task(
    self,
    template_id: str,
    file_ids: List[str],
    context: Optional[Dict[str, str]],
    flush: bool,
) -> Dict:
    """
    Celery task for document assembly.

    Receives typed primitive arguments directly from the API (no Dict wrapper,
    no Pydantic model reconstruction needed in the worker).
    Instantiates documentManager and runs the async business method.
    """
    logger.info("Starting document task")
    logger.info(f"Task id: {self.request.id}")
    logger.info(f"template_id: {template_id}, file_ids count: {len(file_ids)}")

    assembly_manager = DocAssemblyManager()

    try:
        job_id = asyncio.run(
            assembly_manager.create_new_document(
                template_id=template_id,
                file_ids=file_ids,
                context=context,
                flush=flush,
                job_id=self.request.id,
            )
        )

        return {
            "status": "accepted",
            "jobId": job_id,
            "templateId": template_id,
            "coreDataIds": file_ids,
        }

    except (ValueError, FilesNotImportedException):
        raise

