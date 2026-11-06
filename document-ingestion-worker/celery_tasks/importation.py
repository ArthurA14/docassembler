# celery_tasks/importation.py

import asyncio
from typing import Dict, List

from document_common.schemas.api import ImportDocRef
from document_common.schemas.exceptions import RequestTimeoutException
from document_common.utils.logger import Logger

from celery_tasks.celery_app import celery_app
from src.importation_manager import ImportationManager

logger = Logger().get_logger(__name__)


@celery_app.task(
    name="import_documents",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def import_documents_task(
    self,
    docs: List[Dict],
) -> Dict:
    """
    Celery wrapper task around importationManager.process_files()., for document importation.

    Receives a list of document dicts (serializable importDocRef) directly from the API.
    Reconstructs importDocRef objects and passes them to importationManager.process_files().
    """
    logger.info("Starting importation task")
    logger.info(f"Task id: {self.request.id}")
    logger.info(f"Document count: {len(docs)}")

    importation_manager = ImportationManager()
    imported_docs = [ImportDocRef(**doc) for doc in docs]

    try:
        asyncio.run(importation_manager.process_files(files=imported_docs))

        return {
            "status": "completed",
            "coreDataIds": [doc.coreDataId for doc in imported_docs],
        }

    # 408
    except RequestTimeoutException :
        # Re-raise the task so Celery can mark it as failure and the API can display 408-like semantics
        raise
