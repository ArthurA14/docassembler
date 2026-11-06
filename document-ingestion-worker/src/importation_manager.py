from pathlib import Path
from typing import List, Optional

import yaml
from haystack.document_stores.types import DocumentStore
from document_common.components.base import PromptProvider
from document_common.components.base.document_store import DocumentStoreProvider
from document_common.schemas import (
    PreprocessingTemplate,
)
from document_common.schemas.api import ImportDocRef
from document_common.services import PromptService
from document_common.utils.logger import Logger

from config import config
from src.components.preprocessing import PreProcessing


class ImportationManager:
    """
    Manages file importation and preprocessing for a Haystack-based document system.

    Responsibilities:
        - Load and manage prompts from storage.
        - Track and manage file importation and statuses.
        - Preprocess files to generate document chunks (summaries, key points, etc.).
        - Manage document stores for files and document chunks.
        - Configure and run the preprocessing pipeline.

    Core Concepts:
        - **Prompts**: YAML-based instructions for LLMs, used by DocumentFactory components.
        - **Document Stores**: Haystack stores that hold files and derived document chunks.
        - **Preprocessing**: Automatic assembly of summaries and key points using `Processor` and `DocumentFactory`.
        - **importation Lifecycle**: Tracks file status transitions (importED, FAILED) during preprocessing.
    """

    def __init__(
        self,
    ):
        self.logger = Logger().get_logger("documentManager")

        self.store_provider = DocumentStoreProvider(config)
        self.file_ds = self.store_provider.get_store("files")
        self.document_ds = self.store_provider.get_store("documents")

        db_config = config.get_current_config().database
        self.embed_chunks: bool = db_config.embed_chunks

        # Create services
        self.prompt_service = PromptService()

        # Load into memory (PromptProvider) from storage backend
        self._load_all_prompts()
        self.preprocessing_pipeline = self._create_preprocessing_pipeline(
            self.document_ds
        )

    def _load_all_prompts(self, force_reload: bool = False):
        """Load prompts from storage backend into PromptProvider (memory)"""

        all_prompts = self.prompt_service.list_all_prompts()
        self.logger.info(
            f"{len(all_prompts)} prompts loaded"
        )
        
        # Initialize PromptProvider singleton with empty dict if not already initialized
        prompt_provider = PromptProvider()
        if not hasattr(prompt_provider, "prompts"):
            prompt_provider.logger = Logger().get_logger("PromptProvider")
            prompt_provider.prompts = {}

        for prompt_info in all_prompts:
            # Load into PromptProvider (memory) - keep version field intact
            # Use overwrite=True because we're loading fresh from DB during initialization
            prompt_provider.add_prompt_from_str(prompt_info["prompt"], overwrite_existing_prompt=True)

        self.logger.info(
            f"Loaded {len(all_prompts)} prompts from storage backend into memory"
        )

    # ------------ Preprocessing ------------ #
    def _create_preprocessing_pipeline(
        self,
        document_ds: DocumentStore,
        preprocessing_template: Optional[PreprocessingTemplate] = None,
    ):
        """
        Internal method to create a preprocessing pipeline for file importation.

        Args:
            file_ds (DocumentStore): Document store holding original files.
            document_ds (DocumentStore): Document store to store generated chunks.

        Returns:
            PreProcessing: Configured preprocessing pipeline including processors for summaries and key points.

        Notes:
            - The pipeline uses DocumentFactory to generate structured chunks from files.
            - Can be extended with additional processors for other document types.
        """
        if not preprocessing_template:
            path = Path(config.get_current_config().others.default_preprocessing)
            if not path.exists():
                raise FileNotFoundError(f"Default config file not found: {path}")

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            preprocessing_template = PreprocessingTemplate.model_validate(data)

        processings: List[str] = []
        connections: List[tuple[str, str]] = []

        # --- First pass: build each processor ---
        for proc_conf in preprocessing_template.preprocessing:
            processings.append(proc_conf.type)

            if proc_conf.using:
                connections.append((proc_conf.using, proc_conf.type))

        return PreProcessing(
            processings=processings,
            connections=connections,
            chunk_document_store=document_ds,
            embed=self.embed_chunks,
        )

    async def process_files(self, files: List[ImportDocRef]):
        """
        Preprocess a list of files to generate document chunks asynchronously.

        Args:
            files (List[importDocRef]): List of file references to preprocess.

        Notes:
            - Generates summaries and key points using the preprocessing pipeline.
            - Runs in background using `asyncio.create_task`.
        """
        # Refresh prompt in memory
        self._load_all_prompts()
        
        self.logger.info(
            f"Starting preprocessing of files : {[file.coreDataId for file in files]}"
        )
        await self.preprocessing_pipeline.run(files=files)