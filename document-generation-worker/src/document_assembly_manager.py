import asyncio
from typing import List, Optional

from document_common.components.base.document_store import (
    DocumentStoreProvider,
    DocumentDocumentStore,
)
from document_common.components.base.prompt import PromptProvider
from document_common.components.writers.file_writer import FileWriter
from document_common.components.writers.document_writer import DocumentWriter
from document_common.schemas import (
    Document,
    DocumentStatus,
)
from document_common.schemas.exceptions import FilesNotImportedException
from document_common.schemas.file import FileStatus
from document_common.services import PromptService, TemplateService
from document_common.utils.clean_stores import empty_document_store
from document_common.utils.logger import Logger

from config import config
from src.components.doc_assembler.document_factory import DocumentFactory


class DocAssemblyManager:
    """
    Manages document assembly, template handling, and preprocessing of files for a Haystack-based document system.

    Responsibilities:
        - Load and manage prompts and document templates.
        - Track and manage file importation and statuses.
        - Create and run documents asynchronously using the defined templates.
        - Preprocess files to generate document chunks (summaries, key points, etc.).
        - Manage document stores for documents, sections, files, and chunks.
        - Flush files and document stores when needed.

    Core Concepts:
        - **Templates**: YAML files defining the structure and methods for generating documents.
        - **Prompts**: YAML-based instructions for LLMs, used by DocumentFactory components.
        - **Document Stores**: Haystack stores that hold documents, sections, files, and derived document chunks.
        - **Preprocessing**: Automatic assembly of summaries and key points using `Processor` and `DocumentFactory`.
        - **document Lifecycle**: Tracks status transitions (PROCESSING, FAILED, COMPLETED) and supports asynchronous execution.
    """

    def __init__(
        self,
    ):
        self.logger = Logger().get_logger("documentManager")

        db_config = config.get_current_config().database
        self.store_provider = DocumentStoreProvider(config)

        self.document_ds = self.store_provider.get_store("documents")
        self.section_ds = self.store_provider.get_store("sections")
        self.document_ds = self.store_provider.get_store("documents")

        self.embed_chunks: bool = db_config.embed_chunks

        self.document_writer = DocumentWriter(self.document_ds, self.section_ds)
        self.file_writer = FileWriter()

        # Create services
        self.prompt_service = PromptService()
        self.template_service = TemplateService()

        # Load into memory (PromptProvider) from storage backend
        self._load_all_prompts()

    def _load_all_prompts(self, force_reload: bool = False):
        """Load prompts from storage backend into PromptProvider (memory)"""

        all_prompts = self.prompt_service.list_all_prompts()
        self.logger.info(f"{len(all_prompts)} prompts loaded")

        # Initialize PromptProvider singleton with empty dict if not already initialized
        prompt_provider = PromptProvider()
        if not hasattr(prompt_provider, "prompts"):
            prompt_provider.logger = Logger().get_logger("PromptProvider")
            prompt_provider.prompts = {}

        for prompt_info in all_prompts:
            # Load into PromptProvider (memory) - keep version field intact
            # Use overwrite=True because we're loading fresh from DB during initialization
            prompt_provider.add_prompt_from_str(
                prompt_info["prompt"], overwrite_existing_prompt=True
            )

        self.logger.info(
            f"Loaded {len(all_prompts)} prompts from storage backend into memory"
        )

    # ------------ document assembly ------------ #
    def check_files_validity(self, file_ids: List[str]):
        # 1. Check if files requested for document assembly are already imported
        files_not_imported: List[tuple[str, str]] = []
        valid_files = []
        for fid in file_ids:
            file = self.file_writer.get_file_obj_by_id(fid)
            if not file:
                files_not_imported.append((fid, "File not found."))
            else:
                if file.status != FileStatus.importED:
                    files_not_imported.append((fid, file.status))
                else:
                    valid_files.append(file)

        if files_not_imported:
            details = "\n".join(
                [f"File: {fid} Status: {status}" for fid, status in files_not_imported]
            )
            self.logger.error(
                f"Some of the files selected for document assembly are not imported. {details}"
            )
            raise FilesNotImportedException(details)

        return valid_files

    async def create_new_document(
        self,
        template_id: str,
        file_ids: List[str],
        context: Optional[dict[str, str]] = None,
        flush: bool = False,
        job_id: str | None = None
    ) -> str:
        """
        Start asynchronous assembly of a new document.

        Args:
            template_id (str): ID of the template to use.
            file_ids (List[str]): List of file IDs to include in the document.
            context (Optional[dict]): Optional fixed inputs for the document (e.g., subject, persona).
            flush (bool): If True, will delete source files and associated document chunks after document completion.

        Returns:
            str: Job ID of the document assembly task.

        Raises:
            FilesNotimportedException: If any file in `file_ids` is not imported.
            Exception: Any error during document factory initialization or task creation.

        Notes:
            - document assembly runs asynchronously in the background.
            - The job ID can be used to track status or retrieve completed sections.
        """
        self.logger.info(f"Setting up document assembly with template '{template_id}'")

        files = self.check_files_validity(file_ids)

        self.logger.info("Files available and all imported.")

        # 2. Refresh prompts memory
        self._load_all_prompts()

        # 3. We build the document object
        if job_id:
            document = Document(file_ids=file_ids, job_id=job_id)
        else:
            document = Document(file_ids=file_ids)

        # 4. We prepare the document assembly
        self.document_writer.update_document(document=document)
        try:
            # 4.a We build our document Pipeline following our template
            document_factory = DocumentFactory(
                template=self.template_service.get_template(template_id),
                document=document,
                document_writer=self.document_writer,
                context=context,  # ex: {"subject": "China economic sanctions", "persona": "You are a NATO Analyst"},
                document_store=DocumentDocumentStore(self.document_ds, file_ids=file_ids),
            )

            # 4.b We check if files have the correct importation (the right sub chunks have been created.)
            validated, message = document_factory.validate_document_availability(files)
            if not validated:
                raise FilesNotImportedException(message)

            # We start the assembly in the background
            document_task = asyncio.create_task(document_factory.run_async())
            if flush:
                asyncio.create_task(self.flush_files(file_ids, document_task))

            # And immediatly return the job_id
            self.logger.info(
                f"Starting assembly of document {document.job_id} with template {template_id}"
            )
            return document.job_id
        except Exception as e:
            document.status = DocumentStatus.FAILED
            self.document_writer.update_document(document=document)
            self.logger.error(f"Starting document job failed : {e}", exc_info=True)
            raise

    async def flush_files(self, file_ids: List[str], document_task: asyncio.Task):
        """Flush specific files and their derived document chunks after a document task finishes."""
        await document_task

        self.file_writer.delete_files(file_ids=file_ids)

        document_store = DocumentDocumentStore(self.document_ds, file_ids)
        empty_document_store(document_store)

    def get_document(self, job_id: str) -> Optional[Document]:
        return self.document_writer.get_document(job_id=job_id)

    def get_document_section(self, job_id: str, section_id: str):
        return self.document_writer.get_document_section(job_id=job_id, section_id=section_id)
