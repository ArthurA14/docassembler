import hashlib
from pathlib import Path
from typing import Dict, List

import magic
import requests
from haystack import Document, component
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from document_common.components.writers.file_writer import FileWriter
from document_common.schemas.api import ImportDocRef
from document_common.schemas.file import File, FileStatus
from document_common.utils.logger import Logger

from config import config


@component
class FileHandler:
    """
    Handles file uploads, validation, and storage in the Document Store.

    This component processes uploaded files from the API, performs validation checks,
    and prepares file metadata for storage and downstream processing.

    Key Features:
    - File extension validation against allowed extensions list
    - File size validation against maximum allowed size
    - MIME type detection and content validation using python-magic
    - Duplicate detection based on file content hash (SHA256)
    - File metadata management with status tracking (PENDING, FAILED)
    - Integration with FileWriter for Document Store operations
    - Automatic error handling and logging for processing issues

    The component converts raw file data into Haystack Document objects with
    associated File metadata, making them available for downstream processing
    pipelines.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize the FileHandler component.

        Args:
            file_document_store: the document store for File objects
        """
        self.file_writer = FileWriter()

        self.logger = Logger().get_logger(name="document.file_handler")

        # Default configuration
        self.config = config.get_current_config()["file_handler"]
        self.logger.info(f"Current config : {self.config}")

        # Update config with any provided kwargs
        self.config.update(kwargs)

        # Initialize magic for file type detection
        self.mime_magic = magic.Magic(mime=True)
        self.logger.info(f"Initialized FileHandler with config: {self.config}")

    def is_allowed_file(self, filename: str) -> bool:
        """Check if the file extension is allowed"""
        ext = Path(filename).suffix.lower()
        return ext in self.config["allowed_extensions"]

    def is_allowed_size(self, file_size: int) -> bool:
        """Check if the file size is within allowed limits"""
        return file_size <= self.config["max_file_size"]

    def generate_id(self, file_content: str) -> str:
        """Generate a unique ID based on file content"""
        return hashlib.sha256(file_content.encode("utf-8")).hexdigest()

    def validate_file_content(self, file_content: bytes, claimed_extension: str) -> bool:
        """Validate that the file content matches the claimed file type"""
        detected_type = self.mime_magic.from_buffer(file_content)

        # Map extensions to expected MIME types
        extension_mime_map = {
            ".txt": ["text/plain"],
            ".pdf": ["application/pdf"],
            ".jpg": ["image/jpeg"],
            ".jpeg": ["image/jpeg"],
            ".png": ["image/png"],
            ".md": ["text/plain", "text/markdown"],
            ".docx": [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ],
            ".xlsx": [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ],
        }

        expected_types = extension_mime_map.get(claimed_extension.lower(), [])
        if not expected_types:
            return True

        return detected_type in expected_types

    def parse_file(
        self, path: Path, url: str
    ):  # config.get_current_config().api.parser_url
        """Parses a file given a path and the url of the api of a parser. Currently unused."""
        input_data = {
            "files": open(path, "rb"),  # Path to your DOCX file
        }
        response = requests.post(url, files=input_data)
        response.raise_for_status()
        content = response.json()["document"]["md_content"]

        return content

    @component.output_types(file_documents=List[Document])
    def run(self, files: List[ImportDocRef]) -> Dict[str, List[Document]]:
        """
        Process uploaded files and store them.

        Args:
            files: List of files from FastAPI's UploadFile

        Returns:
            Dictionary containing a list of processed files
        """

        file_docs = []
        for file in files:
            self.logger.info(f"Processing file: '{file.coreDataId}'")
            self.logger.debug(f"File content : {file.model_dump_json(indent=2)}")

            # if not self.is_allowed_file(file.coreType):
            #     pass

            # if not file.plainText:
            #     pass

            if not file.plainText:  # parser can be used.
                file_obj = File(
                    name="",
                    path="",
                    status=FileStatus.FAILED,
                    type="",
                    size=0,
                    message="The path to parse the files ourselves is not implemented yet so you need to give a filled plainText.",
                    content_id="",
                )
            else:
                file_size = len(file.plainText)
                content_id = self.generate_id(file.plainText)

                if not self.is_allowed_size(file_size):
                    file_obj = File(
                        name="",
                        path="",
                        status=FileStatus.FAILED,
                        type="",
                        size=file_size,
                        message="File size exceeds maximum allowed size",
                        content_id=content_id,
                        core_data_id=file.coreDataId,
                        core_type=file.coreType,
                        core_data_source=file.coreDataSource,
                    )
                else:  # "success"
                    file_obj = File(
                        name="",
                        path="",
                        status=FileStatus.PENDING,
                        type=file.coreType,
                        size=file_size,
                        message="File stored in database",
                        content_id=content_id,
                        core_data_id=file.coreDataId,
                        core_type=file.coreType,
                        core_data_source=file.coreDataSource,
                    )
                    is_dup, dup_id = self.file_writer.check_duplicate(
                        content_id=content_id, id=file.coreDataId
                    )
                    if is_dup:
                        message = f"File {file.coreDataId} stored in database but another file already has the same content : {dup_id}"
                        file_obj.message = message
                        self.logger.warning(message)
                    self.logger.info(f"Writing file : '{file.coreType}'")

            doc = Document(
                id=file.coreDataId,
                content=file.plainText,
                meta=file_obj.model_dump(mode="json"),
            )

            try:
                self.file_writer.write_documents(doc, DuplicatePolicy.FAIL)

            except DuplicateDocumentError:
                self.logger.warning(
                    f"A file with id : '{file.coreDataId}' is already in the database, we skipped its insertion. We will now check if retrying its importation is necessary."
                )
                # We retrieve the already saved one, with updated metadatas.
                doc = self.file_writer.get_file_by_id(doc.id)
                file_docs.append(doc)

            except Exception as e:
                self.logger.error(f"Error writing file '{file.coreDataId}': {str(e)}")
                file_obj.status = FileStatus.FAILED
                file_obj.message = f"Error writing: {str(e)}"

            else:
                self.logger.info(f"File written in the database: '{file.coreDataId}'")

                if file_obj.status != FileStatus.FAILED:
                    file_docs.append(doc)
                else:
                    self.logger.warning(
                        f"File validation failed for {file.coreDataId} : {file_obj.message}."
                    )

        return {"file_documents": file_docs}
