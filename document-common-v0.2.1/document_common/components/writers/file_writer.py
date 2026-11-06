from typing import List, Optional

from haystack import Document
from haystack.document_stores.errors import MissingDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from document_common.components.base.document_store import DocumentStoreProvider
from document_common.utils.logger import Logger

from document_common.schemas import File, FileStatus


class FileWriter:
    """Manages files (raw documents) in the Document Store for importation tracking.

    This component provides functionality to store, retrieve, update, and delete file metadata
    in the document store. Each file represents a raw document received from the importation API
    and tracks its processing status throughout various importation stages.

    Key Features:
    - Store and retrieve file documents by ID
    - Detect duplicate content across files
    - Track file processing status (PENDING, importED, FAILED, etc.)
    - Record importation type and document types for tracking purposes
    - Support for bulk operations (delete multiple files, list all files)
    - Comprehensive logging of operations and errors
    """

    def __init__(self, document_store: Optional[DocumentStore] = None):
        if not document_store:
            self.document_store = DocumentStoreProvider().get_store('files')
        self.logger = Logger().get_logger(name="document.file_writer")

    def get_file_by_id(self, file_id: str) -> Optional[Document]:
        """Retrieve file and metadata from Document Store by ID"""
        try:
            doc = self.document_store.filter_documents({"field": "id", "operator": "==", "value": file_id})
            if len(doc) == 0:
                self.logger.warning(f"File with ID {file_id} not found")
                return None
            return doc[0]
        except Exception as e:
            self.logger.error(f"Error retrieving file {file_id}: {str(e)}")
            return None
    
    def check_duplicate(self, content_id: str, id: str) -> tuple[bool, Optional[List[str]]]:
        """Check for files with same content but different id"""
        duplicates = self.document_store.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "id", "operator": "!=", "value": id},
                    {"field": "meta.content_id", "operator": "==", "value": content_id},
                ],
            }
        )
        
        if len(duplicates) > 0:
            return True, [dup.id for dup in duplicates]
        else:
            return False, None
    
    def write_documents(self, document: Document, policy: DuplicatePolicy = DuplicatePolicy.NONE):
        self.document_store.write_documents([document], policy)

    def get_file_obj_by_id(self, file_id: str) -> Optional[File]:
        """Retrieve file and metadata from Document Store by ID and converts it to File object."""
        doc = self.get_file_by_id(file_id=file_id)
        if doc:
            file_obj = File.model_validate(doc.meta)
            return file_obj
        return None
    
    def delete_files(self, file_ids: List[str]) -> None:
        """Deletes multile files from Document Store"""
        for file_id in file_ids:
            self.delete_file(file_id=file_id)

    def delete_file(self, file_id: str) -> bool:
        """Delete a file from Document Store"""
        try:
            doc = self.get_file_by_id(file_id)
            if not doc:
                self.logger.warning(f"Cannot delete non-existent file: {file_id}")
                return False

            try :
                self.document_store.delete_documents([file_id])
            except MissingDocumentError as e:
                self.logger.warning(f"While trying to remove file {file_id} from the document store a MissingDocumentError was raised : {e}")
                return False
            else:
                self.logger.info(f"Deleted file metadata from Elasticsearch: {file_id}")
                return True
        except Exception as e:
            self.logger.error(f"Error deleting file {file_id}: {str(e)}")
            return False

    def list_files(self) -> Optional[List[Document]]:
        """List all available files"""

        try:
            docs = self.document_store.filter_documents()
            return docs
        except Exception as e:
            self.logger.error(f"Error listing files: {str(e)}")
            return None
    
    def list_imported_files(self) -> Optional[List[Document]]:
        """List all available AND imported files"""

        try:
            docs = self.document_store.filter_documents({'field': 'meta.status', 'operator': '==', 'value': FileStatus.importED},)
            return docs
        except Exception as e:
            self.logger.error(f"Error listing files: {str(e)}")
            return None

    def set_file(self, file: File, content: str):
        # Simpler set file option
        if file.status == FileStatus.importED and not file.importation_type:
            self.logger.warning(
                f"You are setting the status of file {file.core_data_id} to imported but not giving the importation type, we will consider only originals chunk have been produced."
            )
        if file.status == FileStatus.FAILED and not file.message:
            self.logger.warning(
                f"You are setting the status of file {file.core_data_id} to Failed but not giving an error message."
            )
        
        file_document = Document(content=content, id=file.core_data_id, meta=file.model_dump())
        self.document_store.write_documents([file_document], DuplicatePolicy.OVERWRITE)
        
        return file

    def get_file_status(self, file_id: str) -> Optional[FileStatus]:
        """Returns the file status given a file_id"""
        file = self.get_file_obj_by_id(file_id=file_id)
        if file:
            return file.status
        else:
            return None
