from typing import List

from haystack import Document, Pipeline, component
from haystack.document_stores.types import DocumentStore
from document_common.components.base import Writer
from document_common.components.base.chunker import FileChunker
from document_common.components.base.embedder import DocumentEmbedder
from document_common.components.writers.file_writer import FileWriter
from document_common.schemas.api import ImportDocRef
from document_common.schemas.file import File, FileStatus
from document_common.utils.logger import Logger

from src.components.insert.file_handler import FileHandler
from src.components.preprocessing.processor import ProcessorManager


@component  # type: ignore
class PreProcessing:
    """
    A Haystack component for importing files and generating document chunks.

    This component handles the complete file importation pipeline:
      1. Converts uploaded files into Haystack `Document` objects.
      2. Chunks documents into original chunks.
      3. Optionally embeds the chunks.
      4. Writes chunks to a document store.
      5. Runs additional sub-chunk processing through `ProcessorManager`.

    Returns file metadata with importation status and created document types.

    Haystack Inputs:
        - files: List[importDocRef] -- List of file references to import.

    Haystack Outputs:
        - files: List[File] -- List of files with importation status and doc_types.
    """

    def __init__(
        self,
        chunk_document_store: DocumentStore,
        processings: List[str] | None = None,
        connections: List[tuple[str, str]] = [],
        embed: bool = True,
    ):
        """
        Initialize the pre-processing pipeline.

        Args:
            file_document_store (DocumentStore): Store holding original file-level documents.
            chunk_document_store (DocumentStore): Store where processed chunk-level documents are written.
            processors (dict[str, Processor], optional): Additional processors applied after writing. Defaults to {}.
            connections (List[tuple[str, str]], optional): Custom connections for processor pipeline. Defaults to [].
            embed (bool, optional): Whether to embed chunks before writing them. Defaults to True.
        """
        self.logger = Logger().get_logger(name="document.preprocessing_pipeline")

        self.file_handler = FileHandler()
        self.file_writer = FileWriter()
        self.processings = processings or []

        # First we build the chunking pipeline that will make the "originals"
        self.chunking_pipeling = Pipeline()
        self.chunking_pipeling.add_component("chunker", FileChunker())
        if embed:
            self.chunking_pipeling.add_component("embedder", DocumentEmbedder())
        self.chunking_pipeling.add_component("writer", Writer(document_store=chunk_document_store))

        if embed:
            self.chunking_pipeling.connect("chunker", "embedder")
            self.chunking_pipeling.connect("embedder", "writer")
        else:
            self.chunking_pipeling.connect("chunker", "writer")

        # Then we build the sub chunks
        self.processor_manager = ProcessorManager(
            processings=processings, 
            connections=connections, 
            document_store=chunk_document_store
        ) if self.processings else None

        self.chunk_document_store = chunk_document_store
    
    def _get_or_create_originals(self, doc: Document, file: File) -> List[Document]:
        # First the original chunk must be created or retrieved
        if file.doc_types and "original" in file.doc_types:
            originals = self.chunk_document_store.filter_documents({
                "operator": "AND",
                "conditions": [
                    {"field": "meta.file_id", "operator": "==", "value": file.core_data_id},
                    {"field": "meta.type", "operator": "==", "value": "original"},
                ],
            })
        else:
            try:
                result = self.chunking_pipeling.run({"chunker": {"documents": [doc]}})
                originals = result['writer']['documents']
            except Exception as e:
                self.logger.error(f"Error importing file {doc.id}: {str(e)}", exc_info=True)

                file.status = FileStatus.FAILED
                file.message = str(e)
                self.file_writer.set_file(file=file, content=doc.content)

                raise e
            
        return originals

    @component.output_types(files=List[File])
    async def run(self, files: List[ImportDocRef]) -> dict[str, List[File]]:
        file_documents = self.file_handler.run(
            files
        )["file_documents"]  # We convert our UploadFiles to Haystack Documents with File meta and check their validity (non dup, non empty, etc...)

        files_imported = []
        for doc in file_documents:
            file = File.model_validate(doc.meta)  # We validate that the document is in file format

            doc_types_created = []

            if file.status == FileStatus.importED and file.importation_type == self.processings:
                self.logger.warning(f"File {file.core_data_id} already imported with sub_chunks: {self.processings}")
                files_imported.append(file)
                continue
            
            self.logger.info(f"importing file {file.core_data_id}")

            originals = self._get_or_create_originals(doc, file)
            doc_types_created.append("original")

            # Then the sub chunks:
            if self.processor_manager:
                result = self.processor_manager.run(documents=originals, processings_to_skip=file.importation_type, file_id=file.core_data_id)
                
                doc_types_created.extend(result.get("doc_types"))
                successful_processings = result.get("processings")

                file.importation_type = successful_processings
                file.doc_types=doc_types_created

                if sorted(successful_processings) == sorted(self.processings):
                    message = f"File {file.core_data_id} imported with doc_types {doc_types_created}"
                    self.logger.info(message)

                    file.status=FileStatus.importED
                    file.message = message

                    self.file_writer.set_file(file, doc.content)
                    files_imported.append(file)
                else:
                    message = f"""File {file.core_data_id} partially failed importation. Failed processors : {
                        [p for p in self.processings if p not in successful_processings]
                    }."""

                    file.message = message
                    file.status=FileStatus.FAILED

                    self.file_writer.set_file(file, doc.content)
                    self.logger.error(message)
                    raise RuntimeError(message)
                
        return {"files": files_imported}

