from typing import Dict, List, Optional

from haystack import Document
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from document_common.components.base.document_store import DocumentStoreProvider
from document_common.schemas.citations import CitationStack
from document_common.utils.logger import Logger

from document_common.schemas.document import Document as AssembledDocument, DocumentStatus, Section, SectionStatus


class DocumentWriter:
    """
    Handles reading and writing documents and their sections to document stores.

    Responsibilities:
        - Retrieve documents and document sections by job ID and section ID.
        - Update documents and sections with new content or status.
        - Supports tracking document and section assembly status.
    """

    def __init__(
        self,
        document_document_store: Optional[DocumentStore] = None,
        section_document_store: Optional[DocumentStore] = None
    ):
        self.logger = Logger().get_logger("document_writer")
        self.document_document_store = document_document_store if document_document_store else DocumentStoreProvider().get_store("documents")
        self.section_document_store = section_document_store if document_document_store else DocumentStoreProvider().get_store("sections")

    # These functions are for the api
    def get_document_status(self, job_id: str) -> Optional[DocumentStatus]:
        document = self.get_document(job_id=job_id)
        if document:
            return document.status
        else:
            self.logger.info(f"The document with job_id {job_id} does not exist yet.")

    def get_document_section_status(
        self, job_id: str, section_id: str
    ) -> Optional[SectionStatus]:
        section = self.get_document_section(job_id=job_id, section_id=section_id)
        if section:
            return section.status
        else:
            self.logger.info(
                f"The section with job_id {job_id} and section_id {section_id} does not exist yet."
            )

    def get_document(self, job_id: str) -> Optional[AssembledDocument]:
        """Returns a document matched to the job_id (also containes the document status). Return empty document if status != Done"""
        document_doc = self.document_document_store.filter_documents(
            {"field": "id", "operator": "==", "value": job_id}
        )
        self.logger.info(self.document_document_store.to_dict())
        self.logger.info(document_doc)
        if len(document_doc) > 0:
            document_obj = AssembledDocument.model_validate(document_doc[0].meta)
            document_obj.sections = self.get_document_sections(job_id=job_id)
            return document_obj
        else:
            self.logger.info(f"No document was found with id {job_id}")
            return None

    def get_document_sections(self, job_id: str) -> Optional[List[Section]]:
        """Retrieve document sections for a given job ID."""

        # Try standard field first
        section_docs = self._fetch_sections_by_field(job_id, "meta.job_id")

        # Fallback to keyword field if needed
        if not section_docs:
            self.logger.info(
                f"No sections found for job_id {job_id} using standard field, attempting keyword field"
            )
            section_docs = self._fetch_sections_by_field(job_id, "meta.job_id.keyword")

        # Early return if no documents found
        if not section_docs:
            self.logger.warning(
                f"No sections found for job_id {job_id}. Verify database connection and data availability."
            )
            return None

        # Convert documents to Section models
        return [Section.model_validate(doc.meta) for doc in section_docs]

    def _fetch_sections_by_field(self, job_id: str, field_name: str) -> List[Document]:
        """Helper method to fetch sections by a specific field."""
        documents = self.section_document_store.filter_documents(
            {
                "field": field_name,
                "operator": "==",
                "value": job_id,
            }
        )

        if documents:
            self.logger.debug(
                f"Found {len(documents)} section(s) for job_id {job_id} using field '{field_name}'"
            )

        return documents

    def get_document_section(self, job_id: str, section_id: str) -> Optional[Section]:
        section_doc = self.section_document_store.filter_documents(
            {
                "field": "meta.section_id",
                "operator": "==",
                "value": section_id,
            }
        )

        if len(section_doc) > 0:
            return Section.model_validate(section_doc[0].meta)
        else:
            self.logger.info(
                f"No document was found with job_id {job_id} and id {section_id}"
            )
            return None

    # Citation retrieval methods
    def get_section_citations(self, section_id: str) -> Optional[List[CitationStack]]:
        """
        Retrieve citations for a specific section.

        Args:
            section_id: The unique identifier for the section

        Returns:
            List of CitationStack objects if citations are available and were extracted,
            None if section not found or citations not extracted
        """

        section_doc = self.section_document_store.filter_documents(
            {
                "field": "id",
                "operator": "==",
                "value": section_id,
            }
        )

        if len(section_doc) == 0:
            self.logger.info(f"No section found with id {section_id}")
            return None

        section = Section.model_validate(section_doc[0].meta)

        if section.citation_extraction_status != "completed":
            self.logger.info(
                f"Citations not available for section {section_id}. Status: {section.citation_extraction_status}"
            )
            return None

        return section.citations

    def get_document_citations(self, job_id: str) -> Dict[str, List[CitationStack]]:
        """
        Retrieve all citations for all sections in a document.

        Args:
            job_id: The job identifier for the document

        Returns:
            Dictionary mapping section_id to list of CitationStack objects.
            Only includes sections with successfully extracted citations.
        """

        sections = self.get_document_sections(job_id=job_id)
        if not sections:
            self.logger.info(f"No sections found for job_id {job_id}")
            return {}

        citations_by_section = {}
        for section in sections:
            if section.citation_extraction_status == "completed" and section.citations:
                citations_by_section[section.section_id] = section.citations

        return citations_by_section

    def get_citation_extraction_status(self, job_id: str) -> Dict[str, str]:
        """
        Get citation extraction status for all sections in a document.

        Args:
            job_id: The job identifier for the document

        Returns:
            Dictionary mapping section_id to citation extraction status
            (pending, completed, failed, skipped, or None if not set)
        """
        sections = self.get_document_sections(job_id=job_id)
        if not sections:
            self.logger.info(f"No sections found for job_id {job_id}")
            return {}

        return {
            section.section_id: section.citation_extraction_status or "unknown"
            for section in sections
        }

    # These functions are for the document assembly tool
    def update_document(self, document: AssembledDocument) -> bool:
        try:
            self.logger.debug(document)
            self.document_document_store.write_documents(
                [
                    Document(
                        id=document.job_id,
                        content="",
                        meta=document.model_dump(mode="json"),
                    )
                ],
                DuplicatePolicy.OVERWRITE,
            )
        except Exception as e:
            raise Exception(
                f"Exception while writing document {document} to the document store : {e}"
            )
        else:
            self.logger.info(f"document {document.job_id} updated, status : {document.status}")
            return True

    def update_document_section(self, section: Section) -> bool:
        """This will update the section in the db with the new status, content, everything will be written."""
        if not section.section_id:
            raise ValueError(f"Can't write section {section}, section_id must be set.")

        try:
            self.section_document_store.write_documents(
                [
                    Document(
                        id=section.section_id,
                        content=section.generated_content,
                        meta=section.model_dump(mode="json"),
                    )
                ],
                DuplicatePolicy.OVERWRITE,
            )
        except Exception as e:
            raise Exception(
                f"Exception while writing section {section} to the document store : {e}"
            )
        else:
            return True
