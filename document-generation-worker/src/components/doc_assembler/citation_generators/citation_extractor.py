"""
Citation extraction helper for document sections.

This module provides utilities to extract citations from generated document sections
by linking the generated text back to source documents used during assembly.
"""

from typing import List, Optional

from haystack.document_stores.types import DocumentStore
from document_common.schemas import Section
from document_common.utils.logger import Logger

from src.components.doc_assembler.citation_generators.extraction_strategies import (
    ExtractionStrategyFactory,
)
from src.pipeline.citation import CitationPipeline


class CitationExtractor:
    """
    Helper class to extract citations for document sections.

    This class bridges the gap between document assembly and citation extraction:
    1. Retrieves source documents from document store using section.source_ids
    2. Converts documents to SourceMetadata format expected by CitationPipeline
    3. Runs the citation pipeline to extract citation stacks
    4. Attaches results back to the section

    Uses ExtractionStrategy pattern for modular, testable extraction logic.

    Attributes:
        document_store: Document store containing source documents
        citation_pipeline: Configured CitationPipeline instance
        logger: Logger for tracking extraction process
    """

    def __init__(self, document_store: DocumentStore, fail_on_error: bool = False):
        """
        Initialize the section citation extractor.

        Args:
            document_store: Document store to retrieve source documents from
            fail_on_error: If True, raise exceptions on extraction errors. If False, log and continue.
        """
        self.logger = Logger().get_logger("section_citation_extractor")
        self.document_store = document_store
        self.fail_on_error = fail_on_error

        # Initialize citation pipeline
        self.citation_pipeline = CitationPipeline(
            fail_on_validation_error=False,  # Don't fail on validation errors
        )

        self.logger.info(f"CitationExtractor initialized (fail_on_error: {fail_on_error})")

    def extract_for_section(self, section: Section, request_id: Optional[str] = None) -> Section:
        """
        Extract citations for a single document section with paragraph-aware processing.

        This method intelligently handles two scenarios:
        1. Multi-paragraph sections (sequential): Uses section.sections[] and source_ids[i] for per-paragraph extraction
        2. Single-paragraph sections (direct): Uses flattened source_ids for entire content extraction

        Args:
            section: Section object to extract citations for
            request_id: Optional request ID for tracking (uses section_id if not provided)

        Returns:
            Updated Section object with citations field populated

        Raises:
            ValueError: If fail_on_error=True and extraction fails
        """
        self.logger.debug(f"Section = {section.model_dump_json(indent=2)}")
        request_id = request_id or section.section_id

        # Validate section has content
        if not section.generated_content:
            self.logger.warning(f"[{request_id}] Section has no generated content, skipping citation extraction")
            section.citation_extraction_status = "skipped"
            return section

        # Validate section has source IDs
        self.logger.debug(f"Section source_ids = {section.source_ids}")
        if not section.source_ids or all(not ids for ids in section.source_ids):
            self.logger.warning(f"[{request_id}] Section has no source IDs, skipping citation extraction")
            section.citation_extraction_status = "skipped"
            return section

        try:
            self.logger.info(
                f"[{request_id}] Starting citation extraction for section '{section.title}' " +
                f"(content_length: {len(section.generated_content)}, " +
                f"source_paragraphs: {len(section.source_ids)}, " +
                f"sections_field_populated: {len(section.sections) > 0})"
            )

            # Create appropriate extraction strategy and delegate
            strategy = ExtractionStrategyFactory.create_strategy(
                section=section,
                citation_pipeline=self.citation_pipeline,
                document_store=self.document_store,
                logger=self.logger,
            )

            updated_section = strategy.extract(section, request_id)

            return updated_section

        except Exception as e:
            error_msg = f"Citation extraction failed for section '{section.title}': {str(e)}"
            self.logger.error(f"[{request_id}] {error_msg}")

            section.citation_extraction_status = "failed"

            if self.fail_on_error:
                raise ValueError(error_msg) from e

            return section

    def extract_for_sections(self, sections: List[Section], request_id: Optional[str] = None) -> List[Section]:
        """
        Extract citations for multiple sections in sequence.

        Args:
            sections: List of Section objects to process
            request_id: Optional request ID for tracking

        Returns:
            List of updated Section objects with citations
        """
        updated_sections: List[Section] = []

        for idx, section in enumerate(sections):
            section_request_id = f"{request_id or 'batch'}-section-{idx}"
            updated_section = self.extract_for_section(section, section_request_id)
            updated_sections.append(updated_section)

        # Log summary
        completed = sum(1 for s in updated_sections if s.citation_extraction_status == "completed")
        failed = sum(1 for s in updated_sections if s.citation_extraction_status == "failed")
        skipped = sum(1 for s in updated_sections if s.citation_extraction_status == "skipped")

        self.logger.info(
            f"[{request_id}] Batch citation extraction complete: " +
            f"{completed} completed, {failed} failed, {skipped} skipped " +
            f"(total: {len(sections)})"
        )

        return updated_sections
