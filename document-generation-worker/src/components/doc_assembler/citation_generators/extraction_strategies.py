"""
Extraction strategies for section citation extraction.

Provides abstract base class and concrete implementations for different
citation extraction approaches (single vs multi-paragraph sections).
"""

import time
from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, List, Optional

from haystack.document_stores.types import DocumentStore
from document_common.schemas import Section
from document_common.schemas.citations import SourceMetadata

from config import config
from src.pipeline.citation import CitationPipeline


class TimeoutException(Exception):
    """Raised when operation exceeds timeout."""

    pass


class ExtractionStrategy(ABC):
    """Abstract base class for citation extraction strategies."""

    def __init__(
        self,
        citation_pipeline: CitationPipeline,
        document_store: DocumentStore,
        logger: Logger,
    ):
        """
        Initialize the extraction strategy.

        Args:
            citation_pipeline: Citation extraction pipeline
            document_store: Document store for retrieving sources
            logger: Logger instance
        """
        self.citation_pipeline = citation_pipeline
        self.document_store = document_store
        self.logger = logger

    def _get_config_value(self, key: str, default: Any) -> Any:
        """
        Get configuration value with fallback to default.

        Args:
            key: Configuration key path (e.g., 'extraction.max_paragraph_length')
            default: Default value if not configured

        Returns:
            Configuration value or default
        """
        try:
            citation_config = config.get_current_config().citations
            keys = key.split(".")
            value = citation_config
            for k in keys:
                value = value.get(k) if hasattr(value, "get") else getattr(value, k, None)
                if value is None:
                    return default
            return value if value is not None else default
        except Exception:
            return default

    @staticmethod
    def _validate_pipeline_result(result: dict, request_id: str) -> bool:
        """
        Validate pipeline result structure.

        Args:
            result: Result dict from pipeline.run()
            request_id: Request ID for logging

        Returns:
            True if valid, raises ValueError if invalid
        """
        if not isinstance(result, dict):
            raise ValueError(f"Pipeline result must be dict, got {type(result).__name__}")

        if "result" not in result:
            raise ValueError(f"Pipeline result missing 'result' key. Keys: {list(result.keys())}")

        citation_result = result.get("result")
        if citation_result is None:
            raise ValueError("Pipeline result['result'] is None")

        required_fields = ["citation_stacks", "source_references", "metrics"]
        for field in required_fields:
            if not hasattr(citation_result, field):
                raise ValueError(f"CitationPipelineResult missing field '{field}'. Available: {dir(citation_result)}")

        return True

    @staticmethod
    def _validate_paragraph_length(paragraph_text: str, max_length: int) -> bool:
        """
        Check if paragraph exceeds max length limit.

        Args:
            paragraph_text: Paragraph text to check
            max_length: Maximum allowed length

        Returns:
            True if within limits, False if exceeds
        """
        return len(paragraph_text) <= max_length

    @staticmethod
    def _validate_section_content(section: Section, request_id: str) -> tuple[bool, str]:
        """
        Validate section content for citation extraction with specific failure reasons.

        Args:
            section: Section to validate
            request_id: Request ID for logging

        Returns:
            Tuple of (is_valid, reason) where reason describes skip reason if not valid
        """
        if not section.generated_content:
            return False, "content_empty"

        content = section.generated_content.strip()
        if not content:
            return False, "content_whitespace_only"

        if len(content) < 50:
            return False, "content_too_short"

        if not any(c.isalnum() for c in content):
            return False, "content_no_alphanumeric"

        if not section.source_ids or all(not ids for ids in section.source_ids):
            return False, "no_source_ids"

        return True, "valid"

    @abstractmethod
    def extract(
        self,
        section: Section,
        request_id: Optional[str] = None,
    ) -> Section:
        """
        Extract citations for a section using this strategy.

        Args:
            section: Section to extract citations for
            request_id: Optional request ID for tracking

        Returns:
            Updated section with citations
        """
        pass

    def _retrieve_source_documents(self, source_ids: List[str], request_id: str) -> List[SourceMetadata]:
        """
        Retrieve source documents from document store.

        Args:
            source_ids: List of document IDs
            request_id: Request ID for logging

        Returns:
            List of SourceMetadata objects
        """
        try:
            # Retrieve documents using filter
            filters = [{"field": "id", "operator": "==", "value": doc_id} for doc_id in source_ids]
            documents = self.document_store.filter_documents(filters={"operator": "OR", "conditions": filters})

            self.logger.debug(f"[{request_id}] Retrieved {len(documents)} documents (requested: {len(source_ids)})")

            # Convert to SourceMetadata
            source_documents: List[SourceMetadata] = []
            for doc in documents:
                self.logger.debug(f"Processing document {doc.id} with type {doc.meta.get('type', 'unknown')}")
                source_metadata = SourceMetadata(
                    chunk_id=doc.id,
                    file_id=doc.meta.get("core_data_id", doc.id),
                    content=doc.content or "",
                    type=doc.meta.get("type", "original"),
                    embedding=doc.embedding,
                )
                source_documents.append(source_metadata)

            if len(source_documents) < len(source_ids):
                # Find which specific IDs are missing
                retrieved_ids = {doc.id for doc in source_documents}
                missing_ids = [sid for sid in source_ids if sid not in retrieved_ids]
                missing_count = len(missing_ids)
                missing_ratio = missing_count / len(source_ids)

                # Log specific missing IDs (first 5)
                self.logger.warning(
                    f"[{request_id}] {missing_count} source documents not found: " +
                    f"{missing_ids[:5]}" +
                    f"{'...' if len(missing_ids) > 5 else ''}"
                )

                # Get threshold from config
                max_ratio = self._get_config_value("extraction.max_missing_source_ratio", 0.5)

                # If too many sources missing, log error
                if missing_ratio > max_ratio:
                    self.logger.error(
                        f"[{request_id}] Over {max_ratio * 100:.0f}% of sources missing ({missing_ratio * 100:.1f}%). " +
                        "Citation extraction may be unreliable."
                    )

            return source_documents

        except Exception as e:
            self.logger.error(f"[{request_id}] Failed to retrieve sources: {e}")
            return []


class SingleParagraphStrategy(ExtractionStrategy):
    """Extraction strategy for single-paragraph sections."""

    def extract(
        self,
        section: Section,
        request_id: Optional[str] = None,
    ) -> Section:
        """
        Extract citations for a single-paragraph section.

        Args:
            section: Section to extract citations for
            request_id: Optional request ID for tracking

        Returns:
            Updated section with citations
        """
        request_id = request_id or section.section_id

        # Validate content with specific reason
        is_valid, reason = self._validate_section_content(section, request_id)
        if not is_valid:
            self.logger.warning(f"[{request_id}] Section validation failed: {reason}")
            section.citation_extraction_status = f"skipped_{reason}"
            return section

        try:
            # Flatten source IDs with deduplication tracking
            all_source_ids_flat = section.source_ids
            all_source_ids_deduplicated = list(set(all_source_ids_flat))

            # Log deduplication stats if count changed
            if len(all_source_ids_flat) != len(all_source_ids_deduplicated):
                log_dedup = self._get_config_value("extraction.log_deduplication_stats", True)
                if log_dedup:
                    self.logger.info(
                        f"[{request_id}] Source IDs deduplicated: " +
                        f"{len(all_source_ids_flat)} → {len(all_source_ids_deduplicated)}"
                    )

            # Retrieve sources
            source_documents = self._retrieve_source_documents(all_source_ids_deduplicated, request_id)

            if not source_documents:
                self.logger.warning(f"[{request_id}] No sources retrieved")
                section.citation_extraction_status = "skipped"
                return section

            self.logger.info(f"[{request_id}] Running citation pipeline")

            # Run pipeline
            result = self.citation_pipeline.run(
                generated_text=section.generated_content or "",
                source_documents=source_documents,
                request_id=request_id,
            )

            # Validate pipeline result structure
            self._validate_pipeline_result(result, request_id)

            citation_result = result["result"]
            section.citations = [citation_result.citation_stacks]
            section.source_references = citation_result.source_references
            section.citation_extraction_status = "completed"

            self.logger.info(f"[{request_id}] Extraction complete: {len(section.citations)} stacks found")

            return section

        except Exception as e:
            self.logger.error(f"[{request_id}] Extraction failed: {e}")
            section.citation_extraction_status = "failed"
            return section


class MultiParagraphStrategy(ExtractionStrategy):
    """Extraction strategy for multi-paragraph sections."""

    def extract(
        self,
        section: Section,
        request_id: Optional[str] = None,
    ) -> Section:
        """
        Extract citations for a multi-paragraph section.

        Processes each paragraph independently, then adjusts citation offsets
        to be relative to the full section content.

        Args:
            section: Section to extract citations for
            request_id: Optional request ID for tracking

        Returns:
            Updated section with citations
        """
        request_id = request_id or section.section_id

        # Validate content with specific reason
        is_valid, reason = self._validate_section_content(section, request_id)
        if not is_valid:
            self.logger.warning(f"[{request_id}] Section validation failed: {reason}")
            section.citation_extraction_status = f"skipped_{reason}"
            return section

        if not section.sections or len(section.sections) <= 1:
            self.logger.debug(f"[{request_id}] Using single-paragraph strategy instead")
            strategy = SingleParagraphStrategy(
                self.citation_pipeline,
                self.document_store,
                self.logger,
            )
            return strategy.extract(section, request_id)

        # Validate mismatch between sections and source_ids
        num_paragraphs = len(section.sections)
        num_source_lists = len(section.source_ids)
        if num_paragraphs != num_source_lists:
            self.logger.warning(
                f"[{request_id}] Mismatch: {num_paragraphs} sections " +
                f"but {num_source_lists} source_id lists. " +
                "Later paragraphs may not have citations."
            )

        try:
            all_citation_stacks = []

            self.logger.info(f"[{request_id}] Extracting {num_paragraphs} paragraphs")

            # Get max paragraph length from config
            max_para_length = self._get_config_value("extraction.max_paragraph_length", 10000)

            # Track extraction statistics
            extraction_stats = {
                "attempted": 0,
                "successful": 0,
                "failed": 0,
                "skipped_empty": 0,
                "skipped_too_long": 0,
                "skipped_no_sources": 0,
            }

            for para_idx, sub_section in enumerate(section.sections):
                para_request_id = f"{request_id}-p{para_idx}"
                extraction_stats["attempted"] += 1

                if not sub_section or not sub_section.strip():
                    self.logger.debug(f"[{para_request_id}] Empty paragraph")
                    extraction_stats["skipped_empty"] += 1
                    continue

                # Check paragraph length limit
                if not self._validate_paragraph_length(sub_section, max_para_length):
                    self.logger.warning(
                        f"[{para_request_id}] Paragraph exceeds max length " +
                        f"({len(sub_section)} > {max_para_length}). Skipping extraction."
                    )
                    extraction_stats["skipped_too_long"] += 1
                    continue

                # Bounds check for source_ids
                if para_idx >= len(section.source_ids):
                    self.logger.warning(
                        f"[{para_request_id}] No source_ids for paragraph {para_idx}. " +
                        f"Source IDs only available for first {len(section.source_ids)} paragraphs."
                    )
                    extraction_stats["skipped_no_sources"] += 1
                    continue

                para_source_ids = section.source_ids[para_idx]

                if not para_source_ids:
                    self.logger.debug(f"[{para_request_id}] No sources")
                    extraction_stats["skipped_no_sources"] += 1
                    continue

                try:
                    # Retrieve sources
                    source_documents = self._retrieve_source_documents(para_source_ids, para_request_id)

                    if not source_documents:
                        self.logger.warning(f"[{para_request_id}] No sources retrieved")
                        continue

                    # Run pipeline for this paragraph with timing
                    pipeline_start = time.time()
                    result = self.citation_pipeline.run(
                        generated_text=sub_section,
                        source_documents=source_documents,
                        request_id=para_request_id,
                    )
                    pipeline_duration = time.time() - pipeline_start

                    # Validate pipeline result structure
                    self._validate_pipeline_result(result, para_request_id)

                    citation_result = result["result"]
                    para_citations = citation_result.citation_stacks

                    # Log pipeline metrics
                    citation_ratio = len(para_citations) / len(source_documents) if source_documents else 0
                    self.logger.debug(
                        f"[{para_request_id}] Pipeline metrics: " +
                        f"duration={pipeline_duration:.2f}s, " +
                        f"citations={len(para_citations)}, " +
                        f"sources={len(source_documents)}, " +
                        f"ratio={citation_ratio:.2f}"
                    )

                    sub_section_citations = [sorted(para_citations, key=lambda x: x.span_start)]

                    all_citation_stacks.extend(sub_section_citations)
                    extraction_stats["successful"] += 1

                    # Aggregate source references from all paragraphs
                    if citation_result.source_references:
                        if section.source_references is None:
                            section.source_references = citation_result.source_references
                        else:
                            # Merge source references from this paragraph
                            for source_id, metadata in citation_result.source_references.references.items():
                                section.source_references.add_source(source_id, metadata)

                    self.logger.debug(f"[{para_request_id}] Extracted {len(para_citations)} stacks")

                except Exception as e:
                    extraction_stats["failed"] += 1
                    self.logger.error(f"[{para_request_id}] Extraction failed: {e}", exc_info=True)
                    continue

            # Log extraction statistics
            self.logger.info(
                f"[{request_id}] Paragraph extraction stats: " +
                f"attempted={extraction_stats['attempted']}, " +
                f"successful={extraction_stats['successful']}, " +
                f"failed={extraction_stats['failed']}, " +
                f"skipped_empty={extraction_stats['skipped_empty']}, " +
                f"skipped_too_long={extraction_stats['skipped_too_long']}, " +
                f"skipped_no_sources={extraction_stats['skipped_no_sources']}"
            )

            section.citations = all_citation_stacks

            # Set status based on extraction results
            if extraction_stats["successful"] == 0 and extraction_stats["attempted"] > 0:
                section.citation_extraction_status = "completed_with_errors"
            else:
                section.citation_extraction_status = "completed"

            self.logger.info(
                f"[{request_id}] Multi-paragraph extraction complete: {len(all_citation_stacks)} total stacks"
            )

            return section

        except Exception as e:
            self.logger.error(f"[{request_id}] Multi-paragraph extraction failed: {e}")
            section.citation_extraction_status = "failed"
            return section


class ExtractionStrategyFactory:
    """Factory for creating extraction strategies."""

    @staticmethod
    def create_strategy(
        section: Section,
        citation_pipeline: CitationPipeline,
        document_store: DocumentStore,
        logger: Logger,
    ) -> ExtractionStrategy:
        """
        Create appropriate extraction strategy based on section structure.

        Args:
            section: Section to extract from
            citation_pipeline: Citation pipeline
            document_store: Document store
            logger: Logger instance

        Returns:
            Appropriate ExtractionStrategy instance
        """
        # Multi-paragraph if sections field is populated with multiple items
        if section.sections and len(section.sections) > 1:
            logger.debug("Using MultiParagraphStrategy")
            return MultiParagraphStrategy(citation_pipeline, document_store, logger)

        logger.debug("Using SingleParagraphStrategy")
        return SingleParagraphStrategy(citation_pipeline, document_store, logger)
