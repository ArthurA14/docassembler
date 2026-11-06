"""
Citation Pipeline - Main orchestrator for citation extraction and validation.

This module implements the CitationPipeline, which sequences all citation components
(TextPreprocessor, Reranker, Classifier, Aggregator, Validator) to extract, classify,
aggregate, and validate citations linking generated document text to source documents.

The pipeline supports:
- Comprehensive error handling with fallback strategies
- Structured logging with request ID tracking
- Performance metrics collection
"""

import uuid
from typing import Dict, List, Optional

from haystack import component
from document_common.schemas.citations import (
    Citation,
    CitationPipelineResult,
    SourceMetadata,
    SourceReferences,
)
from document_common.utils.logger import Logger

from config import config
from src.components.base.text_reranker import TextPairReranker
from src.components.citations.aggregator import CitationAggregator
from src.components.citations.classifier.classification import CitationClassifier
from src.components.citations.preprocessor import TextPreprocessor
from src.components.citations.validator import CitationValidator
from src.pipeline.citation.builder import PipelineFactory
from src.pipeline.citation.orchestrator import PipelineOrchestrator
from src.pipeline.citation.stages import PipelineStageInput


@component
class CitationPipeline:
    """
    Main orchestrator for citation extraction pipeline.

    Sequences components in order:
    1. TextPreprocessor - Segment text into sentences
    2. TextPairReranker - Find relevant source chunks
    3. CitationClassifier - Classify citation relationships
    4. CitationAggregator - Stack citations by span
    5. CitationValidator - Validate citation integrity

    The pipeline handles:
    - Error handling at each stage with fallback strategies
    - Structured logging with request ID tracking
    - Performance metrics collection
    - Data flow between stages
    """

    def __init__(
        self,
        fail_on_validation_error: bool = False,
    ):
        """
        Initialize the citation pipeline.

        Args:
            fail_on_validation_error: Raise exception on validation errors vs. returning document
        """
        self.logger = Logger().get_logger("citation_pipeline")
        citation_config = config.get_current_config().citations

        # Extract configuration for each component
        preprocessing_config = citation_config.preprocessing
        classification_config = citation_config.classification
        stacking_config = citation_config.stacking
        validation_config = citation_config.validation
        source_filtering_config = citation_config.get("source_filtering", {})

        # Initialize all components with configuration
        preprocessor = TextPreprocessor(min_sentence_length=preprocessing_config.get("min_sentence_length", 20))

        reranker = TextPairReranker()

        classifier = CitationClassifier(
            prompt_name=classification_config.get("prompt_name", "citation_classification"),
            max_retries=classification_config.get("max_retries", 3),
            confidence_threshold=classification_config.get("confidence_threshold", 0.70),
        )

        aggregator = CitationAggregator(
            max_stack_size=stacking_config.get("max_stack_size", 5),
            min_stack_confidence=stacking_config.get("min_stack_confidence", 0.80),
            boost_factor=stacking_config.get("boost_factor", 0.05),
            boost_method=stacking_config.get("boost_method", "standard"),
            ranking_strategy=stacking_config.get("ranking_strategy", "combined"),
            deduplication_method=stacking_config.get("deduplication_method", "fuzzy"),
            fuzzy_threshold=stacking_config.get("fuzzy_threshold", 85.0),
        )

        validator = CitationValidator(
            validate_quote_match=validation_config.get("require_exact_quotes", True),
            validate_type_consistency=True,
            validate_confidence=True,
            allow_whitespace_normalize=validation_config.get("allow_whitespace_normalize", True),
            max_offset_drift=validation_config.get("max_offset_drift", 10),
            fail_on_error=fail_on_validation_error,
        )

        builder = PipelineFactory.create_pipeline(
            preprocessor=preprocessor,
            classifier=classifier,
            aggregator=aggregator,
            validator=validator,
            reranker=reranker,
            source_filtering_config=source_filtering_config,
        )

        self.stages = builder.get_stages()
        self.orchestrator = PipelineOrchestrator()

    def _build_source_references(
        self, citations: List[Citation], source_documents: List[SourceMetadata]
    ) -> SourceReferences:
        """Build SourceReferences from citations and available source documents.

        Creates a SourceReferences object containing all unique source documents
        that are referenced by the citations. This enables API consumers to look up
        source documents by source_id without making separate requests.

        Args:
            citations: List of Citation objects (may be empty)
            source_documents: All available source documents from database

        Returns:
            SourceReferences containing only the sources referenced in citations
        """
        # Collect all unique source_ids from citations
        cited_source_ids = set()
        for citation in citations:
            cited_source_ids.add(citation.chunk_id)

        # Build references dict with only cited sources
        references_dict: Dict[str, SourceMetadata] = {}
        source_by_id = {doc.chunk_id: doc for doc in source_documents}

        for source_id in cited_source_ids:
            if source_id in source_by_id:
                references_dict[source_id] = source_by_id[source_id]
            else:
                self.logger.warning(f"Citation references source_id '{source_id}' but source not found in documents")

        return SourceReferences(references=references_dict)

    @component.output_types(result=CitationPipelineResult)
    def run(
        self, generated_text: str, source_documents: List[SourceMetadata], request_id: Optional[str] = None
    ) -> Dict[str, CitationPipelineResult]:
        """
        Execute the complete citation pipeline.

        Args:
            generated_text: Generated document text to find citations for
            source_documents: List of source documents with content
            request_id: Optional request identifier for tracking (auto-generated if None)

        Returns:
            Dict with 'result' key containing CitationPipelineResult

        Raises:
            ValueError: If fail_on_validation_error=True and validation fails
        """
        request_id = request_id or str(uuid.uuid4())

        self.logger.info(f"[{request_id}] Starting citation pipeline (text_len: {len(generated_text)})")

        # Create pipeline input
        inputs = PipelineStageInput(
            request_id=request_id,
            generated_text=generated_text,
            source_documents=source_documents,
        )

        # Execute via orchestrator
        return self.orchestrator.execute(self.stages, inputs)
