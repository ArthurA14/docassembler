from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class CitationLabel(str, Enum):
    """Classification of how generated text relates to source material"""

    DIRECT_QUOTE = "direct_quote"  # Verbatim text from source
    CLOSE_PARAPHRASE = "close_paraphrase"  # Minimal rewording, same structure
    LOOSE_PARAPHRASE = "loose_paraphrase"  # Significant rewording, same meaning
    INDEPENDENT = "independent"  # Original synthesis without direct source


class Citation(BaseModel):
    """Single citation linking generated text span to source material.

    Explicitly separates:
    - Generated text span information (where citation appears in document)
    - Source document information (what it cites from)
    - Source text location (where in source document it comes from)
    """

    # ====== GENERATED TEXT SPAN (where citation appears in document) ======
    span_text: str | None = Field(
        default=None, description="Exact text from generated document that is being cited", exclude=True
    )
    span_start: int | None = Field(
        default=None, ge=0, description="Character offset start in generated text", exclude=True
    )
    span_end: int | None = Field(default=None, description="Character offset end in generated text", exclude=True)

    # ====== SOURCE DOCUMENT IDENTIFICATION ======
    file_id: str = Field(..., description="UUID of source file from database")
    chunk_id: str = Field(..., description="Document ID in storage")

    # ====== CITATION CLASSIFICATION ======
    citation_label: CitationLabel = Field(
        ...,
        description="DIRECT_QUOTE | CLOSE_PARAPHRASE | LOOSE_PARAPHRASE | INDEPENDENT",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for citation class attribution"
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic similarity between generated text and source",
    )

    @model_validator(mode="after")
    def validate_generated_span(self):
        """Validate that generated span text and offsets are consistent."""
        # Skip validation if optional fields are not set
        if self.span_start is None or self.span_end is None or self.span_text is None:
            return self
        
        if self.span_end <= self.span_start:
            raise ValueError(
                f"generated_span_end ({self.span_end}) must be > generated_span_start ({self.span_start})"
            )
        expected_length = self.span_end - self.span_start
        if len(self.span_text) != expected_length:
            raise ValueError(
                f"generated_span_text length ({len(self.span_text)}) must match offset span length ({expected_length})"
            )
        return self


class SupportingCitation(BaseModel):
    """Single citation supporting a generated text span"""

    chunk_id: str = Field(..., description="Document hash ID from database", examples=["a1b2c3d4-e5f6-7890-1234-567890abcdef"])
    file_id: str = Field(..., description="Original file UUID", examples=["a1b2c3d4-e5f6-7890-1234-567890abcdef"])
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for citation class attribution", examples=[0.85]
    )
    citation_label: CitationLabel = Field(..., description="Label for citation")
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic similarity between generated text and source",
        examples=[0.80]
    )


class CitationStack(BaseModel):
    """Multiple citations supporting a single generated text span"""

    span_text: str = Field(
        ..., description="Text from generated document", examples=["Alors que la guerre en Ukraine accentue les réalignements géostratégiques, les transferts d’armes russes vers des alliés comme la Chine..."]
    )
    span_start: int = Field(
        ..., ge=0, description="Character offset in generated text", examples=[0]
    )
    span_end: int = Field(
        ..., description="Character offset end in generated text", examples=[270]
    )
    primary_citation: Citation = Field(..., description="Highest confidence source")
    supporting_citations: List[SupportingCitation] = Field(
        default_factory=list, description="Additional corroborating sources"
    )
    stack_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate confidence boosted by multi-source agreement",
        examples=[0.95],
    )

    @model_validator(mode="after")
    def validate_span_text_consistency(self):
        """Validate that the CitationStack span_text matches the primary citation span_text.

        Skip validation when primary_citation.span_text is empty — this only happens during
        deserialization from the document store, where Citation.span_text was excluded because
        exclude=True is set on the Citation model. In that case, the data was valid when
        originally created; we just can't re-validate it after deserialization.
        """
        # Skip validation during deserialization from document store
        # (primary_citation.span_text is None because Citation.span_text has exclude=True)
        if not self.primary_citation.span_text:
            return self

        if self.primary_citation.span_text != self.span_text:
            raise ValueError(
                f"CitationStack span_text ('{self.span_text}') must match primary_citation.span_text ('{self.primary_citation.span_text}')"
            )
        return self

    @field_validator("supporting_citations")
    def limit_stack_size(cls, v):
        if len(v) > 5:  # Configurable max_stack_size
            return sorted(v, key=lambda x: x.confidence, reverse=True)[:5]
        return v


class SourceMetadata(BaseModel):
    """Metadata for source documents from database"""

    chunk_id: str = Field(..., description="Document hash ID from database")
    file_id: str = Field(..., description="Original file UUID")
    content: str = Field(..., description="Full document text")
    type: Literal["original"] = Field(default="original")
    embedding: Optional[List[float]] = Field(default=None, exclude=True)


class SourceReference(BaseModel):
    """Enhanced source metadata with traceability fields for API responses.

    Extends SourceMetadata with provenance information for complete source attribution
    and audit trails. Used in API responses to provide full context about sources.

    Attributes:
        chunk_id: Hash ID of the source document chunk (for lookup)
        file_id: UUID of the original source file (from metadata propagation)
        content: Full text content of the source
    """

    chunk_id: str = Field(
        ..., description="Hash ID of source document chunk from database", examples=["a1b2c3d4-e5f6-7890-1234-567890abcdef"]
    )
    file_id: str = Field(
        ..., description="UUID of original source file for traceability", examples=["a1b2c3d4-e5f6-7890-1234-567890abcdef"]
    )
    content: str = Field(..., description="Full content of source document")


class ClassificationResult(BaseModel):
    """Structured output for citations"""

    reasoning: str = Field(
        ..., description="Brief explanation of classification decision"
    )
    classification: CitationLabel = Field(..., description="Citation type classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")


class PipelineMetrics(BaseModel):
    """Performance and execution metrics for pipeline run"""

    request_id: str = Field(
        ..., description="Unique identifier for this pipeline execution"
    )
    total_duration_seconds: float = Field(
        ..., ge=0.0, description="Total pipeline execution time"
    )
    component_durations: dict[str, float] = Field(
        default_factory=dict, description="Time spent in each component"
    )
    sentences_processed: int = Field(
        default=0, ge=0, description="Number of sentences from preprocessor"
    )
    citations_created: int = Field(
        default=0, ge=0, description="Number of citations created"
    )
    stacks_created: int = Field(
        default=0, ge=0, description="Number of citation stacks created"
    )
    validation_errors: int = Field(
        default=0, ge=0, description="Number of validation errors"
    )
    validation_warnings: int = Field(
        default=0, ge=0, description="Number of validation warnings"
    )


class SourceReferences(BaseModel):
    """Collection of unique source documents referenced by citations.

    Maps chunk_id to SourceMetadata, enabling lookup of source documents
    without duplicating content across multiple citations.

    Example:
        {
            "chunk-uuid-1": SourceMetadata(chunk_id="...", content="..."),
            "chunk-uuid-2": SourceMetadata(chunk_id="...", content="..."),
        }
    """

    references: Dict[str, SourceMetadata] = Field(
        ...,
        description="Mapping of chunk_id to SourceMetadata for all sources referenced in citations",
    )

    def get_source(self, chunk_id: str) -> Optional[SourceMetadata]:
        """Get a source document by ID.

        Args:
            chunk_id: The source document ID

        Returns:
            SourceMetadata if found, None otherwise
        """
        return self.references.get(chunk_id)

    def has_source(self, chunk_id: str) -> bool:
        """Check if a source document exists in references.

        Args:
            chunk_id: The chunk ID

        Returns:
            True if source exists, False otherwise
        """
        return chunk_id in self.references

    def add_source(self, chunk_id: str, metadata: SourceMetadata) -> None:
        """Add or update a source document in references.

        Args:
            chunk_id: The chunk ID
            metadata: The SourceMetadata to store
        """
        self.references[chunk_id] = metadata


class CitationPipelineResult(BaseModel):
    """Complete result of citation pipeline execution

    Includes citation stacks and source references to enable API consumers
    to access both citations and their source documents without separate requests.
    """

    citation_stacks: List[CitationStack] = Field(
        default_factory=list, description="Valid citation stacks produced"
    )
    invalid_citations: List[CitationStack] = Field(
        default_factory=list, description="Citations that failed validation"
    )
    source_references: SourceReferences = Field(
        ...,
        description="All unique source documents referenced in citations, keyed by source_id",
    )
    metrics: PipelineMetrics = Field(..., description="Pipeline execution metrics")
    validation_document: Optional[dict] = Field(
        default=None, description="Detailed validation document"
    )
    errors: List[str] = Field(
        default_factory=list, description="Error messages from pipeline execution"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warning messages from pipeline execution"
    )
