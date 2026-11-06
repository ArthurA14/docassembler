"""
Typed result classes for citation pipeline components.

Replaces untyped Dict[str, Any] results with structured Pydantic models,
providing type safety, automatic validation, and better IDE support.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from document_common.schemas.citations import CitationLabel, CitationStack


class ComponentMetadata(BaseModel):
    """Metadata about component execution."""

    component_name: str = Field(..., description="Name of the component")
    duration_seconds: float = Field(0.0, ge=0, description="Execution time in seconds")
    timestamp: Optional[str] = Field(None, description="Execution timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class ComponentResult(BaseModel):
    """Base class for all component results."""

    success: bool = Field(..., description="Whether component executed successfully")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    metadata: ComponentMetadata = Field(..., description="Execution metadata")

    def is_successful(self) -> bool:
        """Check if component executed successfully."""
        return self.success and not self.errors

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0


class PreprocessorResult(ComponentResult):
    """Result from text preprocessing component."""

    class SentenceInfo(BaseModel):
        """Information about a processed sentence."""

        text: str = Field(..., description="Sentence text")
        normalized_text: str = Field(..., description="Normalized sentence text")
        start_index: int = Field(..., ge=0, description="Start character index")
        end_index: int = Field(..., ge=0, description="End character index")

    sentences: List[SentenceInfo] = Field(default_factory=list, description="Processed sentences")

    def get_sentence_count(self) -> int:
        """Get number of processed sentences."""
        return len(self.sentences)


class ClassificationResultItem(BaseModel):
    """Single classification result."""

    classification: CitationLabel = Field(..., description="Citation type classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Explanation of classification")


class ClassifierResult(ComponentResult):
    """Result from citation classifier component."""

    classification: Optional[CitationLabel] = Field(None, description="Classification result")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence")
    reasoning: Optional[str] = Field(None, description="Classification reasoning")
    error: Optional[str] = Field(None, description="Specific classification error if any")

    @property
    def is_independent(self) -> bool:
        """Check if citation is classified as independent."""
        return self.classification and self.classification.value == "independent"


class AggregatorResult(ComponentResult):
    """Result from citation aggregation component."""

    citation_stacks: List[CitationStack] = Field(default_factory=list, description="Aggregated citation stacks")
    stacks_before_filtering: int = Field(0, ge=0, description="Stack count before filtering")
    stacks_after_filtering: int = Field(0, ge=0, description="Stack count after filtering")

    def get_stack_count(self) -> int:
        """Get number of final citation stacks."""
        return len(self.citation_stacks)

    def get_filtered_count(self) -> int:
        """Get number of stacks filtered out."""
        return self.stacks_before_filtering - self.stacks_after_filtering


class ValidationDocument(BaseModel):
    """Detailed validation document."""

    total_checked: int = Field(0, ge=0, description="Total items checked")
    valid_count: int = Field(0, ge=0, description="Valid items")
    invalid_count: int = Field(0, ge=0, description="Invalid items")
    errors: int = Field(0, ge=0, description="Number of errors")
    warnings: int = Field(0, ge=0, description="Number of warnings")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed info")


class ValidatorResult(ComponentResult):
    """Result from citation validation component."""

    valid_citations: List[CitationStack] = Field(default_factory=list, description="Valid citation stacks")
    invalid_citations: List[CitationStack] = Field(default_factory=list, description="Invalid citation stacks")
    validation_document: Optional[ValidationDocument] = Field(None, description="Detailed validation document")

    def get_valid_count(self) -> int:
        """Get number of valid citations."""
        return len(self.valid_citations)

    def get_invalid_count(self) -> int:
        """Get number of invalid citations."""
        return len(self.invalid_citations)

    def get_total_count(self) -> int:
        """Get total number of citations checked."""
        return self.get_valid_count() + self.get_invalid_count()


class RankerResult(ComponentResult):
    """Result from citation ranker component."""

    ranked_citations: List[CitationStack] = Field(default_factory=list, description="Ranked citation stacks")
    ranking_strategy: str = Field(..., description="Strategy used for ranking")
    reordered: bool = Field(False, description="Whether citations were reordered")


class DeduplicatorResult(ComponentResult):
    """Result from citation deduplicator component."""

    deduplicated_citations: List[CitationStack] = Field(default_factory=list, description="Deduplicated citations")
    duplicates_removed: int = Field(0, ge=0, description="Number of duplicates")
    deduplication_method: str = Field(..., description="Method used")


def create_metadata(
    component_name: str,
    duration_seconds: float = 0.0,
    request_id: Optional[str] = None,
) -> ComponentMetadata:
    """
    Create component metadata.

    Args:
        component_name: Name of the component
        duration_seconds: Execution time
        request_id: Request ID for tracking

    Returns:
        ComponentMetadata instance
    """
    return ComponentMetadata(
        component_name=component_name,
        duration_seconds=duration_seconds,
        request_id=request_id,
    )
