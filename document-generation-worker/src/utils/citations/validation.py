"""
Validation utilities for citations.

Provides validation functions and classes for citations and citation stacks.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import ValidationError
from document_common.schemas.citations import (
    Citation,
    CitationLabel,
    CitationStack,
    SourceMetadata,
)


class ValidationErrorType(str, Enum):
    """Types of validation errors that can occur"""

    SCHEMA_VIOLATION = "schema_violation"
    OFFSET_MISMATCH = "offset_mismatch"
    QUOTE_TEXT_MISMATCH = "quote_text_mismatch"
    INVALID_CONFIDENCE = "invalid_confidence"
    CITATION_TYPE_MISMATCH = "citation_type_mismatch"
    STACK_INTEGRITY = "stack_integrity"
    MISSING_SOURCE = "missing_source"
    INVALID_SPAN = "invalid_span"
    DUPLICATE_SOURCE = "duplicate_source"


class ValidationIssue:
    """Represents a validation issue found during compliance checking"""

    def __init__(
        self,
        error_type: ValidationErrorType,
        message: str,
        citation_index: Optional[int] = None,
        field: Optional[str] = None,
        severity: str = "error",
    ):
        self.error_type = error_type
        self.message = message
        self.citation_index = citation_index
        self.field = field
        self.severity = severity

    def to_dict(self) -> Dict:
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "citation_index": self.citation_index,
            "field": self.field,
            "severity": self.severity,
        }

    def __repr__(self) -> str:
        location = f"[Citation {self.citation_index}]" if self.citation_index is not None else ""
        field_info = f" (field: {self.field})" if self.field else ""
        return f"{self.severity.upper()} {location}: {self.error_type.value} - {self.message}{field_info}"


def validate_citation_type_consistency(citation: Citation) -> Optional[ValidationIssue]:
    """
    Validate that citation type is consistent with similarity score.

    Args:
        citation: Citation to validate

    Returns:
        ValidationIssue if inconsistency found, None otherwise
    """
    if citation.citation_label == CitationLabel.DIRECT_QUOTE:
        if citation.similarity_score < 0.95:
            return ValidationIssue(
                error_type=ValidationErrorType.CITATION_TYPE_MISMATCH,
                message=f"DIRECT_QUOTE should have similarity_score >= 0.95 (got {citation.similarity_score:.2f})",
                field="citation_type",
                severity="warning",
            )

    elif citation.citation_label == CitationLabel.CLOSE_PARAPHRASE:
        if citation.similarity_score < 0.80:
            return ValidationIssue(
                error_type=ValidationErrorType.CITATION_TYPE_MISMATCH,
                message=f"CLOSE_PARAPHRASE should have similarity_score >= 0.80 (got {citation.similarity_score:.2f})",
                field="citation_type",
                severity="warning",
            )

    elif citation.citation_label == CitationLabel.LOOSE_PARAPHRASE:
        if citation.similarity_score < 0.60:
            return ValidationIssue(
                error_type=ValidationErrorType.CITATION_TYPE_MISMATCH,
                message=f"LOOSE_PARAPHRASE should have similarity_score >= 0.60 (got {citation.similarity_score:.2f})",
                field="citation_type",
                severity="warning",
            )

    elif citation.citation_label == CitationLabel.INDEPENDENT:
        if citation.similarity_score > 0.70:
            return ValidationIssue(
                error_type=ValidationErrorType.CITATION_TYPE_MISMATCH,
                message=f"INDEPENDENT should have similarity_score < 0.70 (got {citation.similarity_score:.2f})",
                field="citation_type",
                severity="warning",
            )

    return None


def validate_confidence_score(citation: Citation) -> Optional[ValidationIssue]:
    """
    Validate confidence score is reasonable and consistent.

    Args:
        citation: Citation to validate

    Returns:
        ValidationIssue if invalid, None otherwise
    """
    if citation.confidence < 0.0 or citation.confidence > 1.0:
        return ValidationIssue(
            error_type=ValidationErrorType.INVALID_CONFIDENCE,
            message=f"Confidence must be in range [0.0, 1.0] (got {citation.confidence})",
            field="confidence",
        )

    if citation.confidence > citation.similarity_score + 0.2:
        return ValidationIssue(
            error_type=ValidationErrorType.INVALID_CONFIDENCE,
            message=f"Confidence ({citation.confidence:.2f}) should not exceed similarity_score ({citation.similarity_score:.2f}) by more than 0.2",
            field="confidence",
            severity="warning",
        )

    return None


def validate_citation_stack(
    stack: CitationStack, sources: Optional[Dict[str, SourceMetadata]] = None
) -> List[ValidationIssue]:
    """
    Validate citation stack integrity.

    Args:
        stack: CitationStack to validate
        sources: Optional source metadata for validation

    Returns:
        List of validation issues found
    """
    issues = []

    if stack.primary_citation.confidence < 0.5:
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.STACK_INTEGRITY,
                message=f"Primary citation has low confidence ({stack.primary_citation.confidence:.2f})",
                field="primary_citation",
                severity="warning",
            )
        )

    if len(stack.supporting_citations) > 5:
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.STACK_INTEGRITY,
                message=f"Stack has too many supporting citations ({len(stack.supporting_citations)} > 5)",
                field="supporting_citations",
                severity="warning",
            )
        )

    source_ids = [stack.primary_citation.file_id]
    for supporting in stack.supporting_citations:
        if supporting.file_id in source_ids:
            issues.append(
                ValidationIssue(
                    error_type=ValidationErrorType.DUPLICATE_SOURCE,
                    message=f"Duplicate source_id in stack: {supporting.file_id}",
                    field="supporting_citations",
                    severity="warning",
                )
            )
        source_ids.append(supporting.file_id)

    expected_confidence = stack.primary_citation.confidence
    if stack.supporting_citations:
        boost = min(len(stack.supporting_citations) * 0.05, 0.25)
        expected_confidence = min(1.0, expected_confidence + boost)

    if abs(stack.stack_confidence - expected_confidence) > 0.1:
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.STACK_INTEGRITY,
                message=f"Stack confidence ({stack.stack_confidence:.2f}) does not match expected ({expected_confidence:.2f})",
                field="stack_confidence",
                severity="warning",
            )
        )

    if stack.span_end <= stack.span_start:
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.INVALID_SPAN,
                message=f"Invalid span: end_index ({stack.span_end}) <= start_index ({stack.span_start})",
                field="span_end_index",
            )
        )

    expected_span_length = stack.span_end - stack.span_start
    if len(stack.span_text) != expected_span_length:
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.INVALID_SPAN,
                message=f"Span text length ({len(stack.span_text)}) does not match offset span ({expected_span_length})",
                field="span_text",
            )
        )

    return issues


def validate_citation_schema(citation: Citation) -> Optional[ValidationIssue]:
    """
    Validate citation against Pydantic schema.

    Args:
        citation: Citation to validate

    Returns:
        ValidationIssue if schema validation fails, None otherwise
    """
    try:
        citation.model_validate(citation.model_dump())
        return None
    except ValidationError as e:
        error_details = "; ".join([f"{err['loc']}: {err['msg']}" for err in e.errors()])
        return ValidationIssue(
            error_type=ValidationErrorType.SCHEMA_VIOLATION, message=f"Schema validation failed: {error_details}"
        )


def compute_validation_statistics(issues: List[ValidationIssue]) -> Dict:
    """
    Compute statistics from validation issues.

    Args:
        issues: List of validation issues

    Returns:
        Dict with validation statistics
    """
    total_issues = len(issues)
    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warning")

    error_type_counts = {}
    for issue in issues:
        error_type = issue.error_type.value
        error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

    return {
        "total_issues": total_issues,
        "errors": errors,
        "warnings": warnings,
        "error_type_counts": error_type_counts,
        "is_valid": errors == 0,
    }
