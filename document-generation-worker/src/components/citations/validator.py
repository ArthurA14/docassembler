from typing import Dict, List, Optional, Union

from haystack import component
from pydantic import BaseModel
from document_common.schemas.citations import Citation, CitationStack, SourceMetadata
from document_common.utils.logger import Logger

from config import config
from src.utils.citations.validation import (
    ValidationIssue,
    compute_validation_statistics,
    validate_citation_schema,
    validate_citation_stack,
    validate_citation_type_consistency,
    validate_confidence_score,
)


class ValidationDocument(BaseModel):
    """document of validation results"""

    total_citations: int
    valid_citations: int
    invalid_citations: int
    total_issues: int
    errors: int
    warnings: int
    error_type_counts: Dict[str, int]
    is_valid: bool
    issues: List[Dict]


@component
class CitationValidator:
    """
    Haystack component for validating citation data model compliance.

    Validates citations and citation stacks against schema requirements,
    offset accuracy, semantic consistency, and integrity constraints.
    Used in citation pipeline to ensure data quality before storage/display.
    """

    def __init__(
        self,
        validate_quote_match: bool = True,
        validate_type_consistency: bool = True,
        validate_confidence: bool = True,
        allow_whitespace_normalize: bool = True,
        max_offset_drift: int = 10,
        fail_on_error: bool = False,
    ):
        """
        Initialize the citation validator.

        Args:
            validate_quote_match: Check if quote text matches source at offsets
            validate_type_consistency: Check citation type matches similarity score
            validate_confidence: Validate confidence score ranges and consistency
            allow_whitespace_normalize: Allow whitespace normalization in comparisons
            max_offset_drift: Maximum character drift for offset correction
            fail_on_error: Raise exception on validation errors (vs. returning document)
        """
        self.logger = Logger().get_logger("citation_validator")
        citation_config = config.get_current_config().citations

        validation_config = citation_config.validation
        self.validate_quote_match = validate_quote_match
        self.validate_type_consistency = validate_type_consistency
        self.validate_confidence = validate_confidence
        self.allow_whitespace_normalize = validation_config.get(
            "allow_whitespace_normalize", allow_whitespace_normalize
        )
        self.max_offset_drift = validation_config.get("max_offset_drift", max_offset_drift)
        self.fail_on_error = fail_on_error

        self.total_validations = 0
        self.total_valid = 0
        self.total_invalid = 0

        self.logger.debug(
            f"CitationValidator initialized (quote_match: {self.validate_quote_match}, "
            f"type_consistency: {self.validate_type_consistency}, "
            f"confidence: {self.validate_confidence})"
        )

    @component.output_types(
        valid_citations=List[Union[Citation, CitationStack]],
        invalid_citations=List[Union[Citation, CitationStack]],
        validation_document=ValidationDocument,
    )
    def run(
        self,
        citations: List[Union[Citation, CitationStack]],
        sources: Optional[Dict[str, SourceMetadata]] = None,
    ) -> Dict:
        """
        Validate citations for data model compliance.

        Args:
            citations: List of Citation or CitationStack objects to validate
            sources: Optional dict mapping source_id to SourceMetadata for verification

        Returns:
            Dict with keys:
                - valid_citations: Citations that passed all validation checks
                - invalid_citations: Citations that failed validation
                - validation_document: Detailed validation document
        """
        if not citations:
            self.logger.debug("No citations to validate")
            return {
                "valid_citations": [],
                "invalid_citations": [],
                "validation_document": ValidationDocument(
                    total_citations=0,
                    valid_citations=0,
                    invalid_citations=0,
                    total_issues=0,
                    errors=0,
                    warnings=0,
                    error_type_counts={},
                    is_valid=True,
                    issues=[],
                ),
            }

        self.total_validations += len(citations)

        valid_citations = []
        invalid_citations = []
        all_issues = []

        for idx, citation in enumerate(citations):
            if isinstance(citation, CitationStack):
                issues = self._validate_citation_stack(citation, sources, idx)
            elif isinstance(citation, Citation):
                issues = self._validate_citation(citation, sources, idx)
            else:
                self.logger.error(f"Unknown citation type: {type(citation)}")
                continue

            error_count = sum(1 for issue in issues if issue.severity == "error")

            if error_count == 0:
                valid_citations.append(citation)
                self.total_valid += 1
            else:
                invalid_citations.append(citation)
                self.total_invalid += 1

            all_issues.extend(issues)

        stats = compute_validation_statistics(all_issues)

        validation_document = ValidationDocument(
            total_citations=len(citations),
            valid_citations=len(valid_citations),
            invalid_citations=len(invalid_citations),
            total_issues=stats["total_issues"],
            errors=stats["errors"],
            warnings=stats["warnings"],
            error_type_counts=stats["error_type_counts"],
            is_valid=stats["is_valid"],
            issues=[issue.to_dict() for issue in all_issues],
        )

        self.logger.debug(
            f"Validation complete: {len(valid_citations)}/{len(citations)} valid, "
            f"{stats['errors']} errors, {stats['warnings']} warnings"
        )

        if self.fail_on_error and stats["errors"] > 0:
            error_summary = "\n".join([str(issue) for issue in all_issues if issue.severity == "error"])
            raise ValueError(f"Citation validation failed with {stats['errors']} errors:\n{error_summary}")

        return {
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "validation_document": validation_document,
        }

    def _validate_citation(
        self, citation: Citation, sources: Optional[Dict[str, SourceMetadata]], citation_index: int
    ) -> List[ValidationIssue]:
        """
        Validate a single citation.

        Args:
            citation: Citation to validate
            sources: Source metadata dict
            citation_index: Index in citation list for error documenting

        Returns:
            List of validation issues found
        """
        issues = []

        schema_issue = validate_citation_schema(citation)
        if schema_issue:
            schema_issue.citation_index = citation_index
            issues.append(schema_issue)
            return issues

        if self.validate_confidence:
            confidence_issue = validate_confidence_score(citation)
            if confidence_issue:
                confidence_issue.citation_index = citation_index
                issues.append(confidence_issue)

        if self.validate_type_consistency:
            type_issue = validate_citation_type_consistency(citation)
            if type_issue:
                type_issue.citation_index = citation_index
                issues.append(type_issue)

        if self.validate_quote_match:
            # validate_quote_text_match() was removed - it always returned None
            # after Citation schema refactoring. Quote text validation is now handled
            # by Pydantic's Citation.validate_generated_span() validator.

            return issues

    def _validate_citation_stack(
        self, stack: CitationStack, sources: Optional[Dict[str, SourceMetadata]], stack_index: int
    ) -> List[ValidationIssue]:
        """
        Validate a citation stack.

        Args:
            stack: CitationStack to validate
            sources: Source metadata dict
            stack_index: Index in stack list for error documenting

        Returns:
            List of validation issues found
        """
        issues = []

        stack_issues = validate_citation_stack(stack, sources)
        for issue in stack_issues:
            issue.citation_index = stack_index
            issues.append(issue)

        primary_issues = self._validate_citation(stack.primary_citation, sources, stack_index)
        for issue in primary_issues:
            issue.field = f"primary_citation.{issue.field}" if issue.field else "primary_citation"
            issues.append(issue)

        for support_idx, supporting_citation in enumerate(stack.supporting_citations):
            support_issues = self._validate_citation(supporting_citation, sources, stack_index)
            for issue in support_issues:
                issue.field = (
                    f"supporting_citations[{support_idx}].{issue.field}"
                    if issue.field
                    else f"supporting_citations[{support_idx}]"
                )
                issues.append(issue)

        return issues

    def get_metrics(self) -> Dict:
        """
        Get validation metrics for monitoring.

        Returns:
            Dict with validation statistics
        """
        return {
            "total_validations": self.total_validations,
            "total_valid": self.total_valid,
            "total_invalid": self.total_invalid,
            "valid_rate": self.total_valid / self.total_validations if self.total_validations > 0 else 0,
            "invalid_rate": self.total_invalid / self.total_validations if self.total_validations > 0 else 0,
        }

    def reset_metrics(self) -> None:
        """Reset all validation metrics."""
        self.total_validations = 0
        self.total_valid = 0
        self.total_invalid = 0
        self.logger.debug("Validation metrics reset")
