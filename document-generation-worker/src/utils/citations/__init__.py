"""
Citation utilities module.

Consolidated utilities for citation extraction pipeline including validation,
execution tracking, offset calculation, and source reference management.
"""

# Validation utilities
# Source reference utilities
from src.pipeline.citation.utils.source_references import build_source_references

# Execution utilities
from src.utils.citations.execution import (
    ComponentExecutor,
    ExecutionMetrics,
    execute_component,
)

# Offset utilities
from src.utils.citations.offsets import (
    adjust_span_offsets,
    calculate_section_offsets,
)
from src.utils.citations.validation import (
    ValidationErrorType,
    ValidationIssue,
    compute_validation_statistics,
    validate_citation_schema,
    validate_citation_stack,
    validate_citation_type_consistency,
    validate_confidence_score,
)

__all__ = [
    # Validation
    "ValidationErrorType",
    "ValidationIssue",
    "validate_citation_type_consistency",
    "validate_confidence_score",
    "validate_citation_stack",
    "validate_citation_schema",
    "compute_validation_statistics",
    # Execution
    "ComponentExecutor",
    "execute_component",
    "ExecutionMetrics",
    # Offsets
    "calculate_section_offsets",
    "adjust_span_offsets",
    # Source references
    "build_source_references",
]
