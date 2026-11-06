"""
Base classes for citation pipeline stages.

Defines the common input data structure used across all pipeline stages.
"""

from typing import List, Optional

from document_common.schemas.citations import SourceMetadata

from src.utils.citations.execution import ExecutionMetrics


class PipelineStageInput:
    """Input data for a pipeline stage."""

    def __init__(
        self,
        request_id: str,
        generated_text: str,
        source_documents: List[SourceMetadata],
        execution_metrics: Optional[ExecutionMetrics] = None,
    ):
        """Initialize stage input."""
        self.request_id = request_id
        self.generated_text = generated_text
        self.source_documents = source_documents
        self.execution_metrics = execution_metrics or ExecutionMetrics()
        self.errors: List[str] = []
        self.warnings: List[str] = []
