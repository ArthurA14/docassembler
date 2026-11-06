"""
Validation stage for the citation extraction pipeline.

Validates citation stacks and returns validation results and documents.
"""

from typing import Any, Dict, List, Optional

from document_common.utils.logger import Logger

from src.components.citations.validator import CitationValidator
from src.pipeline.citation.stages.base import PipelineStageInput
from src.utils.citations.execution import ComponentExecutor


class ValidationStage:
    """Validates citation stacks."""

    def __init__(self, validator: CitationValidator):
        """
        Initialize the validation stage.

        Args:
            validator: CitationValidator component
        """
        self.validator = validator
        self.logger = Logger().get_logger("validation_stage")

    def execute(
        self, inputs: PipelineStageInput, citation_stacks: List[Any]
    ) -> tuple[Optional[List[Any]], Optional[List[Any]], Optional[Dict[str, Any]]]:
        """
        Execute validation.

        Args:
            inputs: Pipeline input data
            citation_stacks: Citation stacks to validate

        Returns:
            Tuple of (valid_stacks, invalid_stacks, validation_document_dict) where:
            - valid_stacks: List of validated citation stacks
            - invalid_stacks: List of citations that failed validation
            - validation_document_dict: Validationdocument converted to dict (via model_dump()).
              NOTE: This will be further converted to CitationPipelineResult.validation_document.
              Returns (None, None, None) on error.
        """
        executor = ComponentExecutor("Validation", inputs.request_id)

        try:
            executor.log_info(f"Validating {len(citation_stacks)} citation stacks")

            # Create source lookup dict
            sources_dict = {doc.id: doc for doc in inputs.source_documents}

            validation_result = self.validator.run(
                citations=citation_stacks, sources=sources_dict
            )

            valid_stacks = validation_result.get("valid_citations", [])
            invalid_stacks = validation_result.get("invalid_citations", [])
            validation_document = validation_result.get("validation_document")

            executor.log_info(
                f"Validation complete: {len(valid_stacks)} valid, "
                f"{len(invalid_stacks)} invalid"
            )

            inputs.execution_metrics.record_execution(executor)
            return valid_stacks, invalid_stacks, validation_document

        except Exception as e:
            executor.handle_error(e, "Validation stage failed")
            inputs.errors.extend(executor.errors)
            inputs.execution_metrics.record_execution(executor)
            # Return stacks unvalidated on error
            return citation_stacks, [], None
