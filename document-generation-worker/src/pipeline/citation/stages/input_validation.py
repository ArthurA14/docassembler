"""
Input validation stage for the citation extraction pipeline.

Validates that required inputs are present before proceeding with extraction.
"""

from document_common.utils.logger import Logger

from src.pipeline.citation.stages.base import PipelineStageInput
from src.utils.citations.execution import ComponentExecutor


class InputValidationStage:
    """Validates pipeline inputs."""

    def __init__(self):
        """Initialize the validation stage."""
        self.logger = Logger().get_logger("validation_stage")

    def execute(self, inputs: PipelineStageInput) -> bool:
        """
        Validate inputs.

        Args:
            inputs: Pipeline input data

        Returns:
            True if valid, False otherwise
        """
        executor = ComponentExecutor("InputValidation", inputs.request_id)

        try:
            executor.log_info("Validating inputs")

            if not inputs.generated_text:
                executor.log_error("Generated text is empty")
                inputs.errors.append("Generated text is required")
                return False

            if not inputs.source_documents:
                executor.log_error("No source documents provided")
                inputs.errors.append("At least one source document is required")
                return False

            executor.log_info("Input validation passed")
            inputs.execution_metrics.record_execution(executor)
            return True

        except Exception as e:
            executor.handle_error(e, "Input validation failed")
            inputs.errors.extend(executor.errors)
            inputs.execution_metrics.record_execution(executor)
            return False
