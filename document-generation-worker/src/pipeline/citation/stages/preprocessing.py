"""
Preprocessing stage for the citation extraction pipeline.

Converts raw text into sentences for downstream processing.
"""

from typing import Any, List, Optional

from document_common.utils.logger import Logger

from src.components.citations.preprocessor import TextPreprocessor
from src.pipeline.citation.stages.base import PipelineStageInput
from src.utils.citations.execution import ComponentExecutor


class PreprocessingStage:
    """Preprocesses text into sentences."""

    def __init__(self, preprocessor: TextPreprocessor):
        """
        Initialize the preprocessing stage.

        Args:
            preprocessor: TextPreprocessor component
        """
        self.preprocessor = preprocessor
        self.logger = Logger().get_logger("preprocessing_stage")

    def execute(self, inputs: PipelineStageInput) -> Optional[List[Any]]:
        """
        Execute preprocessing.

        Args:
            inputs: Pipeline input data

        Returns:
            List of processed sentences or None on error
        """
        executor = ComponentExecutor("Preprocessing", inputs.request_id)

        try:
            executor.log_info(f"Preprocessing text (length: {len(inputs.generated_text)})")

            preprocessor_result = self.preprocessor.run(text=inputs.generated_text)
            sentences = preprocessor_result.get("sentences", [])

            if not sentences:
                executor.log_warning("No valid sentences found")
                inputs.warnings.append("No valid sentences found in text")
                inputs.execution_metrics.record_execution(executor)
                return None

            executor.log_info(f"Preprocessed {len(sentences)} sentences")
            inputs.execution_metrics.record_execution(executor)
            return sentences

        except Exception as e:
            executor.handle_error(e, "Preprocessing failed")
            inputs.errors.extend(executor.errors)
            inputs.execution_metrics.record_execution(executor)
            return None
