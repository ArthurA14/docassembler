"""
Aggregation stage for the citation extraction pipeline.

Groups citations into stacks based on similarity and relevance.
"""

from typing import Any, List, Optional

from document_common.schemas.citations import Citation
from document_common.utils.logger import Logger

from src.components.citations.aggregator import CitationAggregator
from src.pipeline.citation.stages.base import PipelineStageInput
from src.utils.citations.execution import ComponentExecutor


class AggregationStage:
    """Aggregates citations into stacks."""

    def __init__(self, aggregator: CitationAggregator):
        """
        Initialize the aggregation stage.

        Args:
            aggregator: CitationAggregator component
        """
        self.aggregator = aggregator
        self.logger = Logger().get_logger("aggregation_stage")

    def execute(self, inputs: PipelineStageInput, citations: List[Citation]) -> Optional[List[Any]]:
        """
        Execute aggregation.

        Args:
            inputs: Pipeline input data
            citations: List of citations to aggregate

        Returns:
            List of citation stacks or None on error
        """
        executor = ComponentExecutor("Aggregation", inputs.request_id)

        try:
            executor.log_info(f"Aggregating {len(citations)} citations")

            aggregator_result = self.aggregator.run(citations=citations, generated_text=inputs.generated_text)

            citation_stacks = aggregator_result.get("citation_stacks", [])
            executor.log_info(f"Created {len(citation_stacks)} citation stacks")

            inputs.execution_metrics.record_execution(executor)
            return citation_stacks

        except Exception as e:
            executor.handle_error(e, "Aggregation stage failed")
            inputs.errors.extend(executor.errors)
            inputs.execution_metrics.record_execution(executor)
            return None
