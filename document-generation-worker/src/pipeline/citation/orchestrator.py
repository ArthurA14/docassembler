"""
Pipeline orchestrator for coordinating citation extraction stages.

Manages the sequential execution of pipeline stages, handling data flow,
error collection, and metrics aggregation across all stages.
"""

from typing import Any, Dict, List, Optional

from document_common.schemas.citations import (
    CitationPipelineResult,
    PipelineMetrics,
    SourceReferences,
)
from document_common.utils.logger import Logger

from src.pipeline.citation.stages import PipelineStageInput
from src.pipeline.citation.utils.source_references import build_source_references
from src.utils.citations.execution import ExecutionMetrics


class PipelineOrchestrator:
    """Orchestrates execution of pipeline stages in sequence."""

    def __init__(self):
        """Initialize the pipeline orchestrator."""
        self.logger = Logger().get_logger("pipeline_orchestrator")
        self.metrics = ExecutionMetrics()

    def execute(
        self,
        stages: List[Any],
        inputs: PipelineStageInput,
    ) -> Dict[str, CitationPipelineResult]:
        """
        Execute pipeline stages in sequence.

        Args:
            stages: List of stages to execute in order
            inputs: Pipeline input with request_id, generated_text, source_documents

        Returns:
            Dict with 'result' key containing CitationPipelineResult
        """
        self.logger.info(f"[{inputs.request_id}] Starting orchestrator with {len(stages)} stages")

        # State tracking through pipeline
        state = {
            "sentences": None,
            "citations": None,
            "citation_stacks": None,
            "valid_stacks": None,
            "invalid_stacks": None,
            "validation_document": None,
        }

        # Execute each stage
        for stage_idx, stage in enumerate(stages):
            stage_name = type(stage).__name__
            self.logger.debug(f"[{inputs.request_id}] Executing stage {stage_idx + 1}/{len(stages)}: {stage_name}")

            try:
                # Stage 1: Input Validation
                if stage_name == "InputValidationStage":
                    if not stage.execute(inputs):
                        self.logger.error(f"[{inputs.request_id}] Input validation failed, aborting pipeline")
                        return self._build_error_result(inputs.request_id, inputs.execution_metrics)

                # Stage 2: Preprocessing
                elif stage_name == "PreprocessingStage":
                    sentences = stage.execute(inputs)
                    if sentences is None:
                        self.logger.error(f"[{inputs.request_id}] Preprocessing failed, aborting")
                        return self._build_error_result(inputs.request_id, inputs.execution_metrics)
                    state["sentences"] = sentences

                # Stage 3: Reranker Filtering (Optional)
                elif stage_name == "SourceSentenceReranker":
                    if state["sentences"] is None:
                        self.logger.error(f"[{inputs.request_id}] No sentences for reranker filtering")
                        return self._build_error_result(inputs.request_id, inputs.execution_metrics)

                    filtered_sources = stage.execute(inputs, state["sentences"])
                    if filtered_sources is None:
                        self.logger.error(f"[{inputs.request_id}] Reranker filtering failed, aborting")
                        return self._build_error_result(inputs.request_id, inputs.execution_metrics)
                    state["filtered_sources"] = filtered_sources

                # Stage 4: Classification
                elif stage_name == "ClassificationStage":
                    if state["sentences"] is None:
                        self.logger.error(f"[{inputs.request_id}] No sentences for classification")
                        return self._build_error_result(inputs.request_id, inputs.execution_metrics)

                    citations = stage.execute(inputs, state["sentences"], state)
                    if citations is None:
                        self.logger.error(f"[{inputs.request_id}] Classification failed, aborting")
                        return self._build_error_result(inputs.request_id, inputs.execution_metrics)

                    state["citations"] = citations

                    # If no citations, return early with empty result
                    if not citations:
                        self.logger.info(f"[{inputs.request_id}] No citations found, returning empty result")
                        return self._build_result(
                            inputs.request_id,
                            inputs.execution_metrics,
                            [],
                            [],
                            None,
                            [],
                            SourceReferences(references={}),
                        )

                # Stage 4: Aggregation
                elif stage_name == "AggregationStage":
                    if state["citations"] is None:
                        self.logger.error(f"[{inputs.request_id}] No citations for aggregation")
                        return self._build_error_result(inputs.request_id, inputs.execution_metrics)

                    citation_stacks = stage.execute(inputs, state["citations"])
                    if citation_stacks is None:
                        self.logger.error(f"[{inputs.request_id}] Aggregation failed, aborting")
                        return self._build_error_result(inputs.request_id, inputs.execution_metrics)

                    state["citation_stacks"] = citation_stacks

                # Stage 5: Validation
                elif stage_name == "ValidationStage":
                    if state["citation_stacks"] is None:
                        self.logger.error(f"[{inputs.request_id}] No stacks for validation")
                        return self._build_error_result(inputs.request_id, inputs.execution_metrics)

                    valid_stacks, invalid_stacks, validation_document = stage.execute(inputs, state["citation_stacks"])

                    state["valid_stacks"] = valid_stacks
                    state["invalid_stacks"] = invalid_stacks
                    state["validation_document"] = validation_document

                else:
                    self.logger.warning(f"[{inputs.request_id}] Unknown stage: {stage_name}")

            except Exception as e:
                self.logger.error(f"[{inputs.request_id}] Stage {stage_name} failed: {e}")
                inputs.errors.append(f"{stage_name} failed: {str(e)}")
                return self._build_error_result(inputs.request_id, inputs.execution_metrics)

        # Build final result from state
        self.logger.debug(f"[{inputs.request_id}] All stages completed, building result")

        # Use validated stacks if available, otherwise use aggregated stacks
        final_stacks = state.get("valid_stacks") or state.get("citation_stacks") or []

        # Build source references from original citations
        source_refs = build_source_references(
            state.get("citations") or [],
            inputs.source_documents,
        )

        return self._build_result(
            inputs.request_id,
            inputs.execution_metrics,
            final_stacks,
            state.get("invalid_stacks") or [],
            state.get("validation_document"),
            inputs.errors,
            source_refs,
        )

    def _build_result(
        self,
        request_id: str,
        execution_metrics: ExecutionMetrics,
        valid_stacks: List[Any],
        invalid_stacks: List[Any],
        validation_document: Optional[Dict[str, Any]],
        errors: List[str],
        source_references: SourceReferences,
    ) -> Dict[str, CitationPipelineResult]:
        """
        Build final CitationPipelineResult from pipeline state.

        Args:
            request_id: Request ID for tracking
            execution_metrics: Collected metrics from all stages
            valid_stacks: Valid citation stacks
            invalid_stacks: Invalid citation stacks
            validation_document: Validation document as dict (converted from Pydantic Validationdocument).
                NOTE: Must be converted via model_dump() before passing to match CitationPipelineResult schema.
            errors: Error messages collected during execution
            source_references: Source documents referenced

        Returns:
            Dict with 'result' key containing CitationPipelineResult
        """
        metrics_dict = execution_metrics.to_dict()

        metrics = PipelineMetrics(
            request_id=request_id,
            total_duration_seconds=metrics_dict["total_duration_seconds"],
            component_durations=metrics_dict["component_durations"],
            sentences_processed=0,  # Would be extracted from stage metrics
            citations_created=sum(
                len(stack.supporting_citations) + 1 for stack in valid_stacks if hasattr(stack, "supporting_citations")
            ),
            stacks_created=len(valid_stacks),
            validation_errors=(validation_document.errors if validation_document else 0),
            validation_warnings=(validation_document.warnings if validation_document else 0),
            component_2_bypassed=True,  # Would be from pipeline config
        )

        result = CitationPipelineResult(
            citation_stacks=valid_stacks,
            invalid_citations=invalid_stacks,
            source_references=source_references,
            metrics=metrics,
            # Convert Validationdocument Pydantic object to dict for schema compatibility
            validation_document=validation_document.model_dump() if validation_document else None,
            errors=errors,
            warnings=execution_metrics.get_all_warnings(),
        )

        self.logger.info(
            f"[{request_id}] Result built: {len(valid_stacks)} valid stacks, "
            f"{len(invalid_stacks)} invalid, duration: {metrics_dict['total_duration_seconds']:.3f}s"
        )

        return {"result": result}

    def _build_error_result(
        self,
        request_id: str,
        execution_metrics: ExecutionMetrics,
    ) -> Dict[str, CitationPipelineResult]:
        """
        Build error result when pipeline execution fails.

        Args:
            request_id: Request ID for tracking
            execution_metrics: Metrics collected before error

        Returns:
            Dict with 'result' key containing empty CitationPipelineResult
        """
        metrics_dict = execution_metrics.to_dict()

        metrics = PipelineMetrics(
            request_id=request_id,
            total_duration_seconds=metrics_dict["total_duration_seconds"],
            component_durations=metrics_dict["component_durations"],
            component_2_bypassed=True,
        )

        result = CitationPipelineResult(
            source_references=SourceReferences(references={}),
            metrics=metrics,
            errors=execution_metrics.get_all_errors(),
            warnings=execution_metrics.get_all_warnings(),
        )

        return {"result": result}
