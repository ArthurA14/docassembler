"""Citation pipeline - Orchestrates citation extraction, classification, and validation."""

from src.pipeline.citation.builder import PipelineBuilder, PipelineFactory
from src.pipeline.citation.orchestrator import PipelineOrchestrator
from src.pipeline.citation.pipeline import CitationPipeline
from src.pipeline.citation.stages import (
    AggregationStage,
    ClassificationStage,
    InputValidationStage,
    PipelineStageInput,
    PreprocessingStage,
    ValidationStage,
)

__all__ = [
    # Main pipeline
    "CitationPipeline",
    # Builder and orchestration
    "PipelineBuilder",
    "PipelineFactory",
    "PipelineOrchestrator",
    # Stages
    "PipelineStageInput",
    "InputValidationStage",
    "PreprocessingStage",
    "ClassificationStage",
    "AggregationStage",
    "ValidationStage",
]
