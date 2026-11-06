"""
Citation extraction pipeline stages.

Modular stages for the citation extraction workflow:
1. Input Validation - Validates required inputs
2. Preprocessing - Converts text to sentences
3. Reranker Filtering - Filters sources per sentence using semantic scoring (optional)
4. Classification - Classifies sentence-source pairs
5. Aggregation - Groups citations into stacks
6. Validation - Validates and documents on stacks
"""

from src.pipeline.citation.stages.aggregation import AggregationStage
from src.pipeline.citation.stages.base import PipelineStageInput
from src.pipeline.citation.stages.classification import ClassificationStage
from src.pipeline.citation.stages.input_validation import InputValidationStage
from src.pipeline.citation.stages.preprocessing import PreprocessingStage
from src.pipeline.citation.stages.source_filtering import SourceSentenceReranker
from src.pipeline.citation.stages.validation import ValidationStage

__all__ = [
    "PipelineStageInput",
    "InputValidationStage",
    "PreprocessingStage",
    "SourceSentenceReranker",
    "ClassificationStage",
    "AggregationStage",
    "ValidationStage",
]
