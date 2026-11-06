"""
Pipeline builder pattern for citation extraction.

Provides a declarative, fluent interface for constructing citation extraction
pipelines. Supports different configurations and enables
easy addition/removal of pipeline stages.
"""

from typing import List, Optional

from document_common.utils.logger import Logger

from src.components.base.text_reranker import TextPairReranker
from src.components.citations.aggregator import CitationAggregator
from src.components.citations.classifier.classification import CitationClassifier
from src.components.citations.preprocessor import TextPreprocessor
from src.components.citations.validator import CitationValidator
from src.pipeline.citation.stages import (
    AggregationStage,
    ClassificationStage,
    InputValidationStage,
    PreprocessingStage,
    SourceSentenceReranker,
    ValidationStage,
)


class PipelineBuilder:
    """Builder for constructing citation extraction pipelines."""

    def __init__(self, source_filtering_config: Optional[dict] = None):
        """
        Initialize the pipeline builder.

        Args:
            source_filtering_config: Configuration dict for reranker filtering stage
        """
        self.logger = Logger().get_logger("pipeline_builder")
        self.stages: List = []
        self._preprocessor: Optional[TextPreprocessor] = None
        self._reranker: Optional[TextPairReranker] = None
        self._classifier: Optional[CitationClassifier] = None
        self._aggregator: Optional[CitationAggregator] = None
        self._validator: Optional[CitationValidator] = None
        self._source_filtering_config: Optional[dict] = source_filtering_config

    def set_components(
        self,
        preprocessor: TextPreprocessor,
        classifier: CitationClassifier,
        aggregator: CitationAggregator,
        validator: CitationValidator,
        reranker: Optional[TextPairReranker] = None,
    ) -> "PipelineBuilder":
        """
        Set the components to be used in the pipeline.

        Args:
            preprocessor: TextPreprocessor component
            classifier: CitationClassifier component
            aggregator: CitationAggregator component
            validator: CitationValidator component
            reranker: Optional TextPairReranker for production mode

        Returns:
            Self for method chaining
        """
        self._preprocessor = preprocessor
        self._classifier = classifier
        self._aggregator = aggregator
        self._validator = validator
        self._reranker = reranker
        return self

    def add_input_validation(self) -> "PipelineBuilder":
        """
        Add input validation stage.

        Returns:
            Self for method chaining
        """
        if any(isinstance(stage, InputValidationStage) for stage in self.stages):
            self.logger.warning("Input validation stage already added")
            return self

        self.stages.append(InputValidationStage())
        return self

    def add_preprocessing(self) -> "PipelineBuilder":
        """
        Add text preprocessing stage.

        Returns:
            Self for method chaining
        """
        if not self._preprocessor:
            raise ValueError("Preprocessor component not set. Call set_components() first.")

        if any(isinstance(stage, PreprocessingStage) for stage in self.stages):
            self.logger.warning("Preprocessing stage already added")
            return self

        self.stages.append(PreprocessingStage(self._preprocessor))
        return self

    def add_source_filtering(self) -> "PipelineBuilder":
        """
        Add reranker filtering stage for semantic source pre-filtering.

        Returns:
            Self for method chaining
        """
        if not self._reranker:
            self.logger.warning("Reranker not set. Skipping reranker filtering stage.")
            return self

        else:
            config = self._source_filtering_config

        if any(isinstance(stage, SourceSentenceReranker) for stage in self.stages):
            self.logger.warning("Reranker filtering stage already added")
            return self

        self.stages.append(
            SourceSentenceReranker(
                reranker=self._reranker,
                enabled=config.get("enabled", True),
                min_reranker_score=config.get("min_reranker_score", 0.50),
                top_k_sources=config.get("top_k_sources", 10),
                fallback_to_all_sources=config.get("fallback_to_all_sources", True),
            )
        )
        return self

    def add_classification(self) -> "PipelineBuilder":
        """
        Add batch classification stage.

        Returns:
            Self for method chaining
        """
        if not self._classifier:
            raise ValueError("Classifier component not set. Call set_components() first.")

        if any(isinstance(stage, ClassificationStage) for stage in self.stages):
            self.logger.warning("Classification stage already added")
            return self

        self.stages.append(
            ClassificationStage(
                classifier=self._classifier,
                reranker=self._reranker,
            )
        )
        return self

    def add_aggregation(self) -> "PipelineBuilder":
        """
        Add citation aggregation stage.

        Returns:
            Self for method chaining
        """
        if not self._aggregator:
            raise ValueError("Aggregator component not set. Call set_components() first.")

        if any(isinstance(stage, AggregationStage) for stage in self.stages):
            self.logger.warning("Aggregation stage already added")
            return self

        self.stages.append(AggregationStage(self._aggregator))
        return self

    def add_validation(self) -> "PipelineBuilder":
        """
        Add citation validation stage.

        Returns:
            Self for method chaining
        """
        if not self._validator:
            raise ValueError("Validator component not set. Call set_components() first.")

        if any(isinstance(stage, ValidationStage) for stage in self.stages):
            self.logger.warning("Validation stage already added")
            return self

        self.stages.append(ValidationStage(self._validator))
        return self

    def build_pipeline(self) -> "PipelineBuilder":
        """
        Build a production mode pipeline (full validation + optional reranker filtering).

        Production pipeline: Input Validation → Preprocessing → [RerankerFiltering] →
        Classification → Aggregation → Validation

        Note: RerankerFiltering is added if reranker and config are available.

        Returns:
            Self for method chaining
        """
        self.logger.debug("Building production mode pipeline")
        self.clear()
        self.add_input_validation().add_preprocessing()

        # Add reranker filtering if configured
        if self._reranker and self._source_filtering_config:
            if self._source_filtering_config.get("enabled", True):
                self.add_source_filtering()

        return self.add_classification().add_aggregation().add_validation()

    def clear(self) -> "PipelineBuilder":
        """
        Clear all stages from the pipeline.

        Returns:
            Self for method chaining
        """
        self.stages = []
        return self

    def get_stages(self) -> List:
        """
        Get the configured pipeline stages.

        Returns:
            List of pipeline stages in execution order
        """
        return self.stages.copy()

    def validate(self) -> bool:
        """
        Validate the pipeline configuration.

        Returns:
            True if valid, False otherwise
        """
        if not self.stages:
            self.logger.error("Pipeline has no stages")
            return False

        required_stages = [PreprocessingStage, ClassificationStage, AggregationStage]
        configured_types = [type(stage) for stage in self.stages]

        for required in required_stages:
            if required not in configured_types:
                self.logger.error(f"Missing required stage: {required.__name__}")
                return False

        self.logger.debug(f"Pipeline validation passed: {len(self.stages)} stages")
        return True

    def get_stage_count(self) -> int:
        """Get the number of stages in the pipeline."""
        return len(self.stages)

    def get_stage_names(self) -> List[str]:
        """Get names of all stages in the pipeline."""
        return [type(stage).__name__ for stage in self.stages]

    def __repr__(self) -> str:
        """Get string representation of the pipeline configuration."""
        stages = " → ".join(self.get_stage_names())
        return f"CitationPipeline(): [{stages}]"


class PipelineFactory:
    """Factory for creating pre-configured pipelines."""

    @staticmethod
    def create_pipeline(
        preprocessor: TextPreprocessor,
        classifier: CitationClassifier,
        aggregator: CitationAggregator,
        validator: CitationValidator,
        reranker: Optional[TextPairReranker] = None,
        source_filtering_config: Optional[dict] = None,
    ) -> PipelineBuilder:
        """
        Create a production mode pipeline.

        Args:
            preprocessor: TextPreprocessor component
            classifier: CitationClassifier component
            aggregator: CitationAggregator component
            validator: CitationValidator component
            reranker: Optional TextPairReranker for top-K filtering
            source_filtering_config: Configuration dict for reranker filtering stage

        Returns:
            Configured PipelineBuilder in production mode
        """
        return (
            PipelineBuilder(source_filtering_config=source_filtering_config)
            .set_components(preprocessor, classifier, aggregator, validator, reranker)
            .build_pipeline()
        )

    @staticmethod
    def create_custom_pipeline() -> PipelineBuilder:
        """
        Create a custom pipeline for advanced use cases.

        Returns:
            Empty PipelineBuilder for custom configuration
        """
        return PipelineBuilder()
