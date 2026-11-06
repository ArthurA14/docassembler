"""
Unit tests for CitationPipeline orchestrator.

Tests cover:
- MVP mode (Component 2 bypass)
- Production mode (all components)
- Error handling at each stage
- Empty input handling
- Metrics collection
- Request ID tracking
"""

from unittest.mock import Mock, patch

import pytest

from src.components.citations.preprocessor import Sentence
from src.pipeline.citation import CitationPipeline
from document_common.schemas.citations import (
    Citation,
    CitationLabel,
    CitationStack,
    ClassificationResult,
    SourceMetadata,
)


@pytest.fixture
def sample_generated_text():
    """Sample generated document text."""
    return "Machine learning requires large datasets. Neural networks need substantial training data."


@pytest.fixture
def sample_source_documents():
    """Sample source documents."""
    return [
        SourceMetadata(
            id="doc1",
            file_id="file1",
            content="Machine learning requires large datasets for effective training.",
            type="original",
        ),
        SourceMetadata(
            id="doc2",
            file_id="file2",
            content="Neural networks need substantial training data and computational resources.",
            type="original",
        ),
    ]


@pytest.fixture
def sample_sentences():
    """Sample preprocessed sentences."""
    return [
        Sentence(
            text="Machine learning requires large datasets",
            normalized_text="Machine learning requires large datasets",
            start_index=0,
            end_index=41,
            length=41,
        ),
        Sentence(
            text="Neural networks need substantial training data",
            normalized_text="Neural networks need substantial training data",
            start_index=42,
            end_index=88,
            length=46,
        ),
    ]


@pytest.fixture
def sample_classification_result():
    """Sample classification result."""
    return ClassificationResult(
        classification=CitationLabel.CLOSE_PARAPHRASE, confidence=0.85, reasoning="Similar meaning, different words"
    )


@pytest.fixture
def sample_citation():
    """Sample citation object."""
    return Citation(
        source_id="doc1",
        source_doc_id="file1",
        quote_text="Machine learning requires large datasets",
        start_index=0,
        end_index=40,
        citation_type=CitationLabel.CLOSE_PARAPHRASE,
        confidence=0.85,
        similarity_score=0.87,
    )


@pytest.fixture
def sample_citation_stack(sample_citation):
    """Sample citation stack."""
    return CitationStack(
        span_text="Machine learning requires large datasets",
        span_start_index=0,
        span_end_index=40,
        primary_citation=sample_citation,
        supporting_citations=[],
        stack_confidence=0.85,
    )


class TestCitationPipelineMVPMode:
    """Test pipeline in MVP mode (Component 2 bypassed)."""

    @patch("src.pipeline.citation.pipeline.TextPreprocessor")
    @patch("src.pipeline.citation.pipeline.CitationClassifier")
    @patch("src.pipeline.citation.pipeline.CitationAggregator")
    @patch("src.pipeline.citation.pipeline.CitationValidator")
    def test_mvp_mode_bypasses_component_2(
        self,
        mock_validator,
        mock_aggregator,
        mock_classifier,
        mock_preprocessor,
        sample_generated_text,
        sample_source_documents,
        sample_sentences,
        sample_classification_result,
        sample_citation,
        sample_citation_stack,
    ):
        """Test that MVP mode bypasses reranker (Component 2)."""
        # Setup mocks
        mock_preprocessor.return_value.run.return_value = {"sentences": sample_sentences}
        mock_classifier.return_value.run.return_value = {"classification": sample_classification_result}
        mock_aggregator.return_value.run.return_value = {"citation_stacks": [sample_citation_stack]}
        mock_validator.return_value.run.return_value = {
            "valid_citations": [sample_citation_stack],
            "invalid_citations": [],
            "validation_document": Mock(errors=0, warnings=0, model_dump=Mock(return_value={})),
        }

        # Create pipeline in MVP mode
        pipeline = CitationPipeline()
        result = pipeline.run(generated_text=sample_generated_text, source_documents=sample_source_documents)

        # Verify Component 2 was bypassed
        assert result["result"].metrics.component_2_bypassed is True
        assert "reranker" not in result["result"].metrics.component_durations

    def test_mvp_mode_empty_sentences(self, sample_source_documents):
        """Test MVP mode with no valid sentences."""
        with patch("src.pipeline.citation.pipeline.TextPreprocessor") as mock_preprocessor:
            mock_preprocessor.return_value.run.return_value = {"sentences": []}

            pipeline = CitationPipeline()
            result = pipeline.run(generated_text="", source_documents=sample_source_documents)

            assert len(result["result"].citation_stacks) == 0
            assert result["result"].metrics.sentences_processed == 0


class TestCitationPipelineErrorHandling:
    """Test error handling at each pipeline stage."""

    def test_preprocessor_error_handling(self, sample_source_documents):
        """Test handling of preprocessor errors."""
        with patch("src.pipeline.citation.pipeline.TextPreprocessor") as mock_preprocessor:
            mock_preprocessor.return_value.run.side_effect = Exception("Preprocessor failed")

            pipeline = CitationPipeline()
            result = pipeline.run(generated_text="Test text", source_documents=sample_source_documents)

            assert len(result["result"].errors) > 0
            assert "Component 1" in result["result"].errors[0]
            assert len(result["result"].citation_stacks) == 0

    @patch("src.pipeline.citation.pipeline.TextPreprocessor")
    def test_classifier_error_handling(
        self, mock_preprocessor, sample_generated_text, sample_source_documents, sample_sentences
    ):
        """Test handling of classifier errors."""
        mock_preprocessor.return_value.run.return_value = {"sentences": sample_sentences}

        with patch("src.pipeline.citation.pipeline.CitationClassifier") as mock_classifier:
            mock_classifier.return_value.run.side_effect = Exception("Classifier failed")

            pipeline = CitationPipeline()
            result = pipeline.run(generated_text=sample_generated_text, source_documents=sample_source_documents)

            # Pipeline should continue despite classifier errors
            assert result["result"] is not None

    @patch("src.pipeline.citation.pipeline.TextPreprocessor")
    @patch("src.pipeline.citation.pipeline.CitationClassifier")
    def test_aggregator_error_handling(
        self,
        mock_classifier,
        mock_preprocessor,
        sample_generated_text,
        sample_source_documents,
        sample_sentences,
        sample_classification_result,
    ):
        """Test handling of aggregator errors."""
        mock_preprocessor.return_value.run.return_value = {"sentences": sample_sentences}
        mock_classifier.return_value.run.return_value = {"classification": sample_classification_result}

        with patch("src.pipeline.citation.pipeline.CitationAggregator") as mock_aggregator:
            mock_aggregator.return_value.run.side_effect = Exception("Aggregator failed")

            pipeline = CitationPipeline()
            result = pipeline.run(generated_text=sample_generated_text, source_documents=sample_source_documents)

            assert len(result["result"].errors) > 0
            assert "Component 5" in result["result"].errors[0]

    @patch("src.pipeline.citation.pipeline.TextPreprocessor")
    @patch("src.pipeline.citation.pipeline.CitationClassifier")
    @patch("src.pipeline.citation.pipeline.CitationAggregator")
    def test_validator_error_handling(
        self,
        mock_aggregator,
        mock_classifier,
        mock_preprocessor,
        sample_generated_text,
        sample_source_documents,
        sample_sentences,
        sample_classification_result,
        sample_citation_stack,
    ):
        """Test handling of validator errors."""
        mock_preprocessor.return_value.run.return_value = {"sentences": sample_sentences}
        mock_classifier.return_value.run.return_value = {"classification": sample_classification_result}
        mock_aggregator.return_value.run.return_value = {"citation_stacks": [sample_citation_stack]}

        with patch("src.pipeline.citation.pipeline.CitationValidator") as mock_validator:
            mock_validator.return_value.run.side_effect = Exception("Validator failed")

            pipeline = CitationPipeline()
            result = pipeline.run(generated_text=sample_generated_text, source_documents=sample_source_documents)

            # Pipeline should continue with unvalidated stacks
            assert len(result["result"].errors) > 0
            assert "Component 6" in result["result"].errors[0]


class TestCitationPipelineMetrics:
    """Test metrics collection and tracking."""

    @patch("src.pipeline.citation.pipeline.TextPreprocessor")
    @patch("src.pipeline.citation.pipeline.CitationClassifier")
    @patch("src.pipeline.citation.pipeline.CitationAggregator")
    @patch("src.pipeline.citation.pipeline.CitationValidator")
    def test_metrics_collection(
        self,
        mock_validator,
        mock_aggregator,
        mock_classifier,
        mock_preprocessor,
        sample_generated_text,
        sample_source_documents,
        sample_sentences,
        sample_classification_result,
        sample_citation_stack,
    ):
        """Test that metrics are collected properly."""
        mock_preprocessor.return_value.run.return_value = {"sentences": sample_sentences}
        mock_classifier.return_value.run.return_value = {"classification": sample_classification_result}
        mock_aggregator.return_value.run.return_value = {"citation_stacks": [sample_citation_stack]}
        mock_validator.return_value.run.return_value = {
            "valid_citations": [sample_citation_stack],
            "invalid_citations": [],
            "validation_document": Mock(errors=0, warnings=1, model_dump=Mock(return_value={})),
        }

        pipeline = CitationPipeline()
        result = pipeline.run(generated_text=sample_generated_text, source_documents=sample_source_documents)

        metrics = result["result"].metrics

        assert metrics.request_id is not None
        assert metrics.total_duration_seconds > 0
        assert "preprocessor" in metrics.component_durations
        assert "classifier" in metrics.component_durations
        assert "aggregator" in metrics.component_durations
        assert "validator" in metrics.component_durations
        assert metrics.sentences_processed == len(sample_sentences)
        assert metrics.stacks_created == 1

    def test_custom_request_id(self, sample_generated_text, sample_source_documents):
        """Test that custom request ID is preserved."""
        with patch("src.pipeline.citation.pipeline.TextPreprocessor") as mock_preprocessor:
            mock_preprocessor.return_value.run.return_value = {"sentences": []}

            pipeline = CitationPipeline()
            custom_id = "custom-request-123"
            result = pipeline.run(
                generated_text=sample_generated_text,
                source_documents=sample_source_documents,
                request_id=custom_id,
            )

            assert result["result"].metrics.request_id == custom_id

    def test_component_metrics_aggregation(self):
        """Test get_metrics aggregates component metrics."""
        pipeline = CitationPipeline()

        # Mock component metrics
        pipeline.classifier.get_metrics = Mock(return_value={"total_classifications": 10})
        pipeline.aggregator.get_metrics = Mock(return_value={"total_stacks_created": 5})
        pipeline.validator.get_metrics = Mock(return_value={"total_validations": 5})

        metrics = pipeline.get_metrics()

        assert "classifier" in metrics
        assert "aggregator" in metrics
        assert "validator" in metrics
        assert metrics["classifier"]["total_classifications"] == 10


class TestCitationPipelineIntegration:
    """Integration tests with minimal mocking."""

    def test_empty_source_documents(self, sample_generated_text):
        """Test pipeline with empty source documents."""
        pipeline = CitationPipeline()
        result = pipeline.run(generated_text=sample_generated_text, source_documents=[])

        assert result["result"] is not None
        assert len(result["result"].citation_stacks) == 0

    def test_empty_generated_text(self, sample_source_documents):
        """Test pipeline with empty generated text."""
        pipeline = CitationPipeline()
        result = pipeline.run(generated_text="", source_documents=sample_source_documents)

        assert result["result"] is not None
        assert result["result"].metrics.sentences_processed == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
