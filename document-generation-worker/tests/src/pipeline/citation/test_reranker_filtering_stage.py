"""
Unit tests for SourceSentenceReranker.

Tests filtering logic, threshold enforcement, top-K limiting, and fallback behavior.
"""

from unittest.mock import Mock

import pytest

from src.components.base.reranker import RerankMatch
from src.components.citations.preprocessor import Sentence
from src.pipeline.citation.stages.base import PipelineStageInput
from src.pipeline.citation.stages.source_filtering import SourceSentenceReranker
from src.schemas.citations import SourceMetadata
from src.utils.citations.execution import ExecutionMetrics


@pytest.fixture
def mock_reranker():
    """Create a mock TextPairReranker."""
    return Mock()


@pytest.fixture
def source_filtering_stage(mock_reranker):
    """Create SourceSentenceReranker with default config."""
    return SourceSentenceReranker(
        reranker=mock_reranker,
        enabled=True,
        min_reranker_score=0.50,
        top_k_sources=10,
        fallback_to_all_sources=True,
    )


@pytest.fixture
def sample_sentences():
    """Create sample sentences for testing."""
    return [
        Sentence(
            text="Machine learning needs data",
            normalized_text="machine learning needs data",
            start_index=0,
            end_index=26,
            length=26,
        ),
        Sentence(
            text="Neural networks are complex",
            normalized_text="neural networks are complex",
            start_index=27,
            end_index=54,
            length=27,
        ),
    ]


@pytest.fixture
def sample_sources():
    """Create sample source documents."""
    return [
        SourceMetadata(
            chunk_id="src-1", file_id="file-1", content="ML requires data for training", type="chunk", embedding=None
        ),
        SourceMetadata(chunk_id="src-2", file_id="file-2", content="Deep learning is complex", type="chunk", embedding=None),
        SourceMetadata(chunk_id="src-3", file_id="file-3", content="Python is a language", type="chunk", embedding=None),
        SourceMetadata(
            chunk_id="src-4", file_id="file-4", content="Neural networks process data", type="chunk", embedding=None
        ),
    ]


@pytest.fixture
def pipeline_input(sample_sources):
    """Create pipeline input."""
    return PipelineStageInput(
        request_id="test-req-1",
        generated_text="Machine learning needs data. Neural networks are complex.",
        source_documents=sample_sources,
        execution_metrics=ExecutionMetrics(),
    )


class TestSourceSentenceRerankerBasic:
    """Test basic filtering functionality."""

    def test_filtering_disabled_returns_all_sources(self, mock_reranker, sample_sentences, pipeline_input):
        """When disabled, should return all sources for all sentences."""
        stage = SourceSentenceReranker(
            reranker=mock_reranker,
            enabled=False,
        )

        result = stage.execute(pipeline_input, sample_sentences)

        assert result is not None
        assert len(result) == len(sample_sentences)
        for sentence_idx in range(len(sample_sentences)):
            assert result[sentence_idx] == pipeline_input.source_documents

    def test_empty_sources_returns_empty_mapping(self, mock_reranker, sample_sentences):
        """When sources are empty, should return empty mapping."""
        stage = SourceSentenceReranker(reranker=mock_reranker, enabled=True)
        empty_input = PipelineStageInput(
            request_id="test",
            generated_text="Some text",
            source_documents=[],
            execution_metrics=ExecutionMetrics(),
        )

        result = stage.execute(empty_input, sample_sentences)

        assert result is not None
        for sentence_idx in range(len(sample_sentences)):
            assert result[sentence_idx] == []

    def test_empty_sentences_returns_empty_mapping(self, mock_reranker, pipeline_input):
        """When sentences are empty, should return empty mapping."""
        stage = SourceSentenceReranker(reranker=mock_reranker, enabled=True)

        result = stage.execute(pipeline_input, [])

        assert result is not None
        assert len(result) == 0


class TestRerankerFilteringThresholds:
    """Test score threshold enforcement."""

    def test_min_score_threshold_filtering(self, mock_reranker, sample_sentences, sample_sources, pipeline_input):
        """Sources below min_reranker_score should be filtered out."""
        # Setup reranker to return matches with varying scores
        rerank_results = {
            0: [
                RerankMatch(index=0, score=0.92, text="...", method="reranker"),  # Keep
                RerankMatch(index=1, score=0.45, text="...", method="reranker"),  # Filter (< 0.50)
                RerankMatch(index=2, score=0.78, text="...", method="reranker"),  # Keep
            ],
            1: [
                RerankMatch(index=3, score=0.51, text="...", method="reranker"),  # Keep
            ],
        }
        mock_reranker.run_for_sentences.return_value = rerank_results

        stage = SourceSentenceReranker(
            reranker=mock_reranker,
            enabled=True,
            min_reranker_score=0.50,
            top_k_sources=10,
            fallback_to_all_sources=False,
        )

        result = stage.execute(pipeline_input, sample_sentences)

        # Sentence 0 should have 2 sources (indices 0, 2)
        assert len(result[0]) == 2
        assert result[0][0].chunk_id == "src-1"
        assert result[0][1].chunk_id == "src-3"

        # Sentence 1 should have 1 source (index 3)
        assert len(result[1]) == 1
        assert result[1][0].chunk_id == "src-4"

    def test_top_k_limiting(self, mock_reranker, sample_sentences, sample_sources, pipeline_input):
        """Only top-K sources should be returned per sentence."""
        # Setup reranker with more matches than top_k
        rerank_results = {
            0: [
                RerankMatch(index=0, score=0.95, text="...", method="reranker"),
                RerankMatch(index=1, score=0.90, text="...", method="reranker"),
                RerankMatch(index=2, score=0.85, text="...", method="reranker"),
                RerankMatch(index=3, score=0.80, text="...", method="reranker"),
            ],
            1: [],
        }
        mock_reranker.run_for_sentences.return_value = rerank_results

        stage = SourceSentenceReranker(
            reranker=mock_reranker,
            enabled=True,
            min_reranker_score=0.50,
            top_k_sources=2,  # Limit to 2
            fallback_to_all_sources=False,
        )

        result = stage.execute(pipeline_input, sample_sentences)

        # Sentence 0 should have only 2 sources (top-2)
        assert len(result[0]) == 2
        assert result[0][0].chunk_id == "src-1"  # Highest score
        assert result[0][1].chunk_id == "src-2"  # Second highest

        # Sentence 1 should fallback to all sources (no matches)
        assert len(result[1]) == len(sample_sources)


class TestRerankerFilteringFallback:
    """Test fallback behavior."""

    def test_fallback_to_all_on_empty_results(self, mock_reranker, sample_sentences, sample_sources, pipeline_input):
        """When no sources match threshold, should fallback to all if enabled."""
        rerank_results = {
            0: [],  # No matches for sentence 0
            1: [],  # No matches for sentence 1
        }
        mock_reranker.run_for_sentences.return_value = rerank_results

        stage = SourceSentenceReranker(
            reranker=mock_reranker,
            enabled=True,
            min_reranker_score=0.50,
            top_k_sources=10,
            fallback_to_all_sources=True,
        )

        result = stage.execute(pipeline_input, sample_sentences)

        # Both sentences should have all sources
        assert len(result[0]) == len(sample_sources)
        assert len(result[1]) == len(sample_sources)

    def test_no_fallback_returns_empty(self, mock_reranker, sample_sentences, sample_sources, pipeline_input):
        """When fallback disabled and no matches, should return empty list."""
        rerank_results = {
            0: [],
            1: [],
        }
        mock_reranker.run_for_sentences.return_value = rerank_results

        stage = SourceSentenceReranker(
            reranker=mock_reranker,
            enabled=True,
            min_reranker_score=0.50,
            top_k_sources=10,
            fallback_to_all_sources=False,
        )

        result = stage.execute(pipeline_input, sample_sentences)

        # Both sentences should have empty source lists
        assert result[0] == []
        assert result[1] == []

    def test_fallback_on_reranker_error(self, mock_reranker, sample_sentences, pipeline_input):
        """On reranker error, should fallback to all sources if enabled."""
        mock_reranker.run_for_sentences.side_effect = Exception("Reranker API error")

        stage = SourceSentenceReranker(
            reranker=mock_reranker,
            enabled=True,
            fallback_to_all_sources=True,
        )

        result = stage.execute(pipeline_input, sample_sentences)

        # Should fallback: all sentences have all sources
        assert result is not None
        for sentence_idx in range(len(sample_sentences)):
            assert result[sentence_idx] == pipeline_input.source_documents


class TestRerankerFilteringMetrics:
    """Test metrics calculation."""

    def test_filtering_stats_calculation(self, mock_reranker, sample_sentences, sample_sources, pipeline_input):
        """Metrics should correctly document filtering statistics."""
        rerank_results = {
            0: [
                RerankMatch(index=0, score=0.92, text="...", method="reranker"),
                RerankMatch(index=2, score=0.78, text="...", method="reranker"),
            ],
            1: [
                RerankMatch(index=3, score=0.81, text="...", method="reranker"),
            ],
        }
        mock_reranker.run_for_sentences.return_value = rerank_results

        stage = SourceSentenceReranker(
            reranker=mock_reranker,
            enabled=True,
            min_reranker_score=0.50,
            top_k_sources=10,
        )

        result = stage.execute(pipeline_input, sample_sentences)

        # Check that metrics were recorded
        assert len(pipeline_input.execution_metrics.component_durations) > 0
        assert "RerankerFiltering" in pipeline_input.execution_metrics.component_durations


class TestRerankerFilteringEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_sources_above_threshold(self, mock_reranker, sample_sentences, sample_sources, pipeline_input):
        """All sources above threshold should all be kept."""
        rerank_results = {
            0: [
                RerankMatch(index=i, score=0.90 - i * 0.05, text="...", method="reranker")
                for i in range(len(sample_sources))
            ],
            1: [
                RerankMatch(index=i, score=0.90 - i * 0.05, text="...", method="reranker")
                for i in range(len(sample_sources))
            ],
        }
        mock_reranker.run_for_sentences.return_value = rerank_results

        stage = SourceSentenceReranker(
            reranker=mock_reranker,
            enabled=True,
            min_reranker_score=0.50,
            top_k_sources=10,
        )

        result = stage.execute(pipeline_input, sample_sentences)

        assert len(result[0]) == len(sample_sources)
        assert len(result[1]) == len(sample_sources)

    def test_high_threshold_filters_most(self, mock_reranker, sample_sentences, sample_sources, pipeline_input):
        """High threshold should filter most sources."""
        rerank_results = {
            0: [
                RerankMatch(index=0, score=0.95, text="...", method="reranker"),
                RerankMatch(index=1, score=0.60, text="...", method="reranker"),
                RerankMatch(index=2, score=0.55, text="...", method="reranker"),
            ],
            1: [],
        }
        mock_reranker.run_for_sentences.return_value = rerank_results

        stage = SourceSentenceReranker(
            reranker=mock_reranker,
            enabled=True,
            min_reranker_score=0.80,  # High threshold
            top_k_sources=10,
            fallback_to_all_sources=False,
        )

        result = stage.execute(pipeline_input, sample_sentences)

        # Sentence 0 should have only 1 source (0.95 > 0.80)
        assert len(result[0]) == 1
        assert result[0][0].chunk_id == "src-1"

        # Sentence 1 should have no sources
        assert result[1] == []

    def test_single_sentence_single_source(self, mock_reranker):
        """Should handle single sentence and single source correctly."""
        sentences = [Sentence(text="Text", normalized_text="text", start_index=0, end_index=4, length=4)]
        sources = [SourceMetadata(chunk_id="src-1", file_id="f1", content="content", type="chunk")]

        rerank_results = {
            0: [RerankMatch(index=0, score=0.99, text="...", method="reranker")],
        }
        mock_reranker.run_for_sentences.return_value = rerank_results

        stage = SourceSentenceReranker(reranker=mock_reranker, enabled=True)
        input_data = PipelineStageInput(
            request_id="test",
            generated_text="Text",
            source_documents=sources,
            execution_metrics=ExecutionMetrics(),
        )

        result = stage.execute(input_data, sentences)

        assert len(result[0]) == 1
        assert result[0][0].chunk_id == "src-1"
