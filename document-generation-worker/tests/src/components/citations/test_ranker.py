import pytest

from src.components.citations.ranker import CitationRanker, RankingStrategy
from src.schemas.citations import Citation, CitationLabel


class TestCitationRanker:
    """Test suite for CitationRanker"""

    @pytest.fixture
    def sample_citations(self):
        """Create sample citations with varying scores"""
        return [
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="High confidence quote",
                start_index=0,
                end_index=21,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.95,
                similarity_score=0.97,
            ),
            Citation(
                source_id="source-2",
                source_doc_id="doc-2",
                quote_text="Medium confidence quote",
                start_index=0,
                end_index=23,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.85,
                similarity_score=0.88,
            ),
            Citation(
                source_id="source-3",
                source_doc_id="doc-3",
                quote_text="Low confidence quote",
                start_index=0,
                end_index=20,
                citation_type=CitationLabel.LOOSE_PARAPHRASE,
                confidence=0.75,
                similarity_score=0.78,
            ),
        ]

    def test_empty_citations(self):
        """Test ranking with empty list"""
        ranker = CitationRanker(strategy=RankingStrategy.CONFIDENCE)
        result = ranker.rank([])
        assert result == []

    def test_single_citation(self):
        """Test ranking with single citation"""
        ranker = CitationRanker(strategy=RankingStrategy.CONFIDENCE)
        citation = Citation(
            source_id="source-1",
            source_doc_id="doc-1",
            quote_text="Test",
            start_index=0,
            end_index=4,
            citation_type=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.97,
        )

        result = ranker.rank([citation])
        assert len(result) == 1
        assert result[0] == citation

    def test_confidence_ranking(self, sample_citations):
        """Test ranking by confidence score"""
        ranker = CitationRanker(strategy=RankingStrategy.CONFIDENCE)
        result = ranker.rank(sample_citations)

        # Should be sorted by confidence descending
        assert len(result) == 3
        assert result[0].confidence == pytest.approx(0.95)
        assert result[1].confidence == 0.85
        assert result[2].confidence == 0.75

    def test_source_priority_ranking(self, sample_citations):
        """Test ranking by source priority"""
        priorities = {
            "source-1": 0.5,  # Medium priority
            "source-2": 0.9,  # High priority
            "source-3": 0.3,  # Low priority
        }

        ranker = CitationRanker(strategy=RankingStrategy.SOURCE_PRIORITY, source_priorities=priorities)
        result = ranker.rank(sample_citations)

        # Should be sorted by source priority descending
        assert result[0].core_data_id == "source-2"  # Highest priority
        assert result[1].core_data_id == "source-1"
        assert result[2].core_data_id == "source-3"

    def test_source_priority_with_default(self):
        """Test source priority uses default for unknown sources"""
        citations = [
            Citation(
                source_id="known-source",
                source_doc_id="doc-1",
                quote_text="Known",
                start_index=0,
                end_index=5,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.80,
                similarity_score=0.85,
            ),
            Citation(
                source_id="unknown-source",
                source_doc_id="doc-2",
                quote_text="Unknown",
                start_index=0,
                end_index=7,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.85,
                similarity_score=0.88,
            ),
        ]

        ranker = CitationRanker(strategy=RankingStrategy.SOURCE_PRIORITY, source_priorities={"known-source": 0.9})
        result = ranker.rank(citations)

        # Known source with high priority should be first
        assert result[0].core_data_id == "known-source"

    def test_temporal_ranking_preserve_order(self, sample_citations):
        """Test temporal ranking with preserved order"""
        ranker = CitationRanker(strategy=RankingStrategy.TEMPORAL)
        result = ranker.rank(sample_citations, preserve_order=True)

        # Should be reversed (most recent first)
        assert result[0] == sample_citations[2]
        assert result[1] == sample_citations[1]
        assert result[2] == sample_citations[0]

    def test_temporal_ranking_no_preserve(self, sample_citations):
        """Test temporal ranking without preserved order"""
        ranker = CitationRanker(strategy=RankingStrategy.TEMPORAL)
        result = ranker.rank(sample_citations, preserve_order=False)

        # Should fall back to confidence ranking
        assert result[0].confidence == pytest.approx(0.95)

    def test_combined_ranking(self, sample_citations):
        """Test combined ranking strategy"""
        priorities = {
            "source-1": 0.9,
            "source-2": 0.8,
            "source-3": 0.7,
        }

        ranker = CitationRanker(strategy=RankingStrategy.COMBINED, source_priorities=priorities)
        result = ranker.rank(sample_citations, preserve_order=True)

        # Combined should consider all factors
        assert len(result) == 3
        # Since sample_citations are already optimally sorted (descending confidence + priority),
        # combined ranking should maintain this order
        assert result[0].core_data_id == "source-1"  # Highest confidence + priority
        assert result[1].core_data_id == "source-2"  # Medium confidence + priority
        assert result[2].core_data_id == "source-3"  # Lowest confidence + priority

    def test_combined_ranking_default_weights(self, sample_citations):
        """Test combined ranking uses default weights correctly"""
        ranker = CitationRanker(strategy=RankingStrategy.COMBINED)
        result = ranker.rank(sample_citations)

        # Should use default weights (confidence 50%, similarity 30%, etc.)
        assert len(result) == 3

    def test_set_source_priority(self):
        """Test setting individual source priority"""
        ranker = CitationRanker(strategy=RankingStrategy.SOURCE_PRIORITY)

        ranker.set_source_priority("source-1", 0.9)
        priority = ranker.get_source_priority("source-1")

        assert priority == 0.9

    def test_set_source_priority_out_of_range(self):
        """Test source priority clamping"""
        ranker = CitationRanker(strategy=RankingStrategy.SOURCE_PRIORITY)

        # Should clamp to valid range
        ranker.set_source_priority("source-1", 1.5)
        assert ranker.get_source_priority("source-1") == 1.0

        ranker.set_source_priority("source-2", -0.5)
        assert ranker.get_source_priority("source-2") == 0.0

    def test_set_source_priorities_bulk(self):
        """Test setting multiple source priorities"""
        ranker = CitationRanker(strategy=RankingStrategy.SOURCE_PRIORITY)

        priorities = {
            "source-1": 0.9,
            "source-2": 0.7,
            "source-3": 0.5,
        }

        ranker.set_source_priorities(priorities)

        assert ranker.get_source_priority("source-1") == 0.9
        assert ranker.get_source_priority("source-2") == 0.7
        assert ranker.get_source_priority("source-3") == 0.5

    def test_get_source_priority_default(self):
        """Test getting priority for unknown source returns default"""
        ranker = CitationRanker(strategy=RankingStrategy.SOURCE_PRIORITY)

        priority = ranker.get_source_priority("unknown-source")
        assert priority == 0.5  # Default

    def test_update_combined_weights(self):
        """Test updating combined ranking weights"""
        ranker = CitationRanker(strategy=RankingStrategy.COMBINED)

        new_weights = {
            "confidence": 0.60,
            "similarity": 0.25,
            "source_priority": 0.10,
            "recency": 0.05,
        }

        ranker.update_combined_weights(new_weights)

        assert ranker.combined_weights["confidence"] == 0.60
        assert ranker.combined_weights["similarity"] == 0.25

    def test_update_combined_weights_normalization(self):
        """Test combined weights are normalized if they don't sum to 1.0"""
        ranker = CitationRanker(strategy=RankingStrategy.COMBINED)

        # Weights that don't sum to 1.0
        new_weights = {
            "confidence": 0.60,
            "similarity": 0.60,
            "source_priority": 0.60,
            "recency": 0.60,
        }

        ranker.update_combined_weights(new_weights)

        # Should be normalized
        total = sum(ranker.combined_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_metrics_tracking(self, sample_citations):
        """Test metrics are tracked correctly"""
        ranker = CitationRanker(strategy=RankingStrategy.CONFIDENCE)

        ranker.rank(sample_citations)
        ranker.rank(sample_citations)

        metrics = ranker.get_metrics()

        assert metrics["total_ranked"] == 6  # 3 citations x 2 calls
        assert metrics["strategy"] == RankingStrategy.CONFIDENCE.value

    def test_metrics_reset(self, sample_citations):
        """Test metrics can be reset"""
        ranker = CitationRanker(strategy=RankingStrategy.CONFIDENCE)

        ranker.rank(sample_citations)

        metrics_before = ranker.get_metrics()
        assert metrics_before["total_ranked"] > 0

        ranker.reset_metrics()
        metrics_after = ranker.get_metrics()

        assert metrics_after["total_ranked"] == 0

    def test_ranking_stability(self, sample_citations):
        """Test ranking is stable (same input produces same output)"""
        ranker = CitationRanker(strategy=RankingStrategy.CONFIDENCE)

        result1 = ranker.rank(sample_citations)
        result2 = ranker.rank(sample_citations)

        assert [c.core_data_id for c in result1] == [c.core_data_id for c in result2]

    def test_combined_ranking_recency_factor(self):
        """Test combined ranking considers recency"""
        citations = [
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="Older citation",
                start_index=0,
                end_index=14,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.85,
                similarity_score=0.87,
            ),
            Citation(
                source_id="source-2",
                source_doc_id="doc-2",
                quote_text="Newer citation",
                start_index=0,
                end_index=14,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.85,
                similarity_score=0.87,
            ),
        ]

        ranker = CitationRanker(
            strategy=RankingStrategy.COMBINED,
            combined_weights={
                "confidence": 0.25,
                "similarity": 0.25,
                "source_priority": 0.25,
                "recency": 0.25,  # High recency weight
            },
        )

        result = ranker.rank(citations, preserve_order=True)

        # With preserve_order=True and high recency weight, newer should rank higher
        assert result[0].core_data_id == "source-2"

    def test_all_strategies_execute(self, sample_citations):
        """Test all ranking strategies execute without errors"""
        strategies = [
            RankingStrategy.CONFIDENCE,
            RankingStrategy.SOURCE_PRIORITY,
            RankingStrategy.TEMPORAL,
            RankingStrategy.COMBINED,
        ]

        for strategy in strategies:
            ranker = CitationRanker(strategy=strategy)
            result = ranker.rank(sample_citations)
            assert len(result) == len(sample_citations)
