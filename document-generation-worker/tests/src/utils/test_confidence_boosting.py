import pytest

from src.schemas.citations import Citation, CitationLabel
from src.utils.confidence_boosting import (
    assess_stack_quality,
    calculate_confidence_with_method,
    calculate_diversity_weighted_boost,
    calculate_progressive_boost,
    calculate_quality_weighted_boost,
    calculate_stack_confidence,
)


class TestStackConfidenceCalculation:
    """Test suite for stack confidence calculation functions"""

    @pytest.fixture
    def primary_citation(self):
        """Create a primary citation with high confidence"""
        return Citation(
            source_id="source-1",
            source_doc_id="doc-1",
            quote_text="Primary quote text",
            start_index=0,
            end_index=18,
            citation_type=CitationLabel.DIRECT_QUOTE,
            confidence=0.90,
            similarity_score=0.95,
        )

    @pytest.fixture
    def supporting_citations(self):
        """Create supporting citations with varying confidence"""
        return [
            Citation(
                source_id="source-2",
                source_doc_id="doc-2",
                quote_text="Supporting quote 1",
                start_index=0,
                end_index=18,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.85,
                similarity_score=0.88,
            ),
            Citation(
                source_id="source-3",
                source_doc_id="doc-3",
                quote_text="Supporting quote 2",
                start_index=0,
                end_index=18,
                citation_type=CitationLabel.LOOSE_PARAPHRASE,
                confidence=0.80,
                similarity_score=0.82,
            ),
        ]

    def test_stack_confidence_no_supporting(self, primary_citation):
        """Test stack confidence with no supporting citations"""
        confidence = calculate_stack_confidence(primary_citation, [], boost_factor=0.05, max_boost=0.25)

        assert confidence == primary_citation.confidence
        assert confidence == 0.90

    def test_stack_confidence_with_supporting(self, primary_citation, supporting_citations):
        """Test stack confidence with supporting citations"""
        confidence = calculate_stack_confidence(
            primary_citation, supporting_citations, boost_factor=0.05, max_boost=0.25
        )

        # Base 0.90 + (2 supporters * 0.05 boost) = 1.00, capped at 1.0
        assert confidence == 1.0

    def test_stack_confidence_single_support(self, primary_citation):
        """Test stack confidence with single supporting citation"""
        supporting = [
            Citation(
                source_id="source-2",
                source_doc_id="doc-2",
                quote_text="Support",
                start_index=0,
                end_index=7,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.85,
                similarity_score=0.87,
            )
        ]

        confidence = calculate_stack_confidence(primary_citation, supporting, boost_factor=0.05)

        # Base 0.90 + (1 * 0.05) = 0.95
        assert confidence == pytest.approx(0.95)

    def test_stack_confidence_quality_threshold(self, primary_citation):
        """Test that low-quality supporters don't boost confidence"""
        low_quality_supporting = [
            Citation(
                source_id="source-2",
                source_doc_id="doc-2",
                quote_text="Low quality",
                start_index=0,
                end_index=11,
                citation_type=CitationLabel.INDEPENDENT,
                confidence=0.50,  # Below default quality threshold
                similarity_score=0.55,
            )
        ]

        confidence = calculate_stack_confidence(
            primary_citation, low_quality_supporting, boost_factor=0.05, quality_threshold=0.7
        )

        # No boost because supporting citation is below quality threshold
        assert confidence == primary_citation.confidence

    def test_stack_confidence_max_boost_cap(self, primary_citation):
        """Test that max boost is enforced"""
        # Create many supporting citations
        many_supporters = [
            Citation(
                source_id=f"source-{i}",
                source_doc_id=f"doc-{i}",
                quote_text=f"Support {i}",
                start_index=0,
                end_index=9,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.85,
                similarity_score=0.87,
            )
            for i in range(10)
        ]

        confidence = calculate_stack_confidence(primary_citation, many_supporters, boost_factor=0.05, max_boost=0.25)

        # Base 0.90 + max_boost 0.25 = 1.15, but capped at 1.0
        assert confidence == 1.0

    def test_progressive_boost(self, primary_citation, supporting_citations):
        """Test progressive boost calculation"""
        confidence = calculate_progressive_boost(
            primary_citation, supporting_citations, base_boost=0.05, max_boost=0.25
        )

        # First: 0.05 * 1.0 = 0.05
        # Second: 0.05 * 0.8 = 0.04
        # Total: 0.09
        # Result: 0.90 + 0.09 = 0.99
        assert 0.98 <= confidence <= 1.0

    def test_progressive_boost_diminishing_returns(self, primary_citation):
        """Test that progressive boost has diminishing returns"""
        supporters = [
            Citation(
                source_id=f"source-{i}",
                source_doc_id=f"doc-{i}",
                quote_text=f"Support {i}",
                start_index=0,
                end_index=9,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.85,
                similarity_score=0.87,
            )
            for i in range(5)
        ]

        confidence = calculate_progressive_boost(primary_citation, supporters, base_boost=0.05, max_boost=0.25)

        # Should hit max boost before all citations contribute equally
        assert confidence <= 1.0

    def test_diversity_weighted_boost(self, primary_citation, supporting_citations):
        """Test diversity-weighted boost with unique sources"""
        confidence = calculate_diversity_weighted_boost(
            primary_citation, supporting_citations, boost_factor=0.05, max_boost=0.25
        )

        # 2 unique sources (different from primary) * 0.05 = 0.10
        # Result: 0.90 + 0.10 = 1.00
        assert confidence == 1.0

    def test_diversity_weighted_boost_duplicate_sources(self, primary_citation):
        """Test diversity boost ignores duplicate sources"""
        duplicate_supporters = [
            Citation(
                source_id="source-2",  # Same source for both
                source_doc_id="doc-2",
                quote_text="Support 1",
                start_index=0,
                end_index=9,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.85,
                similarity_score=0.87,
            ),
            Citation(
                source_id="source-2",  # Duplicate
                source_doc_id="doc-3",
                quote_text="Support 2",
                start_index=10,
                end_index=19,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.80,
                similarity_score=0.82,
            ),
        ]

        confidence = calculate_diversity_weighted_boost(
            primary_citation, duplicate_supporters, boost_factor=0.05, max_boost=0.25
        )

        # Only 1 unique source * 0.05 = 0.05
        # Result: 0.90 + 0.05 = 0.95
        assert confidence == pytest.approx(0.95)

    def test_quality_weighted_boost(self, primary_citation, supporting_citations):
        """Test quality-weighted boost calculation"""
        confidence = calculate_quality_weighted_boost(
            primary_citation, supporting_citations, boost_factor=0.05, max_boost=0.25
        )

        # First: 0.05 * 0.85 = 0.0425
        # Second: 0.05 * 0.80 = 0.04
        # Total: 0.0825
        # Result: 0.90 + 0.0825 = 0.9825
        assert 0.98 <= confidence <= 0.99

    def test_quality_weighted_boost_high_confidence(self, primary_citation):
        """Test quality boost with high confidence supporters"""
        high_quality_supporters = [
            Citation(
                source_id="source-2",
                source_doc_id="doc-2",
                quote_text="High quality",
                start_index=0,
                end_index=12,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.95,
                similarity_score=0.97,
            )
        ]

        confidence = calculate_quality_weighted_boost(
            primary_citation, high_quality_supporters, boost_factor=0.05, max_boost=0.25
        )

        # 0.05 * 0.95 = 0.0475
        # Result: 0.90 + 0.0475 = 0.9475
        assert 0.94 <= confidence <= 0.95


class TestStackQualityAssessment:
    """Test suite for stack quality assessment"""

    @pytest.fixture
    def primary_citation(self):
        """Create a primary citation"""
        return Citation(
            source_id="source-1",
            source_doc_id="doc-1",
            quote_text="Primary",
            start_index=0,
            end_index=7,
            citation_type=CitationLabel.DIRECT_QUOTE,
            confidence=0.90,
            similarity_score=0.95,
        )

    def test_quality_assessment_high_quality_stack(self, primary_citation):
        """Test quality assessment for high-quality stack"""
        supporting = [
            Citation(
                source_id="source-2",
                source_doc_id="doc-2",
                quote_text="Support 1",
                start_index=0,
                end_index=9,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.85,
                similarity_score=0.87,
            ),
            Citation(
                source_id="source-3",
                source_doc_id="doc-3",
                quote_text="Support 2",
                start_index=0,
                end_index=9,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.80,
                similarity_score=0.82,
            ),
        ]

        assessment = assess_stack_quality(primary_citation, supporting, min_confidence=0.5, diversity_threshold=2)

        assert assessment["num_sources"] == 3
        assert assessment["num_unique_sources"] == 3
        assert assessment["num_supporting"] == 2
        assert assessment["primary_confidence"] == 0.90
        assert assessment["has_low_confidence"] is False
        assert assessment["lacks_diversity"] is False
        assert assessment["weak_support"] is False
        assert assessment["is_high_quality"] is True
        assert 0.85 <= assessment["avg_confidence"] < 0.90
        assert assessment["quality_score"] > 0.75

    def test_quality_assessment_low_confidence(self):
        """Test quality assessment with low confidence primary"""
        low_confidence_primary = Citation(
            source_id="source-1",
            source_doc_id="doc-1",
            quote_text="Low conf",
            start_index=0,
            end_index=8,
            citation_type=CitationLabel.LOOSE_PARAPHRASE,
            confidence=0.40,
            similarity_score=0.50,
        )

        assessment = assess_stack_quality(low_confidence_primary, [], min_confidence=0.5)

        assert assessment["has_low_confidence"] is True
        assert assessment["weak_support"] is True
        assert assessment["is_high_quality"] is False

    def test_quality_assessment_lacks_diversity(self, primary_citation):
        """Test quality assessment with poor source diversity"""
        same_source_supporting = [
            Citation(
                source_id="source-1",  # Same as primary
                source_doc_id="doc-2",
                quote_text="Support",
                start_index=10,
                end_index=17,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.85,
                similarity_score=0.87,
            )
        ]

        assessment = assess_stack_quality(primary_citation, same_source_supporting, diversity_threshold=2)

        assert assessment["num_unique_sources"] == 1
        assert assessment["lacks_diversity"] is True

    def test_quality_assessment_weak_support(self, primary_citation):
        """Test quality assessment with no supporting citations"""
        assessment = assess_stack_quality(primary_citation, [])

        assert assessment["weak_support"] is True
        assert assessment["num_supporting"] == 0


class TestConfidenceWithMethod:
    """Test suite for method-based confidence calculation"""

    @pytest.fixture
    def primary_citation(self):
        """Create a primary citation"""
        return Citation(
            source_id="source-1",
            source_doc_id="doc-1",
            quote_text="Primary",
            start_index=0,
            end_index=7,
            citation_type=CitationLabel.DIRECT_QUOTE,
            confidence=0.90,
            similarity_score=0.95,
        )

    @pytest.fixture
    def supporting_citations(self):
        """Create supporting citations"""
        return [
            Citation(
                source_id="source-2",
                source_doc_id="doc-2",
                quote_text="Support 1",
                start_index=0,
                end_index=9,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.85,
                similarity_score=0.87,
            )
        ]

    def test_method_standard(self, primary_citation, supporting_citations):
        """Test standard method selection"""
        confidence = calculate_confidence_with_method(
            primary_citation, supporting_citations, method="standard", boost_factor=0.05
        )

        # 0.90 + 0.05 = 0.95
        assert confidence == pytest.approx(0.95)

    def test_method_progressive(self, primary_citation, supporting_citations):
        """Test progressive method selection"""
        confidence = calculate_confidence_with_method(
            primary_citation, supporting_citations, method="progressive", boost_factor=0.05
        )

        assert 0.94 <= confidence <= 0.96

    def test_method_diversity(self, primary_citation, supporting_citations):
        """Test diversity method selection"""
        confidence = calculate_confidence_with_method(
            primary_citation, supporting_citations, method="diversity", boost_factor=0.05
        )

        assert confidence == pytest.approx(0.95)

    def test_method_quality(self, primary_citation, supporting_citations):
        """Test quality method selection"""
        confidence = calculate_confidence_with_method(
            primary_citation, supporting_citations, method="quality", boost_factor=0.05
        )

        # 0.90 + (0.05 * 0.85) = 0.9425
        assert 0.94 <= confidence <= 0.95

    def test_method_unknown_defaults_to_standard(self, primary_citation, supporting_citations):
        """Test unknown method defaults to standard"""
        confidence = calculate_confidence_with_method(
            primary_citation, supporting_citations, method="unknown_method", boost_factor=0.05
        )

        # Should use standard method
        assert confidence == pytest.approx(0.95)

    def test_method_case_insensitive(self, primary_citation, supporting_citations):
        """Test method names are case insensitive"""
        confidence_upper = calculate_confidence_with_method(
            primary_citation, supporting_citations, method="PROGRESSIVE", boost_factor=0.05
        )

        confidence_lower = calculate_confidence_with_method(
            primary_citation, supporting_citations, method="progressive", boost_factor=0.05
        )

        assert confidence_upper == confidence_lower
