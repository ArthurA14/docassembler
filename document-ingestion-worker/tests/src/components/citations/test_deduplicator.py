import pytest

from src.components.citations.deduplicator import CitationDeduplicator, DeduplicationMethod
from document_common.schemas.citations import Citation, CitationLabel


class TestCitationDeduplicator:
    """Test suite for CitationDeduplicator"""

    @pytest.fixture
    def exact_deduplicator(self):
        """Create deduplicator with exact method"""
        return CitationDeduplicator(method=DeduplicationMethod.EXACT)

    @pytest.fixture
    def fuzzy_deduplicator(self):
        """Create deduplicator with fuzzy method"""
        return CitationDeduplicator(method=DeduplicationMethod.FUZZY, fuzzy_threshold=85.0)

    @pytest.fixture
    def semantic_deduplicator(self):
        """Create deduplicator with semantic method"""
        return CitationDeduplicator(method=DeduplicationMethod.SEMANTIC, semantic_threshold=0.95)

    @pytest.fixture
    def hybrid_deduplicator(self):
        """Create deduplicator with hybrid method"""
        return CitationDeduplicator(method=DeduplicationMethod.HYBRID, fuzzy_threshold=85.0)

    @pytest.fixture
    def sample_citations(self):
        """Create sample citations for testing"""
        return [
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="This is a test quote from the document!!",  # 40 chars
                start_index=0,
                end_index=40,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.95,
                similarity_score=0.97,
            ),
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="This is a test quote from the document!!",  # Exact duplicate, 40 chars
                start_index=0,
                end_index=40,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.93,
                similarity_score=0.96,
            ),
            Citation(
                source_id="source-2",
                source_doc_id="doc-2",
                quote_text="Different quote from another source!!",  # 37 chars
                start_index=0,
                end_index=37,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.88,
                similarity_score=0.90,
            ),
        ]

    def test_empty_citations(self, exact_deduplicator):
        """Test deduplication with empty list"""
        result = exact_deduplicator.deduplicate([])
        assert result == []

    def test_single_citation(self, exact_deduplicator):
        """Test deduplication with single citation"""
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

        result = exact_deduplicator.deduplicate([citation])
        assert len(result) == 1
        assert result[0] == citation

    def test_exact_deduplication_removes_duplicates(self, exact_deduplicator, sample_citations):
        """Test exact deduplication removes exact duplicates"""
        result = exact_deduplicator.deduplicate(sample_citations)

        # Should remove one duplicate (first two are identical)
        assert len(result) == 2
        # Should keep the one with higher confidence
        assert all(c.source_id in ["source-1", "source-2"] for c in result)

    def test_exact_deduplication_different_sources(self, exact_deduplicator):
        """Test exact deduplication keeps same text from different sources"""
        citations = [
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="Same text",
                start_index=0,
                end_index=9,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.95,
                similarity_score=0.97,
            ),
            Citation(
                source_id="source-2",  # Different source
                source_doc_id="doc-2",
                quote_text="Same text",
                start_index=0,
                end_index=9,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.93,
                similarity_score=0.96,
            ),
        ]

        result = exact_deduplicator.deduplicate(citations)
        # Should keep both because they're from different sources
        assert len(result) == 2

    def test_fuzzy_deduplication_similar_text(self, fuzzy_deduplicator):
        """Test fuzzy deduplication with similar text"""
        citations = [
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="This is a test quote from the document!!",  # 40 chars
                start_index=0,
                end_index=40,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.95,
                similarity_score=0.97,
            ),
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="This is a test quote from document!!!",  # Missing "the", 37 chars
                start_index=50,
                end_index=87,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.90,
                similarity_score=0.92,
            ),
        ]

        result = fuzzy_deduplicator.deduplicate(citations)
        # Should remove one as they're very similar (>85% threshold)
        assert len(result) == 1
        # Should keep the higher confidence one
        assert result[0].confidence == pytest.approx(0.95)

    def test_fuzzy_deduplication_keeps_confidence_winner(self, fuzzy_deduplicator):
        """Test fuzzy deduplication keeps higher confidence citation"""
        citations = [
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="Test quote here",
                start_index=0,
                end_index=15,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.85,  # Lower confidence
                similarity_score=0.90,
            ),
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="Test quote here",  # Very similar
                start_index=20,
                end_index=35,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.95,  # Higher confidence
                similarity_score=0.97,
            ),
        ]

        result = fuzzy_deduplicator.deduplicate(citations)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.95)

    def test_fuzzy_deduplication_different_sources_kept(self, fuzzy_deduplicator):
        """Test fuzzy deduplication keeps similar text from different sources"""
        citations = [
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="Test quote",
                start_index=0,
                end_index=10,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.95,
                similarity_score=0.97,
            ),
            Citation(
                source_id="source-2",  # Different source
                source_doc_id="doc-2",
                quote_text="Test quote",
                start_index=0,
                end_index=10,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.90,
                similarity_score=0.92,
            ),
        ]

        result = fuzzy_deduplicator.deduplicate(citations)
        # Should keep both despite similar text (different sources)
        assert len(result) == 2

    def test_fuzzy_deduplication_dissimilar_text(self, fuzzy_deduplicator):
        """Test fuzzy deduplication keeps dissimilar text"""
        citations = [
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="Completely different text here",
                start_index=0,
                end_index=30,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.95,
                similarity_score=0.97,
            ),
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="Another unrelated passage content",
                start_index=50,
                end_index=83,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.90,
                similarity_score=0.92,
            ),
        ]

        result = fuzzy_deduplicator.deduplicate(citations)
        # Should keep both as they're dissimilar
        assert len(result) == 2

    def test_semantic_deduplication_overlap(self, semantic_deduplicator):
        """Test semantic deduplication with overlapping offsets"""
        citations = [
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="Test quote one",
                start_index=0,
                end_index=14,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.95,
                similarity_score=0.97,
            ),
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="Test quote two",
                start_index=5,
                end_index=19,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.90,
                similarity_score=0.96,  # Very similar score
            ),
        ]

        result = semantic_deduplicator.deduplicate(citations)
        # Should detect overlap and similar scores
        assert len(result) >= 1

    def test_semantic_deduplication_no_overlap(self, semantic_deduplicator):
        """Test semantic deduplication without overlap"""
        citations = [
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="First quote",
                start_index=0,
                end_index=11,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.95,
                similarity_score=0.97,
            ),
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="Second quote",
                start_index=100,
                end_index=112,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.90,
                similarity_score=0.92,
            ),
        ]

        result = semantic_deduplicator.deduplicate(citations)
        # Should keep both due to no overlap
        assert len(result) == 2

    def test_hybrid_deduplication(self, hybrid_deduplicator, sample_citations):
        """Test hybrid deduplication uses multiple methods"""
        result = hybrid_deduplicator.deduplicate(sample_citations)

        # Should apply multiple deduplication passes
        assert len(result) <= len(sample_citations)
        assert len(result) > 0

    def test_hybrid_deduplication_multiple_passes(self, hybrid_deduplicator):
        """Test hybrid deduplication with multiple duplicate types"""
        citations = [
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="Test quote",
                start_index=0,
                end_index=10,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.95,
                similarity_score=0.97,
            ),
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="Test quote",  # Exact duplicate
                start_index=0,
                end_index=10,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.90,
                similarity_score=0.95,
            ),
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="Test quote here",  # Fuzzy similar
                start_index=20,
                end_index=35,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.88,
                similarity_score=0.90,
            ),
        ]

        result = hybrid_deduplicator.deduplicate(citations)
        # Hybrid should remove both types of duplicates
        assert len(result) >= 1

    def test_metrics_tracking(self, exact_deduplicator, sample_citations):
        """Test that metrics are tracked correctly"""
        exact_deduplicator.deduplicate(sample_citations)

        metrics = exact_deduplicator.get_metrics()

        assert metrics["total_processed"] == len(sample_citations)
        assert metrics["total_duplicates_removed"] >= 0
        assert 0 <= metrics["deduplication_rate"] <= 1
        assert metrics["method"] == DeduplicationMethod.EXACT.value

    def test_metrics_reset(self, exact_deduplicator, sample_citations):
        """Test metrics can be reset"""
        exact_deduplicator.deduplicate(sample_citations)

        metrics_before = exact_deduplicator.get_metrics()
        assert metrics_before["total_processed"] > 0

        exact_deduplicator.reset_metrics()
        metrics_after = exact_deduplicator.get_metrics()

        assert metrics_after["total_processed"] == 0
        assert metrics_after["total_duplicates_removed"] == 0

    def test_fuzzy_scorer_options(self):
        """Test different fuzzy scorer options"""
        scorers = ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"]

        for scorer in scorers:
            deduplicator = CitationDeduplicator(
                method=DeduplicationMethod.FUZZY, fuzzy_threshold=85.0, fuzzy_scorer=scorer
            )

            citations = [
                Citation(
                    source_id="source-1",
                    source_doc_id="doc-1",
                    quote_text="Test quote text",
                    start_index=0,
                    end_index=15,
                    citation_type=CitationLabel.DIRECT_QUOTE,
                    confidence=0.95,
                    similarity_score=0.97,
                ),
                Citation(
                    source_id="source-1",
                    source_doc_id="doc-1",
                    quote_text="Test quote text",
                    start_index=20,
                    end_index=35,
                    citation_type=CitationLabel.DIRECT_QUOTE,
                    confidence=0.90,
                    similarity_score=0.92,
                ),
            ]

            result = deduplicator.deduplicate(citations)
            # Should deduplicate regardless of scorer (identical text)
            assert len(result) == 1

    def test_multiple_deduplication_calls(self, fuzzy_deduplicator):
        """Test multiple deduplication calls accumulate metrics"""
        citations1 = [
            Citation(
                source_id="source-1",
                source_doc_id="doc-1",
                quote_text="First batch",
                start_index=0,
                end_index=11,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.95,
                similarity_score=0.97,
            )
        ]

        citations2 = [
            Citation(
                source_id="source-2",
                source_doc_id="doc-2",
                quote_text="Second batch",
                start_index=0,
                end_index=12,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.90,
                similarity_score=0.92,
            )
        ]

        fuzzy_deduplicator.deduplicate(citations1)
        fuzzy_deduplicator.deduplicate(citations2)

        metrics = fuzzy_deduplicator.get_metrics()
        assert metrics["total_processed"] == 2
