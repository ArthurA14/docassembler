"""Tests for citation aggregation logic with per-section sources extraction."""

from api_gateway.utils.citation_aggregator import (
    _convert_to_source_reference,
    extract_sources_for_citations,
)
from src.schemas.citations import (
    Citation,
    CitationLabel,
    CitationStack,
    SourceMetadata,
    SourceReference,
)


class TestExtractSourcesForCitations:
    """Tests for extract_sources_for_citations function."""

    def test_extract_single_citation_source(self):
        """Test extracting sources from single citation."""
        # Create a citation
        citation = Citation(
            generated_span_text="cited text",
            generated_span_start=0,
            generated_span_end=10,
            source_id="source-uuid-1",
            source_doc_id="doc-1",
            citation_type=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.92,
        )

        citation_stack = CitationStack(
            span_text="cited text",
            span_start_index=0,
            span_end_index=10,
            primary_citation=citation,
            supporting_citations=[],
            stack_confidence=0.95,
        )

        # Create corresponding source metadata
        source_metadata = SourceMetadata(
            chunk_id="source-uuid-1",
            file_id="file-uuid-1",
            content="Full source content here",
            type="original",
        )

        # Extract sources
        sources = extract_sources_for_citations(
            [citation_stack],
            {"source-uuid-1": source_metadata},
        )

        # Verify
        assert len(sources) == 1
        assert "source-uuid-1" in sources
        assert sources["source-uuid-1"].source_id == "source-uuid-1"
        assert sources["source-uuid-1"].file_id == "file-uuid-1"

    def test_extract_multiple_citations_deduplicates(self):
        """Test that multiple citations referencing same source are deduplicated."""
        # Create two citations referencing same source
        citation1 = Citation(
            generated_span_text="text1",
            generated_span_start=0,
            generated_span_end=5,
            source_id="source-uuid-1",
            source_doc_id="doc-1",
            citation_type=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.92,
        )

        citation2 = Citation(
            generated_span_text="text2",
            generated_span_start=10,
            generated_span_end=15,
            source_id="source-uuid-1",
            source_doc_id="doc-1",
            citation_type=CitationLabel.LOOSE_PARAPHRASE,
            confidence=0.85,
            similarity_score=0.80,
        )

        stack1 = CitationStack(
            span_text="text1",
            span_start_index=0,
            span_end_index=5,
            primary_citation=citation1,
            supporting_citations=[],
            stack_confidence=0.95,
        )

        stack2 = CitationStack(
            span_text="text2",
            span_start_index=10,
            span_end_index=15,
            primary_citation=citation2,
            supporting_citations=[],
            stack_confidence=0.85,
        )

        source_metadata = SourceMetadata(
            chunk_id="source-uuid-1",
            file_id="file-uuid-1",
            content="Full source content",
            type="original",
        )

        sources = extract_sources_for_citations(
            [stack1, stack2],
            {"source-uuid-1": source_metadata},
        )

        # Should only have one entry despite two citations
        assert len(sources) == 1
        assert "source-uuid-1" in sources

    def test_extract_primary_and_supporting_citations(self):
        """Test extracting sources from primary and supporting citations."""
        primary = Citation(
            generated_span_text="primary",
            generated_span_start=0,
            generated_span_end=7,
            source_id="source-1",
            source_doc_id="doc-1",
            citation_type=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.92,
        )

        supporting1 = Citation(
            generated_span_text="supporting1",
            generated_span_start=0,
            generated_span_end=11,
            source_id="source-2",
            source_doc_id="doc-2",
            citation_type=CitationLabel.CLOSE_PARAPHRASE,
            confidence=0.85,
            similarity_score=0.80,
        )

        supporting2 = Citation(
            generated_span_text="supporting2",
            generated_span_start=0,
            generated_span_end=11,
            source_id="source-3",
            source_doc_id="doc-3",
            citation_type=CitationLabel.LOOSE_PARAPHRASE,
            confidence=0.75,
            similarity_score=0.70,
        )

        stack = CitationStack(
            span_text="cited text",
            span_start_index=0,
            span_end_index=10,
            primary_citation=primary,
            supporting_citations=[supporting1, supporting2],
            stack_confidence=0.88,
        )

        source_metadata = {
            "source-1": SourceMetadata(chunk_id="source-1", file_id="file-1", content="Content 1", type="original"),
            "source-2": SourceMetadata(chunk_id="source-2", file_id="file-2", content="Content 2", type="chunk"),
            "source-3": SourceMetadata(chunk_id="source-3", file_id="file-3", content="Content 3", type="original"),
        }

        sources = extract_sources_for_citations([stack], source_metadata)

        # Should have all three sources
        assert len(sources) == 3
        assert set(sources.keys()) == {"source-1", "source-2", "source-3"}

    def test_extract_skips_missing_source_metadata(self):
        """Test that citations referencing missing source metadata are skipped."""
        citation = Citation(
            generated_span_text="text",
            generated_span_start=0,
            generated_span_end=4,
            source_id="missing-source",
            source_doc_id="doc-1",
            citation_type=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.92,
        )

        stack = CitationStack(
            span_text="text",
            span_start_index=0,
            span_end_index=4,
            primary_citation=citation,
            supporting_citations=[],
            stack_confidence=0.95,
        )

        # Empty source metadata dict
        sources = extract_sources_for_citations([stack], {})

        # Should be empty since source is missing
        assert len(sources) == 0

    def test_extract_empty_citations_list(self):
        """Test with empty citations list."""
        sources = extract_sources_for_citations([], {})
        assert sources == {}


class TestConvertToSourceReference:
    """Tests for _convert_to_source_reference helper function."""

    def test_convert_basic_source_metadata(self):
        """Test converting SourceMetadata to SourceReference."""
        source_metadata = SourceMetadata(
            chunk_id="source-1",
            file_id="file-uuid-1",
            content="Full source content",
            type="original",
        )

        source_ref = _convert_to_source_reference("source-1", source_metadata)

        assert isinstance(source_ref, SourceReference)
        assert source_ref.source_id == "source-1"
        assert source_ref.file_id == "file-uuid-1"
        assert source_ref.content == "Full source content"
        # core_data_id, core_type, core_data_source should have default empty strings
        assert source_ref.core_data_id == ""
        assert source_ref.core_type == ""
        assert source_ref.core_data_source == ""

    def test_convert_with_embedding(self):
        """Test converting SourceMetadata with embedding field."""
        source_metadata = SourceMetadata(
            chunk_id="source-1",
            file_id="file-uuid-1",
            content="Content",
            type="chunk",
            embedding=[0.1, 0.2, 0.3],
        )

        source_ref = _convert_to_source_reference("source-1", source_metadata)

        # Embedding is not transferred to SourceReference, only explicitly pulled fields
        assert source_ref.source_id == "source-1"
        assert source_ref.file_id == "file-uuid-1"

    def test_convert_preserves_all_fields(self):
        """Test that all fields are preserved during conversion."""
        source_metadata = SourceMetadata(
            chunk_id="source-id",
            file_id="file-id",
            content="Content here",
            type="original",
        )

        source_ref = _convert_to_source_reference("source-id", source_metadata)

        assert source_ref.source_id == "source-id"
        assert source_ref.file_id == "file-id"
        assert source_ref.content == "Content here"
