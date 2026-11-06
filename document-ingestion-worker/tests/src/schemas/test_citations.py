import pytest

from document_common.schemas.citations import Citation, CitationLabel, CitationStack, SourceMetadata


class TestCitation:
    def test_valid_citation(self):
        citation = Citation(
            source_id="uuid-123",
            source_doc_id="doc-456",
            quote_text="Hello",
            start_index=0,
            end_index=5,
            citation_type=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.88,
        )
        assert citation.start_index == 0
        assert citation.end_index == 5

    def test_negative_start_index_fails(self):
        with pytest.raises(ValueError):
            Citation(
                source_id="uuid-123",
                source_doc_id="doc-456",
                quote_text="test",
                start_index=-1,
                end_index=4,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.9,
                similarity_score=0.8,
            )

    def test_end_before_start_fails(self):
        with pytest.raises(ValueError, match="end_index.*must be > start_index"):
            Citation(
                source_id="uuid-123",
                source_doc_id="doc-456",
                quote_text="test",
                start_index=10,
                end_index=5,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.9,
                similarity_score=0.8,
            )

    def test_mismatched_quote_length_fails(self):
        with pytest.raises(ValueError, match="quote_text length must match offset span"):
            Citation(
                source_id="uuid-123",
                source_doc_id="doc-456",
                quote_text="short",
                start_index=0,
                end_index=20,
                citation_type=CitationLabel.DIRECT_QUOTE,
                confidence=0.9,
                similarity_score=0.8,
            )


class TestCitationStack:
    def test_valid_stack(self):
        primary = Citation(
            source_id="uuid-1",
            source_doc_id="doc-1",
            quote_text="primary",
            start_index=0,
            end_index=7,
            citation_type=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.9,
        )
        stack = CitationStack(
            span_text="generated text",
            span_start_index=0,
            span_end_index=14,
            primary_citation=primary,
            supporting_citations=[],
            stack_confidence=0.95,
        )
        assert stack.primary_citation == primary
        assert len(stack.supporting_citations) == 0

    def test_stack_size_limit(self):
        primary = Citation(
            source_id="uuid-1",
            source_doc_id="doc-1",
            quote_text="text",
            start_index=0,
            end_index=4,
            citation_type=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.9,
        )
        supporting = [
            Citation(
                source_id=f"uuid-{i}",
                source_doc_id=f"doc-{i}",
                quote_text="text",
                start_index=0,
                end_index=4,
                citation_type=CitationLabel.CLOSE_PARAPHRASE,
                confidence=0.8 - i * 0.05,
                similarity_score=0.7,
            )
            for i in range(10)
        ]
        stack = CitationStack(
            span_text="generated",
            span_start_index=0,
            span_end_index=9,
            primary_citation=primary,
            supporting_citations=supporting,
            stack_confidence=0.9,
        )
        assert len(stack.supporting_citations) == 5


class TestSourceMetadata:
    def test_valid_metadata(self):
        metadata = SourceMetadata(
            id="hash-123",
            file_id="file-uuid-456",
            content="Document content here",
            type="original",
        )
        assert metadata.id == "hash-123"
        assert metadata.type == "original"
        assert metadata.embedding is None

    def test_with_embedding(self):
        metadata = SourceMetadata(
            id="hash-123",
            file_id="file-uuid-456",
            content="Document content",
            type="chunk",
            embedding=[0.1, 0.2, 0.3],
        )
        assert len(metadata.embedding) == 3


class TestSerialization:
    def test_citation_json_roundtrip(self):
        citation = Citation(
            source_id="uuid-123",
            source_doc_id="doc-456",
            quote_text="test",
            start_index=0,
            end_index=4,
            citation_type=CitationLabel.DIRECT_QUOTE,
            confidence=0.9,
            similarity_score=0.85,
        )
        json_data = citation.model_dump()
        restored = Citation(**json_data)
        assert restored == citation

    def test_stack_json_roundtrip(self):
        primary = Citation(
            source_id="uuid-1",
            source_doc_id="doc-1",
            quote_text="text",
            start_index=0,
            end_index=4,
            citation_type=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.9,
        )
        stack = CitationStack(
            span_text="generated",
            span_start_index=0,
            span_end_index=9,
            primary_citation=primary,
            supporting_citations=[],
            stack_confidence=0.95,
        )
        json_data = stack.model_dump()
        restored = CitationStack(**json_data)
        assert restored.span_text == stack.span_text
