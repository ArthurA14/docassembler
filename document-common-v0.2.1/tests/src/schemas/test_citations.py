import pytest
from pydantic import ValidationError

from document_common.schemas.citations import (
    Citation,
    CitationLabel,
    CitationStack,
    ClassificationResult,
    PipelineMetrics,
    SourceMetadata,
    SourceReference,
    SourceReferences,
    SupportingCitation,
    CitationPipelineResult,
)


class TestCitationLabel:
    def test_all_labels_exist(self):
        assert CitationLabel.DIRECT_QUOTE == "direct_quote"
        assert CitationLabel.CLOSE_PARAPHRASE == "close_paraphrase"
        assert CitationLabel.LOOSE_PARAPHRASE == "loose_paraphrase"
        assert CitationLabel.INDEPENDENT == "independent"

    def test_iteration(self):
        labels = list(CitationLabel)
        assert len(labels) == 4


class TestCitation:
    def test_valid_citation(self):
        c = Citation(
            span_text="hello world",
            span_start=0,
            span_end=11,
            file_id="uuid-1234",
            chunk_id="doc-uuid-5678",
            citation_label=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.88,
        )
        assert c.span_text == "hello world"
        assert c.span_start == 0
        assert c.span_end == 11
        assert c.citation_label == CitationLabel.DIRECT_QUOTE
        assert c.confidence == 0.95

    def test_span_length_mismatch_raises(self):
        with pytest.raises(ValidationError) as exc:
            Citation(
                span_text="hello",
                span_start=0,
                span_end=10,
                file_id="uuid-1234",
                chunk_id="chunk-1234",
                citation_label=CitationLabel.DIRECT_QUOTE,
                confidence=0.95,
                similarity_score=0.88,
            )
        assert "must match offset span length" in str(exc.value)

    def test_span_end_before_start_raises(self):
        with pytest.raises(ValidationError) as exc:
            Citation(
                span_text="hello",
                span_start=10,
                span_end=5,
                file_id="uuid-1234",
                chunk_id="chunk-1234",
                citation_label=CitationLabel.DIRECT_QUOTE,
                confidence=0.95,
                similarity_score=0.88,
            )
        assert "must be >" in str(exc.value)

    def test_confidence_out_of_range_low(self):
        with pytest.raises(ValidationError):
            Citation(
                span_text="hello",
                span_start=0,
                span_end=5,
                file_id="uuid-1234",
                chunk_id="chunk-1234",
                citation_label=CitationLabel.DIRECT_QUOTE,
                confidence=-0.1,
                similarity_score=0.88,
            )

    def test_confidence_out_of_range_high(self):
        with pytest.raises(ValidationError):
            Citation(
                span_text="hello",
                span_start=0,
                span_end=5,
                file_id="uuid-1234",
                chunk_id="chunk-1234",
                citation_label=CitationLabel.DIRECT_QUOTE,
                confidence=1.1,
                similarity_score=0.88,
            )

    def test_similarity_score_out_of_range(self):
        with pytest.raises(ValidationError):
            Citation(
                span_text="hello",
                span_start=0,
                span_end=5,
                file_id="uuid-1234",
                chunk_id="chunk-1234",
                citation_label=CitationLabel.DIRECT_QUOTE,
                confidence=0.5,
                similarity_score=1.5,
            )

    def test_boundary_confidence_zero(self):
        c = Citation(
            span_text="hello",
            span_start=0,
            span_end=5,
            file_id="uuid-1234",
            chunk_id="chunk-1234",
            citation_label=CitationLabel.DIRECT_QUOTE,
            confidence=0.0,
            similarity_score=0.0,
        )
        assert c.confidence == 0.0

    def test_boundary_confidence_one(self):
        c = Citation(
            span_text="hello",
            span_start=0,
            span_end=5,
            file_id="uuid-1234",
            chunk_id="chunk-1234",
            citation_label=CitationLabel.DIRECT_QUOTE,
            confidence=1.0,
            similarity_score=1.0,
        )
        assert c.confidence == 1.0

    def test_model_dump(self):
        c = Citation(
            span_text="hello",
            span_start=0,
            span_end=5,
            file_id="uuid-1234",
            chunk_id="chunk-1234",
            citation_label=CitationLabel.DIRECT_QUOTE,
            confidence=0.9,
            similarity_score=0.8,
        )
        # span_text has exclude=True, so verify via model attribute
        assert c.span_text == "hello"
        d = c.model_dump()
        assert d["citation_label"] == "direct_quote"
        assert d["file_id"] == "uuid-1234"
        assert d["chunk_id"] == "chunk-1234"


class TestSupportingCitation:
    def test_valid_supporting_citation(self):
        sc = SupportingCitation(
            chunk_id="sub-doc-uuid",
            file_id="core-uuid",
            confidence=0.85,
            citation_label=CitationLabel.CLOSE_PARAPHRASE,
            similarity_score=0.80,
        )
        assert sc.confidence == 0.85
        assert sc.citation_label == CitationLabel.CLOSE_PARAPHRASE

    def test_supporting_citation_confidence_bounds(self):
        with pytest.raises(ValidationError):
            SupportingCitation(
                chunk_id="sub-doc-uuid",
                file_id="core-uuid",
                confidence=1.5,
                citation_label=CitationLabel.DIRECT_QUOTE,
                similarity_score=0.80,
            )


class TestCitationStack:
    def test_valid_citation_stack(self):
        primary = Citation(
            span_text="hello world",
            span_start=0,
            span_end=11,
            file_id="uuid-1234",
            chunk_id="chunk-1234",
            citation_label=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.88,
        )
        stack = CitationStack(
            span_text="hello world",
            span_start=0,
            span_end=11,
            primary_citation=primary,
            stack_confidence=0.95,
        )
        assert len(stack.supporting_citations) == 0

    def test_citation_stack_with_supporting(self):
        primary = Citation(
            span_text="hello",
            span_start=0,
            span_end=5,
            file_id="uuid-1234",
            chunk_id="chunk-1234",
            citation_label=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.88,
        )
        supporting = SupportingCitation(
            chunk_id="sub-doc-uuid-2",
            file_id="core-uuid-2",
            confidence=0.80,
            citation_label=CitationLabel.LOOSE_PARAPHRASE,
            similarity_score=0.75,
        )
        stack = CitationStack(
            span_text="hello",
            span_start=0,
            span_end=5,
            primary_citation=primary,
            supporting_citations=[supporting],
            stack_confidence=0.90,
        )
        assert len(stack.supporting_citations) == 1
        assert stack.supporting_citations[0].confidence == 0.80

    def test_limit_stack_size_to_five(self):
        primary = Citation(
            span_text="hello",
            span_start=0,
            span_end=5,
            file_id="uuid-1234",
            chunk_id="chunk-1234",
            citation_label=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.88,
        )
        supporting = [
            SupportingCitation(
                chunk_id=f"sub-{i}",
                file_id=f"core-{i}",
                confidence=0.5 + i * 0.05,
                citation_label=CitationLabel.DIRECT_QUOTE,
                similarity_score=0.5 + i * 0.05,
            )
            for i in range(10)
        ]
        stack = CitationStack(
            span_text="hello",
            span_start=0,
            span_end=5,
            primary_citation=primary,
            supporting_citations=supporting,
            stack_confidence=0.90,
        )
        assert len(stack.supporting_citations) <= 5
        # Should be sorted by confidence descending
        for i in range(len(stack.supporting_citations) - 1):
            assert stack.supporting_citations[i].confidence >= stack.supporting_citations[i + 1].confidence

    def test_span_text_mismatch_in_stack(self):
        primary = Citation(
            span_text="hello",
            span_start=0,
            span_end=5,
            file_id="uuid-1234",
            chunk_id="chunk-1234",
            citation_label=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.88,
        )
        with pytest.raises(ValidationError):
            CitationStack(
                span_text="hello world",
                span_start=0,
                span_end=5,
                primary_citation=primary,
                stack_confidence=0.90,
            )


class TestSourceMetadata:
    def test_valid_source_metadata(self):
        sm = SourceMetadata(
            chunk_id="doc-hash-123",
            file_id="file-uuid-456",
            content="This is the document content.",
        )
        assert sm.type == "original"
        assert sm.content == "This is the document content."

    def test_source_metadata_with_embedding(self):
        sm = SourceMetadata(
            chunk_id="doc-hash-123",
            file_id="file-uuid-456",
            content="Content here",
            embedding=[0.1, 0.2, 0.3],
        )
        assert sm.embedding == [0.1, 0.2, 0.3]


class TestSourceReference:
    def test_valid_source_reference(self):
        sr = SourceReference(
            chunk_id="sub-doc-uuid",
            file_id="core-uuid",
            content="Full content of source document",
        )
        assert sr.chunk_id == "sub-doc-uuid"


class TestSourceReferences:
    def test_empty_source_references(self):
        sr = SourceReferences(references={})
        assert sr.references == {}

    def test_add_source(self):
        sr = SourceReferences(references={})
        meta = SourceMetadata(chunk_id="hash-1", file_id="file-1", content="content")
        sr.add_source("sub-1", meta)
        assert sr.has_source("sub-1")
        assert sr.get_source("sub-1") == meta

    def test_get_source_not_found(self):
        sr = SourceReferences(references={})
        assert sr.get_source("nonexistent") is None

    def test_has_source_false(self):
        sr = SourceReferences(references={"sub-1": SourceMetadata(chunk_id="1", file_id="f1", content="c")})
        assert sr.has_source("sub-2") is False

    def test_update_source(self):
        sr = SourceReferences(references={})
        meta1 = SourceMetadata(chunk_id="hash-1", file_id="file-1", content="content1")
        meta2 = SourceMetadata(chunk_id="hash-2", file_id="file-2", content="content2")
        sr.add_source("sub-1", meta1)
        sr.add_source("sub-1", meta2)
        assert sr.get_source("sub-1").chunk_id == "hash-2"


class TestClassificationResult:
    def test_valid_classification_result(self):
        cr = ClassificationResult(
            reasoning="This is a direct quote from the source.",
            classification=CitationLabel.DIRECT_QUOTE,
            confidence=0.92,
        )
        assert cr.classification == CitationLabel.DIRECT_QUOTE
        assert cr.confidence == 0.92

    def test_classification_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ClassificationResult(
                reasoning="test",
                classification=CitationLabel.INDEPENDENT,
                confidence=1.5,
            )


class TestPipelineMetrics:
    def test_default_metrics(self):
        pm = PipelineMetrics(
            request_id="req-123",
            total_duration_seconds=5.2,
        )
        assert pm.request_id == "req-123"
        assert pm.sentences_processed == 0
        assert pm.citations_created == 0
        assert pm.stacks_created == 0
        assert pm.component_durations == {}

    def test_full_metrics(self):
        pm = PipelineMetrics(
            request_id="req-456",
            total_duration_seconds=10.5,
            component_durations={"chunking": 2.0, "retrieval": 3.0, "assembly": 5.5},
            sentences_processed=50,
            citations_created=10,
            stacks_created=8,
            validation_errors=0,
            validation_warnings=2,
        )
        assert pm.sentences_processed == 50
        assert pm.citations_created == 10
        assert pm.stacks_created == 8
        assert pm.validation_warnings == 2

    def test_duration_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            PipelineMetrics(
                request_id="req-789",
                total_duration_seconds=-1.0,
            )


class TestCitationPipelineResult:
    def test_empty_result(self):
        source_refs = SourceReferences(references={})
        metrics = PipelineMetrics(request_id="req-1", total_duration_seconds=1.0)
        result = CitationPipelineResult(
            citation_stacks=[],
            invalid_citations=[],
            source_references=source_refs,
            metrics=metrics,
        )
        assert result.citation_stacks == []
        assert result.invalid_citations == []
        assert result.errors == []
        assert result.warnings == []

    def test_result_with_data(self):
        primary = Citation(
            span_text="test",
            span_start=0,
            span_end=4,
            file_id="uuid-1",
            chunk_id="chunk-1",
            citation_label=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            similarity_score=0.90,
        )
        stack = CitationStack(
            span_text="test",
            span_start=0,
            span_end=4,
            primary_citation=primary,
            stack_confidence=0.95,
        )
        source_refs = SourceReferences(references={})
        metrics = PipelineMetrics(request_id="req-1", total_duration_seconds=2.0)
        result = CitationPipelineResult(
            citation_stacks=[stack],
            invalid_citations=[],
            source_references=source_refs,
            metrics=metrics,
            validation_document={"detail": "all good"},
            errors=[],
            warnings=["minor warning"],
        )
        assert len(result.citation_stacks) == 1
        assert result.warnings == ["minor warning"]
