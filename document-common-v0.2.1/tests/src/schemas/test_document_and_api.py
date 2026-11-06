import pytest
from pydantic import ValidationError

from document_common.schemas.document import Section, SectionStatus, Document, DocumentStatus
from document_common.schemas.api import DocRef, ImportDocRef


class TestSection:
    def test_minimal_section(self):
        s = Section(title="Introduction", order=1, name="introduction")
        assert s.title == "Introduction"
        assert s.order == 1
        assert s.name == "introduction"
        assert s.generated_content is None
        assert s.source_ids == []
        assert s.sections == []
        assert s.status == SectionStatus.PENDING

    def test_full_section(self):
        s = Section(
            title="Methods",
            order=2,
            name="methods",
            generated_content="We used a survey...",
            source_ids=["doc-1", "doc-2"],
            num_lines_min=10,
            num_lines_max=20,
            job_id="job-123",
            status=SectionStatus.COMPLETED,
        )
        assert s.generated_content == "We used a survey..."
        assert s.source_ids == ["doc-1", "doc-2"]
        assert s.num_lines_min == 10
        assert s.num_lines_max == 20
        assert s.job_id == "job-123"
        assert s.status == SectionStatus.COMPLETED
        assert s.section_id is not None

    def test_section_has_uuid(self):
        s1 = Section(title="A", order=1, name="a")
        s2 = Section(title="B", order=2, name="b")
        assert s1.section_id != s2.section_id

    def test_section_status_enum(self):
        assert SectionStatus.PENDING.value == "pending"
        assert SectionStatus.PROCESSING.value == "processing"
        assert SectionStatus.FAILED.value == "failed"
        assert SectionStatus.COMPLETED.value == "completed"

    def test_section_with_citations(self):
        from document_common.schemas.citations import CitationStack, Citation, CitationLabel
        primary = Citation(
            span_text="test", span_start=0, span_end=4,
            file_id="uuid-1", chunk_id="chunk-1",
            citation_label=CitationLabel.DIRECT_QUOTE,
            confidence=0.95, similarity_score=0.90,
        )
        stack = CitationStack(
            span_text="test", span_start=0, span_end=4,
            primary_citation=primary, stack_confidence=0.95,
        )
        s = Section(
            title="A", order=1, name="a",
            citations=[[stack]],
            citation_extraction_status="completed",
        )
        assert s.citations is not None
        assert len(s.citations) == 1
        assert s.citation_extraction_status == "completed"

    def test_section_model_dump(self):
        s = Section(title="Test", order=1, name="test")
        d = s.model_dump()
        assert d["title"] == "Test"
        assert d["order"] == 1
        assert "section_id" in d


class TestDocument:
    def test_minimal_document(self):
        r = Document(job_id="job-123", file_ids=["file-1", "file-2"])
        assert r.status == DocumentStatus.PENDING
        assert r.sections is None
        assert r.job_id == "job-123"
        assert r.created_at is not None

    def test_full_document(self):
        s1 = Section(title="Intro", order=1, name="intro")
        s2 = Section(title="Methods", order=2, name="methods")
        r = Document(
            status=DocumentStatus.COMPLETED,
            sections=[s1, s2],
            job_id="job-456",
            file_ids=["file-1"],
        )
        assert r.status == DocumentStatus.COMPLETED
        assert len(r.sections) == 2
        assert r.sections[0].title == "Intro"
        assert r.sections[1].title == "Methods"

    def test_document_status_enum(self):
        assert DocumentStatus.PENDING.value == "pending"
        assert DocumentStatus.PROCESSING.value == "processing"
        assert DocumentStatus.FAILED.value == "failed"
        assert DocumentStatus.COMPLETED.value == "completed"

    def test_document_model_dump(self):
        r = Document(job_id="job-1", file_ids=["f1"])
        d = r.model_dump()
        assert d["job_id"] == "job-1"
        assert d["file_ids"] == ["f1"]


class TestDocRef:
    def test_valid_doc_ref(self):
        dr = DocRef(
            coreDataId="a1b2c3d4-e5f6-7890-1234-567890abcdef",
            coreType="document",
            coreDataSource="Document Manager",
        )
        assert dr.coreDataId == "a1b2c3d4-e5f6-7890-1234-567890abcdef"
        assert dr.coreType == "document"
        assert dr.coreDataSource == "Document Manager"

    def test_doc_ref_multimedia(self):
        dr = DocRef(
            coreDataId="uuid-1",
            coreType="multimedia",
            coreDataSource="SourceBox",
        )
        assert dr.coreType == "multimedia"


class TestImportDocRef:
    def test_valid_import_doc_ref(self):
        idr = ImportDocRef(
            coreDataId="uuid-1",
            coreType="document",
            coreDataSource="Document Manager",
            plainText="This is some document content.",
        )
        assert idr.plainText == "This is some document content."

    def test_empty_plaintext_raises(self):
        with pytest.raises(ValidationError) as exc:
            ImportDocRef(
                coreDataId="uuid-1",
                coreType="document",
                coreDataSource="Document Manager",
                plainText="",
            )
        assert "plainText cannot be empty" in str(exc.value)

    def test_whitespace_plaintext_raises(self):
        with pytest.raises(ValidationError) as exc:
            ImportDocRef(
                coreDataId="uuid-1",
                coreType="document",
                coreDataSource="Document Manager",
                plainText="   \n\t  ",
            )
        assert "plainText cannot be empty" in str(exc.value)
