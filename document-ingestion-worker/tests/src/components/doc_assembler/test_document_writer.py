from unittest.mock import MagicMock

import pytest
from haystack import Document

from src.components.doc_assembler.models import Section, SectionStatus
from src.components.doc_assembler.document_writer import DocumentWriter
from document_common.schemas.citations import Citation, CitationLabel, CitationStack


class TestDocumentWriterCitationMethods:
    """Unit tests for citation retrieval methods in documentWriter."""

    @pytest.fixture
    def mock_stores(self):
        """Create mock document stores for documents and sections."""
        document_store = MagicMock()
        section_store = MagicMock()
        return document_store, section_store

    @pytest.fixture
    def document_writer(self, mock_stores):
        """Create a documentWriter with mocked stores."""
        document_store, section_store = mock_stores
        return DocumentWriter(
            document_document_store=document_store,
            section_document_store=section_store,
        )

    @pytest.fixture
    def sample_citation_stacks(self):
        """Create sample citation stacks."""
        citation1 = Citation(
            source_id="doc-1",
            source_doc_id="doc-1",
            quote_text="Machine learning algorithms are used for pattern recognition",
            start_index=0,
            end_index=60,
            citation_type=CitationLabel.CLOSE_PARAPHRASE,
            similarity_score=0.85,
            confidence=0.88,
        )
        citation2 = Citation(
            source_id="doc-2",
            source_doc_id="doc-2",
            quote_text="Neural networks consist of interconnected nodes",
            start_index=0,
            end_index=47,
            citation_type=CitationLabel.DIRECT_QUOTE,
            similarity_score=0.95,
            confidence=0.92,
        )
        return [
            CitationStack(
                span_text="machine learning algorithms",
                span_start_index=0,
                span_end_index=27,
                primary_citation=citation1,
                supporting_citations=[],
                stack_confidence=0.88,
            ),
            CitationStack(
                span_text="neural networks",
                span_start_index=0,
                span_end_index=15,
                primary_citation=citation2,
                supporting_citations=[],
                stack_confidence=0.92,
            ),
        ]

    def test_get_section_citations_success(self, document_writer, mock_stores, sample_citation_stacks):
        """Test successful retrieval of citations for a section."""
        _, section_store = mock_stores

        # Create section with citations
        section = Section(
            name="test_section",
            order=1,
            title="Test Section",
            job_id="job-123",
            section_id="section-1",
            generated_content="Content about ML and neural networks.",
            source_ids=[["doc-1", "doc-2"]],
            status=SectionStatus.COMPLETED,
            citations=sample_citation_stacks,
            citation_extraction_status="completed",
        )

        # Mock document store to return section
        section_doc = Document(
            id="section-1",
            content=section.generated_content,
            meta=section.model_dump(mode="json"),
        )
        section_store.filter_documents = MagicMock(return_value=[section_doc])

        # Get citations
        citations = document_writer.get_section_citations(section_id="section-1")

        assert citations is not None
        assert len(citations) == 2
        assert citations[0].span_text == "machine learning algorithms"
        assert citations[1].span_text == "neural networks"

    def test_get_section_citations_not_found(self, document_writer, mock_stores):
        """Test retrieval when section doesn't exist."""
        _, section_store = mock_stores

        # Mock document store to return empty list
        section_store.filter_documents = MagicMock(return_value=[])

        citations = document_writer.get_section_citations(section_id="nonexistent")

        assert citations is None

    def test_get_section_citations_not_extracted(self, document_writer, mock_stores):
        """Test retrieval when citations were not extracted for the section."""
        _, section_store = mock_stores

        # Create section without citations
        section = Section(
            name="test_section",
            order=1,
            title="Test Section",
            job_id="job-123",
            section_id="section-1",
            generated_content="Content",
            source_ids=[],
            status=SectionStatus.COMPLETED,
            citations=None,
            citation_extraction_status="skipped",
        )

        section_doc = Document(
            id="section-1",
            content=section.generated_content,
            meta=section.model_dump(mode="json"),
        )
        section_store.filter_documents = MagicMock(return_value=[section_doc])

        citations = document_writer.get_section_citations(section_id="section-1")

        assert citations is None

    def test_get_section_citations_failed_extraction(self, document_writer, mock_stores):
        """Test retrieval when citation extraction failed."""
        _, section_store = mock_stores

        section = Section(
            name="test_section",
            order=1,
            title="Test Section",
            job_id="job-123",
            section_id="section-1",
            generated_content="Content",
            source_ids=[["doc-1"]],
            status=SectionStatus.COMPLETED,
            citations=None,
            citation_extraction_status="failed",
        )

        section_doc = Document(
            id="section-1",
            content=section.generated_content,
            meta=section.model_dump(mode="json"),
        )
        section_store.filter_documents = MagicMock(return_value=[section_doc])

        citations = document_writer.get_section_citations(section_id="section-1")

        assert citations is None

    def test_get_document_citations_success(self, document_writer, mock_stores, sample_citation_stacks):
        """Test successful retrieval of all citations for a document."""
        document_store, section_store = mock_stores

        # Create multiple sections with citations
        sections = [
            Section(
                name="section_1",
                order=1,
                title="Section 1",
                job_id="job-123",
                section_id="section-1",
                generated_content="Content 1",
                source_ids=[["doc-1"]],
                status=SectionStatus.COMPLETED,
                citations=sample_citation_stacks[:1],
                citation_extraction_status="completed",
            ),
            Section(
                name="section_2",
                order=1,
                title="Section 2",
                job_id="job-123",
                section_id="section-2",
                generated_content="Content 2",
                source_ids=[["doc-2"]],
                status=SectionStatus.COMPLETED,
                citations=sample_citation_stacks[1:],
                citation_extraction_status="completed",
            ),
            Section(
                name="section_3",
                order=1,
                title="Section 3",
                job_id="job-123",
                section_id="section-3",
                generated_content="Content 3",
                source_ids=[],
                status=SectionStatus.COMPLETED,
                citations=None,
                citation_extraction_status="skipped",
            ),
        ]

        section_docs = [
            Document(id=s.section_id, content=s.generated_content, meta=s.model_dump(mode="json")) for s in sections
        ]
        section_store.filter_documents = MagicMock(return_value=section_docs)

        # Get all citations
        citations_by_section = document_writer.get_document_citations(job_id="job-123")

        assert len(citations_by_section) == 2  # Only sections with completed citations
        assert "section-1" in citations_by_section
        assert "section-2" in citations_by_section
        assert "section-3" not in citations_by_section  # Skipped section excluded

        assert len(citations_by_section["section-1"]) == 1
        assert len(citations_by_section["section-2"]) == 1

    def test_get_document_citations_no_sections(self, document_writer, mock_stores):
        """Test retrieval when document has no sections."""
        _, section_store = mock_stores

        section_store.filter_documents = MagicMock(return_value=[])

        citations_by_section = document_writer.get_document_citations(job_id="job-123")

        assert citations_by_section == {}

    def test_get_document_citations_no_extracted_citations(self, document_writer, mock_stores):
        """Test retrieval when no sections have extracted citations."""
        _, section_store = mock_stores

        # Create sections without citations
        sections = [
            Section(
                name="section_1",
                order=1,
                title="Section 1",
                job_id="job-123",
                section_id="section-1",
                generated_content="Content 1",
                source_ids=[],
                status=SectionStatus.COMPLETED,
                citations=None,
                citation_extraction_status="skipped",
            ),
            Section(
                name="section_2",
                order=1,
                title="Section 2",
                job_id="job-123",
                section_id="section-2",
                generated_content="Content 2",
                source_ids=[["doc-1"]],
                status=SectionStatus.COMPLETED,
                citations=None,
                citation_extraction_status="failed",
            ),
        ]

        section_docs = [
            Document(id=s.section_id, content=s.generated_content, meta=s.model_dump(mode="json")) for s in sections
        ]
        section_store.filter_documents = MagicMock(return_value=section_docs)

        citations_by_section = document_writer.get_document_citations(job_id="job-123")

        assert citations_by_section == {}

    def test_get_citation_extraction_status_success(self, document_writer, mock_stores):
        """Test retrieval of citation extraction status for all sections."""
        _, section_store = mock_stores

        sections = [
            Section(
                name="section_1",
                order=1,
                title="Section 1",
                job_id="job-123",
                section_id="section-1",
                generated_content="Content 1",
                source_ids=[["doc-1"]],
                status=SectionStatus.COMPLETED,
                citation_extraction_status="completed",
            ),
            Section(
                name="section_2",
                order=1,
                title="Section 2",
                job_id="job-123",
                section_id="section-2",
                generated_content="Content 2",
                source_ids=[["doc-2"]],
                status=SectionStatus.COMPLETED,
                citation_extraction_status="failed",
            ),
            Section(
                name="section_3",
                order=1,
                title="Section 3",
                job_id="job-123",
                section_id="section-3",
                generated_content="Content 3",
                source_ids=[],
                status=SectionStatus.COMPLETED,
                citation_extraction_status="skipped",
            ),
            Section(
                name="section_4",
                order=1,
                title="Section 4",
                job_id="job-123",
                section_id="section-4",
                generated_content="Content 4",
                source_ids=[["doc-3"]],
                status=SectionStatus.PROCESSING,
                citation_extraction_status=None,
            ),
        ]

        section_docs = [
            Document(id=s.section_id, content=s.generated_content, meta=s.model_dump(mode="json")) for s in sections
        ]
        section_store.filter_documents = MagicMock(return_value=section_docs)

        status_map = document_writer.get_citation_extraction_status(job_id="job-123")

        assert len(status_map) == 4
        assert status_map["section-1"] == "completed"
        assert status_map["section-2"] == "failed"
        assert status_map["section-3"] == "skipped"
        assert status_map["section-4"] == "unknown"  # None converted to "unknown"

    def test_get_citation_extraction_status_no_sections(self, document_writer, mock_stores):
        """Test status retrieval when document has no sections."""
        _, section_store = mock_stores

        section_store.filter_documents = MagicMock(return_value=[])

        status_map = document_writer.get_citation_extraction_status(job_id="job-123")

        assert status_map == {}


class TestIntegration:
    """Integration tests for citation retrieval workflow."""

    def test_full_citation_retrieval_workflow(self):
        """Test complete workflow of storing and retrieving citations."""
        document_store = MagicMock()
        section_store = MagicMock()
        writer = DocumentWriter(
            document_document_store=document_store,
            section_document_store=section_store,
        )

        # Create section with citations
        citation = Citation(
            source_id="doc-1",
            source_doc_id="doc-1",
            quote_text="Source content",
            start_index=0,
            end_index=14,
            citation_type=CitationLabel.DIRECT_QUOTE,
            similarity_score=0.95,
            confidence=0.92,
        )
        citation_stack = CitationStack(
            span_text="test text",
            span_start_index=0,
            span_end_index=9,
            primary_citation=citation,
            supporting_citations=[],
            stack_confidence=0.92,
        )

        section = Section(
            name="integration_test",
            order=1,
            title="Integration Test",
            job_id="job-456",
            section_id="section-5",
            generated_content="Test content with citations.",
            source_ids=[["doc-1"]],
            status=SectionStatus.COMPLETED,
            citations=[citation_stack],
            citation_extraction_status="completed",
        )

        # Simulate writing section
        section_doc = Document(
            id=section.section_id,
            content=section.generated_content,
            meta=section.model_dump(mode="json"),
        )

        # Mock retrieval
        section_store.filter_documents = MagicMock(return_value=[section_doc])

        # Retrieve citations
        citations = writer.get_section_citations(section_id="section-5")

        assert citations is not None
        assert len(citations) == 1
        assert citations[0].span_text == "test text"
        assert citations[0].primary_citation.core_data_id == "doc-1"
