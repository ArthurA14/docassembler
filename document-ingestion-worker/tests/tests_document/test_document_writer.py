"""
Tests for documentWriter (document gen: update document/section status, get document/sections).

Strategy:
  - documentWriter persists document and Section to document stores.
  - We test: update_document / update_document_section write with correct id and meta;
    get_document / get_document_sections / get_document_status return expected or None when empty.
"""
import pytest
from unittest.mock import MagicMock
from haystack import Document
from haystack.document_stores.types import DocumentStore

from src.components.doc_assembler.models import Document as AssembledDocument, DocumentStatus, Section, SectionStatus
from src.components.doc_assembler.document_writer import DocumentWriter


def _make_document(job_id: str = "job-1", status: DocumentStatus = DocumentStatus.PENDING):
    return AssembledDocument(file_ids=["f1"], job_id=job_id, status=status)


def _make_section(section_id: str = "sec-1", job_id: str = "job-1", content: str = ""):
    return Section(
        title="Intro",
        order=0,
        name="intro",
        section_id=section_id,
        job_id=job_id,
        generated_content=content,
        status=SectionStatus.COMPLETED,
    )


class TestDocumentWriterUpdate:
    """update_document and update_document_section write to stores."""

    @pytest.fixture
    def mock_document_store(self):
        return MagicMock(spec=DocumentStore)

    @pytest.fixture
    def mock_section_store(self):
        return MagicMock(spec=DocumentStore)

    @pytest.fixture
    def writer(self, mock_document_store, mock_section_store):
        return DocumentWriter(
            document_document_store=mock_document_store,
            section_document_store=mock_section_store,
        )

    def test_update_document_writes_document_with_job_id(self, writer, mock_document_store):
        document = _make_document(job_id="job-123")
        result = writer.update_document(document=document)
        assert result is True
        mock_document_store.write_documents.assert_called_once()
        call_args = mock_document_store.write_documents.call_args[0]
        docs = call_args[0]
        assert len(docs) == 1
        assert docs[0].id == "job-123"
        assert docs[0].meta["job_id"] == "job-123"
        assert docs[0].meta["status"] == DocumentStatus.PENDING.value

    def test_update_document_section_writes_with_section_id(self, writer, mock_section_store):
        section = _make_section(section_id="sec-abc", content="Generated text.")
        result = writer.update_document_section(section=section)
        assert result is True
        mock_section_store.write_documents.assert_called_once()
        docs = mock_section_store.write_documents.call_args[0][0]
        assert docs[0].id == "sec-abc"
        assert docs[0].content == "Generated text."
        assert docs[0].meta["section_id"] == "sec-abc"

    def test_update_document_section_raises_when_section_id_missing(self, writer):
        section = _make_section(content="x")
        section.section_id = None  # type: ignore
        with pytest.raises(ValueError) as exc_info:
            writer.update_document_section(section=section)
        assert "section_id must be set" in str(exc_info.value)


class TestDocumentWriterGet:
    """get_document, get_document_sections, get_document_status, get_document_section."""

    @pytest.fixture
    def mock_document_store(self):
        return MagicMock(spec=DocumentStore)

    @pytest.fixture
    def mock_section_store(self):
        return MagicMock(spec=DocumentStore)

    @pytest.fixture
    def writer(self, mock_document_store, mock_section_store):
        return DocumentWriter(
            document_document_store=mock_document_store,
            section_document_store=mock_section_store,
        )

    def test_get_document_returns_none_when_empty(self, writer, mock_document_store):
        mock_document_store.filter_documents.return_value = []
        result = writer.get_document(job_id="missing")
        assert result is None

    def test_get_document_returns_document_when_found(self, writer, mock_document_store, mock_section_store):
        document_meta = _make_document(job_id="job-1").model_dump(mode="json")
        mock_document_store.filter_documents.return_value = [
            Document(id="job-1", content="", meta=document_meta)
        ]
        mock_section_store.filter_documents.return_value = []
        result = writer.get_document(job_id="job-1")
        assert result is not None
        assert result.job_id == "job-1"

    def test_get_document_status_returns_none_when_document_missing(self, writer, mock_document_store):
        mock_document_store.filter_documents.return_value = []
        result = writer.get_document_status(job_id="missing")
        assert result is None

    def test_get_document_section_returns_none_when_missing(self, writer, mock_section_store):
        mock_section_store.filter_documents.return_value = []
        result = writer.get_document_section(job_id="job-1", section_id="sec-1")
        assert result is None
