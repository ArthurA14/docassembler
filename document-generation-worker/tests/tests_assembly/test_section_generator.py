"""
Tests for SectionGenerator base and DirectSectionGenerator (document gen: registry, run with document_writer).

Strategy:
  - SectionGenerator: register(code), from_code(code) => instance; get_document_requirements(code);
    document_writer without document raises; update_document_section delegates to document_writer.
  - DirectSectionGenerator: run(sections) with mocked Textassembly and document_writer => section status updated, content set.
"""
import pytest
from unittest.mock import MagicMock, patch
from haystack import Document
from haystack.document_stores.types import DocumentStore

from src.components.doc_assembler.models import Document as AssembledDocument, Section, SectionStatus
from src.components.doc_assembler.document_writer import DocumentWriter
from src.components.doc_assembler.section_generators.base_registry import SectionGenerator
from src.components.doc_assembler.section_generators.generators.direct import DirectSectionGenerator


class TestSectionGeneratorRegistry:
    """from_code, get_document_requirements, register."""

    def test_from_code_unknown_raises(self):
        with pytest.raises(ValueError) as exc_info:
            SectionGenerator.from_code("unknown_code", section=MagicMock(), document_store=MagicMock())
        assert "Unknown" in str(exc_info.value) or "unknown_code" in str(exc_info.value)

    def test_from_code_returns_direct_instance(self):
        section = Section(title="Intro", order=0, name="intro", job_id="j1")
        mock_store = MagicMock(spec=DocumentStore)
        instance = SectionGenerator.from_code(
            "direct",
            prompt_name="p1",
            section=section,
            document_store=mock_store,
        )
        assert isinstance(instance, DirectSectionGenerator)
        assert instance.section.name == "intro"

    def test_get_document_requirements_direct_returns_critical_optional(self):
        req = SectionGenerator.get_document_requirements("direct")
        assert "critical" in req
        assert "optional" in req
        assert isinstance(req["critical"], list)
        assert req["critical"] == []

    def test_get_document_requirements_narrative_has_critical(self):
        req = SectionGenerator.get_document_requirements("narrative")
        assert "critical" in req
        assert "refined_summary" in req["critical"] or "original" in req["critical"]


class TestSectionGeneratorInit:
    """document_writer without document raises."""

    def test_document_writer_without_document_raises(self):
        section = Section(title="A", order=0, name="a", job_id="j1")
        mock_store = MagicMock(spec=DocumentStore)
        with pytest.raises(ValueError) as exc_info:
            SectionGenerator.from_code(
                "direct",
                prompt_name="p1",
                section=section,
                document_store=mock_store,
                document=None,
                document_writer=MagicMock(),
            )
        assert "document_writer" in str(exc_info.value).lower() or "document" in str(exc_info.value).lower()


class TestDirectSectionGeneratorRun:
    """run(sections) updates section content and status via document_writer."""

    @pytest.fixture
    def section(self):
        return Section(title="Intro", order=0, name="intro", job_id="job-1", section_id="sec-1")

    @pytest.fixture
    def mock_store(self):
        return MagicMock(spec=DocumentStore)

    @pytest.fixture
    def mock_document_writer(self):
        return MagicMock(spec=DocumentWriter)

    @pytest.fixture
    def document(self):
        return AssembledDocument(file_ids=["f1"], job_id="job-1")

    def test_run_sets_content_and_completed_status(
        self, section, mock_store, mock_document_writer, document
    ):
        with patch(
            "src.components.doc_assembler.section_generators.generators.direct.TextAssembly"
        ) as mock_tg_class:
            mock_tg = MagicMock()
            mock_tg.run.return_value = {"llm_answer": "Generated introduction."}
            mock_tg_class.return_value = mock_tg
            gen = DirectSectionGenerator(
                prompt_name="intro_prompt",
                section=section,
                document_store=mock_store,
                document=document,
                document_writer=mock_document_writer,
            )
            result = gen.run(sections=[])
            assert result["section"][0].generated_content == "Generated introduction."
            assert result["section"][0].status == SectionStatus.COMPLETED
            assert mock_document_writer.update_document_section.call_count >= 1

    def test_run_with_single_section_input(self, section, mock_store, mock_document_writer, document):
        with patch(
            "src.components.doc_assembler.section_generators.generators.direct.TextAssembly"
        ) as mock_tg_class:
            mock_tg = MagicMock()
            mock_tg.run.return_value = {"llm_answer": "Content."}
            mock_tg_class.return_value = mock_tg
            gen = DirectSectionGenerator(
                prompt_name="p1",
                section=section,
                document_store=mock_store,
                document=document,
                document_writer=mock_document_writer,
            )
            other = Section(title="Other", order=0, name="other", job_id="job-1", generated_content="Other content.")
            result = gen.run(sections=other)  # single Section, not list
            assert len(result["section"]) == 1
            assert result["section"][0].generated_content == "Content."
