"""
Tests for documentFactory and SectionGeneratorManager (document gen: template validation, doc availability, section pipeline).

Strategy:
  - documentFactory: load template (Path/str), validate (unique names, using refs, no cycle, one entry);
    validate_document_availability(files) => (bool, message); run_async updates status.
  - SectionGeneratorManager: one entry point (exactly one section with no deps); run() returns sections from pipeline.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

import yaml
from haystack import Document
from haystack.document_stores.types import DocumentStore

from src.components.doc_assembler.models import Document as AssembledDocument, DocumentStatus, Section
from document_common.schemas.document import DocumentTemplate
from src.components.doc_assembler.document_factory import DocumentFactory, SectionGeneratorManager
from src.components.doc_assembler.section_generators.base_registry import SectionGenerator
from src.components.doc_assembler.section_generators.generators.direct import DirectSectionGenerator
from src.schemas import File, FileStatus


def _minimal_template_yaml(entries: list[dict]) -> str:
    """Build a minimal document template YAML string."""
    document = []
    for e in entries:
        block = {
            "section": {
                "title": e.get("title", "Section"),
                "name": e["name"],
                "prompt": e.get("prompt", "p1"),
                "method": e.get("method", "direct"),
                "num_lines": {"min": 1, "max": 10},
                **({} if "using" not in e else {"using": e["using"]}),
            }
        }
        document.append(block)
    data = {
        "template_name": "test_template",
        "description": "Test",
        "document": AssembledDocument,
    }
    return yaml.dump(data)


class TestDocumentFactoryTemplateValidation:
    """_validate_document_template: unique names, valid using, no cycle, one entry."""

    @pytest.fixture
    def mock_store(self):
        return MagicMock(spec=DocumentStore)

    @pytest.fixture
    def mock_writer(self):
        return MagicMock()

    def test_valid_template_one_section_passes(self, mock_store, mock_writer):
        template_str = _minimal_template_yaml([{"name": "intro", "method": "direct"}])
        document = AssembledDocument(file_ids=["f1"])
        section = Section(title="Intro", order=0, name="intro", job_id=document.job_id)
        with patch(
            "src.components.doc_assembler.section_generators.generators.direct.TextAssembly"
        ):
            real_sg = DirectSectionGenerator(
                prompt_name="p1",
                section=section,
                document_store=mock_store,
            )
        with patch(
            "src.components.doc_assembler.document_factory.SectionGenerator.from_code",
            return_value=real_sg,
        ), patch(
            "src.components.doc_assembler.document_factory.SectionGenerator.get_document_requirements",
            return_value={"critical": [], "optional": []},
        ):
            factory = DocumentFactory(
                template=template_str,
                document=document,
                document_writer=mock_writer,
                document_store=mock_store,
            )
            assert factory.document.job_id == document.job_id

    def test_duplicate_section_names_raise(self, mock_store, mock_writer):
        template_str = _minimal_template_yaml([
            {"name": "intro", "method": "direct"},
            {"name": "intro", "method": "direct"},
        ])
        document = AssembledDocument(file_ids=["f1"])
        with pytest.raises(ValueError) as exc_info:
            DocumentFactory(
                template=template_str,
                document=document,
                document_writer=mock_writer,
                document_store=mock_store,
            )
        assert "Duplicate" in str(exc_info.value) or "duplicate" in str(exc_info.value).lower()

    def test_template_wrong_type_raises(self, mock_store, mock_writer):
        """template must be Path or str; other type => ValueError."""
        document = AssembledDocument(file_ids=["f1"])
        with pytest.raises(ValueError) as exc_info:
            DocumentFactory(
                template=42,  # type: ignore
                document=document,
                document_writer=mock_writer,
                document_store=mock_store,
            )
        assert "Path" in str(exc_info.value) or "str" in str(exc_info.value)

    def test_template_invalid_yaml_structure_raises(self, mock_store, mock_writer):
        """YAML without required keys (e.g. document) => validation error, factory n'accepte pas."""
        document = AssembledDocument(file_ids=["f1"])
        invalid_yaml = "template_name: ok\ndescription: x\n"  # missing 'document' list
        with pytest.raises((ValueError, KeyError, TypeError, Exception)) as exc_info:
            DocumentFactory(
                template=invalid_yaml,
                document=document,
                document_writer=mock_writer,
                document_store=mock_store,
            )
        # Pydantic documentTemplate or constructor fails
        assert exc_info.value is not None

    def test_using_nonexistent_section_raises(self, mock_store, mock_writer):
        template_str = _minimal_template_yaml([
            {"name": "intro", "method": "direct"},
            {"name": "part2", "method": "direct", "using": ["nonexistent"]},
        ])
        document = AssembledDocument(file_ids=["f1"])
        with pytest.raises(ValueError) as exc_info:
            DocumentFactory(
                template=template_str,
                document=document,
                document_writer=mock_writer,
                document_store=mock_store,
            )
        assert "nonexistent" in str(exc_info.value) or "no section" in str(exc_info.value).lower()

    def test_template_from_path_loads(self, mock_store, mock_writer, tmp_path):
        template_str = _minimal_template_yaml([{"name": "only", "method": "direct"}])
        template_file = tmp_path / "template.yaml"
        template_file.write_text(template_str)
        document = AssembledDocument(file_ids=["f1"])
        section = Section(title="Only", order=0, name="only", job_id=document.job_id)
        with patch(
            "src.components.doc_assembler.section_generators.generators.direct.TextAssembly"
        ):
            real_sg = DirectSectionGenerator(
                prompt_name="p1", section=section, document_store=mock_store
            )
        with patch(
            "src.components.doc_assembler.document_factory.SectionGenerator.from_code",
            return_value=real_sg,
        ), patch(
            "src.components.doc_assembler.document_factory.SectionGenerator.get_document_requirements",
            return_value={"critical": [], "optional": []},
        ):
            factory = DocumentFactory(
                template=Path(template_file),
                document=document,
                document_writer=mock_writer,
                document_store=mock_store,
            )
            assert factory.document is document


class TestDocumentFactoryValidateDocumentAvailability:
    """validate_document_availability(files) => (validated, message)."""

    @pytest.fixture
    def mock_store(self):
        return MagicMock(spec=DocumentStore)

    @pytest.fixture
    def mock_writer(self):
        return MagicMock()

    def test_all_critical_present_validates(self, mock_store, mock_writer):
        template_str = _minimal_template_yaml([{"name": "intro", "method": "direct"}])
        document = AssembledDocument(file_ids=["f1"])
        section = Section(title="Intro", order=0, name="intro", job_id=document.job_id)
        with patch(
            "src.components.doc_assembler.section_generators.generators.direct.TextAssembly"
        ):
            real_sg = DirectSectionGenerator(
                prompt_name="p1", section=section, document_store=mock_store
            )
        with patch(
            "src.components.doc_assembler.document_factory.SectionGenerator.from_code",
            return_value=real_sg,
        ), patch(
            "src.components.doc_assembler.document_factory.SectionGenerator.get_document_requirements",
            return_value={"critical": ["original"], "optional": []},
        ):
            factory = DocumentFactory(
                template=template_str,
                document=document,
                document_writer=mock_writer,
                document_store=mock_store,
            )
        file_with_doc = File(
            name="", path="", type=".txt", size=0, status=FileStatus.importED,
            message="", content_id="c", doc_types=["original"],
        )
        valid, msg = factory.validate_document_availability([file_with_doc])
        assert valid is True

    def test_missing_critical_fails_validation(self, mock_store, mock_writer):
        template_str = _minimal_template_yaml([{"name": "intro", "method": "narrative"}])
        document = AssembledDocument(file_ids=["f1"])
        section = Section(title="Intro", order=0, name="intro", job_id=document.job_id)
        with patch(
            "src.components.doc_assembler.section_generators.generators.direct.TextAssembly"
        ):
            real_sg = DirectSectionGenerator(
                prompt_name="p1", section=section, document_store=mock_store
            )
        with patch(
            "src.components.doc_assembler.document_factory.SectionGenerator.from_code",
            return_value=real_sg,
        ), patch(
            "src.components.doc_assembler.document_factory.SectionGenerator.get_document_requirements",
            return_value={"critical": ["refined_summary"], "optional": []},
        ):
            factory = DocumentFactory(
                template=template_str,
                document=document,
                document_writer=mock_writer,
                document_store=mock_store,
            )
        file_without = File(
            name="", path="", type=".txt", size=0, status=FileStatus.importED,
            message="", content_id="c", doc_types=["original"],
        )
        valid, msg = factory.validate_document_availability([file_without])
        assert valid is False
        assert "refined_summary" in msg or "critical" in msg.lower()


class TestSectionGeneratorManager:
    """One entry point; run returns sections."""

    def test_raises_when_no_single_entry_point(self):
        mock_store = MagicMock(spec=DocumentStore)
        section_a = Section(title="A", order=0, name="a", job_id="j1")
        section_b = Section(title="B", order=1, name="b", job_id="j1")
        with patch(
            "src.components.doc_assembler.section_generators.generators.direct.TextAssembly"
        ):
            sg_a = DirectSectionGenerator(
                prompt_name="p1", section=section_a, document_store=mock_store
            )
            sg_b = DirectSectionGenerator(
                prompt_name="p1", section=section_b, document_store=mock_store
            )
        with pytest.raises(ValueError) as exc_info:
            SectionGeneratorManager(
                section_generators=[sg_a, sg_b],
                connections={},  # both have no deps => two "first"
            )
        assert "do not require" in str(exc_info.value).lower() or "start" in str(exc_info.value).lower()
