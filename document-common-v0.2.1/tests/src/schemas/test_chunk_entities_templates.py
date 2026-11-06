import pytest
from pydantic import ValidationError

from document_common.schemas.chunk import Chunk
from document_common.schemas.entities import Entity
from document_common.schemas.document_template import (
    NumLines,
    SectionContent,
    SectionBlock,
    DocumentTemplate,
)
from document_common.schemas.preprocessing_template import ProcessorTemplate, PreprocessingTemplate


class TestChunk:
    def test_minimal_chunk(self):
        c = Chunk(type="original", file_id="file-123")
        assert c.type == "original"
        assert c.file_id == "file-123"
        assert c.version is None
        assert c.parent_id is None
        assert c.order is None

    def test_full_chunk(self):
        c = Chunk(
            type="hquery",
            file_id="file-456",
            version=2,
            project_name="my-project",
            parent_id="parent-uuid",
            order=3,
            core_data_id="core-uuid-1",
            core_type="document",
            core_data_source="Document Manager",
        )
        assert c.type == "hquery"
        assert c.version == 2
        assert c.project_name == "my-project"
        assert c.parent_id == "parent-uuid"
        assert c.order == 3
        assert c.core_data_id == "core-uuid-1"

    def test_chunk_summary_type(self):
        c = Chunk(type="summary", file_id="file-789")
        assert c.type == "summary"


class TestEntity:
    def test_minimal_entity(self):
        e = Entity(name="John Doe", context="CEO of Company X")
        assert e.name == "John Doe"
        assert e.context == "CEO of Company X"


class TestNumLines:
    def test_valid_num_lines(self):
        nl = NumLines(min=10, max=20)
        assert nl.min == 10
        assert nl.max == 20


class TestSectionContent:
    def test_minimal_section_content(self):
        sc = SectionContent(
            title="Introduction",
            name="introduction",
            prompt="Write an introduction.",
            method="llm",
            num_lines=NumLines(min=5, max=10),
        )
        assert sc.title == "Introduction"
        assert sc.method == "llm"
        assert sc.num_lines.min == 5

    def test_section_content_with_parameters(self):
        sc = SectionContent(
            title="Analysis",
            name="analysis",
            prompt="Analyze the data.",
            method="analysis",
            method_parameters={"model": "gpt-4"},
            using=["doc-1", "doc-2"],
            num_lines=NumLines(min=10, max=30),
        )
        assert sc.method_parameters == {"model": "gpt-4"}
        assert sc.using == ["doc-1", "doc-2"]

    def test_name_must_be_snake_case_valid(self):
        sc = SectionContent(
            title="Test",
            name="valid_name",
            prompt="test",
            method="test",
            num_lines=NumLines(min=1, max=10),
        )
        assert sc.name == "valid_name"

    def test_name_must_be_snake_case_with_numbers(self):
        sc = SectionContent(
            title="Test",
            name="section_1",
            prompt="test",
            method="test",
            num_lines=NumLines(min=1, max=10),
        )
        assert sc.name == "section_1"

    def test_name_must_be_snake_case_uppercase_raises(self):
        with pytest.raises(ValidationError) as exc:
            SectionContent(
                title="Test",
                name="InvalidName",
                prompt="test",
                method="test",
                num_lines=NumLines(min=1, max=10),
            )
        assert "snake_case" in str(exc.value)

    def test_name_must_be_snake_case_starts_with_number_raises(self):
        with pytest.raises(ValidationError) as exc:
            SectionContent(
                title="Test",
                name="1invalid",
                prompt="test",
                method="test",
                num_lines=NumLines(min=1, max=10),
            )
        assert "snake_case" in str(exc.value)

    def test_name_must_be_snake_case_hyphen_raises(self):
        with pytest.raises(ValidationError) as exc:
            SectionContent(
                title="Test",
                name="invalid-name",
                prompt="test",
                method="test",
                num_lines=NumLines(min=1, max=10),
            )
        assert "snake_case" in str(exc.value)


class TestSectionBlock:
    def test_section_block(self):
        sc = SectionContent(
            title="Intro", name="intro", prompt="intro prompt", method="llm",
            num_lines=NumLines(min=5, max=10),
        )
        sb = SectionBlock(section=sc)
        assert sb.section.title == "Intro"


class TestDocumentTemplate:
    def test_valid_document_template(self):
        sc1 = SectionContent(
            title="Introduction", name="introduction", prompt="intro", method="llm",
            num_lines=NumLines(min=5, max=10),
        )
        sc2 = SectionContent(
            title="Methods", name="methods", prompt="methods", method="analysis",
            num_lines=NumLines(min=10, max=20),
        )
        rt = DocumentTemplate(
            template_name="my-document",
            description="A test document template",
            document=[SectionBlock(section=sc1), SectionBlock(section=sc2)],
        )
        assert rt.template_name == "my-document"
        assert len(rt.document) == 2
        assert rt.document[0].section.title == "Introduction"

    def test_empty_document(self):
        rt = DocumentTemplate(
            template_name="empty",
            description="No sections",
            document=[],
        )
        assert rt.document == []


class TestProcessorTemplate:
    def test_minimal_processor(self):
        pt = ProcessorTemplate(type="direct")
        assert pt.type == "direct"
        assert pt.using is None

    def test_processor_with_using(self):
        pt = ProcessorTemplate(type="entity_extraction", using="preprocessor-1")
        assert pt.type == "entity_extraction"
        assert pt.using == "preprocessor-1"


class TestPreprocessingTemplate:
    def test_valid_preprocessing_template(self):
        pt1 = ProcessorTemplate(type="direct")
        pt2 = ProcessorTemplate(type="entity_extraction", using="proc-1")
        pt = PreprocessingTemplate(
            preprocessing=[pt1, pt2],
            context={"key": "value"},
        )
        assert len(pt.preprocessing) == 2
        assert pt.context == {"key": "value"}
