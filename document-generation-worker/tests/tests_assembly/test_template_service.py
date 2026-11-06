"""
Tests for TemplateService (document gen: get_template, upsert_template, list_all_templates, template_exists).

Strategy:
  - TemplateService delegates to TemplateStorage. We mock storage and test service logic.
"""
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from haystack import Document

from src.services.template_service import TemplateService


class TestTemplateServiceGetTemplate:
    """get_template returns content or raises."""

    @pytest.fixture
    def mock_storage(self):
        return MagicMock()

    @pytest.fixture
    def service(self, mock_storage):
        return TemplateService(storage=mock_storage)

    def test_get_template_raises_when_not_found(self, service, mock_storage):
        mock_storage.get_template.return_value = None
        with pytest.raises(ValueError) as exc_info:
            service.get_template("missing")
        assert "not found" in str(exc_info.value).lower()

    def test_get_template_returns_content_when_found(self, service, mock_storage):
        doc = Document(id="t1", content="template_name: t1\ndescription: Test\ndocument: []", meta={})
        mock_storage.get_template.return_value = doc
        result = service.get_template("t1")
        assert "template_name" in result
        assert "t1" in result


class TestTemplateServiceUpsertAndList:
    """upsert_template, list_all_templates, template_exists."""

    @pytest.fixture
    def mock_storage(self):
        return MagicMock()

    @pytest.fixture
    def service(self, mock_storage):
        return TemplateService(storage=mock_storage)

    def test_upsert_template_calls_storage(self, service, mock_storage):
        service.upsert_template("t1", "template_name: t1\ndescription: D\ndocument: []", description="D")
        mock_storage.upsert_template.assert_called_once()
        call_args = mock_storage.upsert_template.call_args[0]
        assert call_args[0] == "t1"
        assert "template_name" in call_args[1]

    def test_template_exists_delegates_to_storage(self, service, mock_storage):
        mock_storage.template_exists.return_value = True
        assert service.template_exists("t1") is True
        mock_storage.template_exists.assert_called_once_with("t1")

    def test_list_all_templates_returns_from_storage(self, service, mock_storage):
        mock_storage.list_all_templates.return_value = [{"name": "t1"}, {"name": "t2"}]
        result = service.list_all_templates()
        assert len(result) == 2


class TestTemplateServiceEmptyContent:
    """Edge case: empty content can cause crashes or confusing behavior."""

    @pytest.fixture
    def mock_storage(self):
        return MagicMock()

    @pytest.fixture
    def service(self, mock_storage):
        return TemplateService(storage=mock_storage)

    def test_upsert_template_empty_string_does_not_crash(self, service, mock_storage):
        """Upsert with content='' should not raise."""
        service.upsert_template("empty_tpl", "")
        mock_storage.upsert_template.assert_called_once()
        assert mock_storage.upsert_template.call_args[0][1] == ""

    def test_get_template_when_stored_content_empty_raises(self, service, mock_storage):
        """Doc exists but content='' => get_template raises (treated as not found)."""
        doc = Document(id="t1", content="", meta={})
        mock_storage.get_template.return_value = doc
        with pytest.raises(ValueError) as exc_info:
            service.get_template("empty_tpl")
        assert "not found" in str(exc_info.value).lower()
