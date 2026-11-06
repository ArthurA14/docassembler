"""
Tests for PromptService (document gen: get_prompt, list_all_prompts, upsert_prompt, get_latest_content).

Strategy:
  - PromptService delegates to PromptStorage. We mock storage and test service logic.
"""
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from haystack import Document

from src.services.prompt_service import PromptService


class TestPromptServiceGetPrompt:
    """get_prompt returns dict or None."""

    @pytest.fixture
    def mock_storage(self):
        return MagicMock()

    @pytest.fixture
    def service(self, mock_storage):
        return PromptService(storage=mock_storage)

    def test_get_prompt_returns_none_when_not_found(self, service, mock_storage):
        mock_storage.get_prompt.return_value = None
        result = service.get_prompt("missing")
        assert result is None

    def test_get_prompt_returns_dict_when_found(self, service, mock_storage):
        doc = Document(
            id="p1",
            content="version: 1\nmy_prompt:\n  system: Hi",
            meta={"prompt_name": "my_prompt", "type": "t1", "mode": "no_think", "version": 1},
        )
        mock_storage.get_prompt.return_value = doc
        result = service.get_prompt("my_prompt")
        assert result is not None
        assert result["name"] == "my_prompt"
        assert "content" in result
        assert result.get("version") == 1


class TestPromptServiceUpsertAndList:
    """upsert_prompt, list_all_prompts, get_latest_content."""

    @pytest.fixture
    def mock_storage(self):
        return MagicMock()

    @pytest.fixture
    def service(self, mock_storage):
        return PromptService(storage=mock_storage)

    def test_upsert_prompt_calls_storage(self, service, mock_storage):
        service.upsert_prompt("new_prompt", "version: 1\nnew_prompt:\n  system: x\n  query: y")
        mock_storage.upsert_prompt.assert_called_once()
        call_kw = mock_storage.upsert_prompt.call_args[1]
        assert call_kw["prompt_name"] == "new_prompt"
        assert "content" in call_kw

    def test_get_latest_content_raises_when_not_found(self, service, mock_storage):
        mock_storage.get_prompt.return_value = None
        with pytest.raises(ValueError) as exc_info:
            service.get_latest_content("missing")
        assert "not found" in str(exc_info.value).lower()

    def test_list_all_prompts_returns_from_storage(self, service, mock_storage):
        mock_storage.list_all_prompts.return_value = [{"name": "p1"}, {"name": "p2"}]
        result = service.list_all_prompts()
        assert len(result) == 2
        assert result[0]["name"] == "p1"
