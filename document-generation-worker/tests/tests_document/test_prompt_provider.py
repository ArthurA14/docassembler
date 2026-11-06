"""
Tests for PromptProvider (document gen: prompts in memory, get_prompt, add_prompt_from_str).

Strategy:
  - PromptProvider is a singleton; get_prompt(name) returns List[ChatMessage]; unknown name raises.
  - add_prompt_from_str(yaml) loads prompts; add_prompt(name, template, version) / overwrite behavior.
  - Ensure provider has .prompts dict (singleton may not have been initialized with a folder).
"""
import pytest
from pathlib import Path
from unittest.mock import patch

from haystack.dataclasses.chat_message import ChatMessage

from src.components.base.prompt import PromptProvider


def _provider_with_prompts():
    """Return singleton and ensure prompts dict and logger exist (init may not have run)."""
    from src.utils.logger import Logger
    provider = PromptProvider()
    if not hasattr(provider, "prompts"):
        provider.prompts = {}
    if not hasattr(provider, "logger"):
        provider.logger = Logger().get_logger("PromptProvider")
    return provider


class TestPromptProviderGetPrompt:
    """get_prompt returns messages; unknown name raises."""

    def test_get_prompt_unknown_raises(self):
        provider = _provider_with_prompts()
        with pytest.raises(ValueError) as exc_info:
            provider.get_prompt("nonexistent_prompt_xyz")
        assert "not found" in str(exc_info.value).lower() or "nonexistent" in str(exc_info.value).lower()

    def test_add_prompt_from_str_and_get_prompt(self):
        yaml_str = """
version: 1
test_prompt_provider_p1:
  system: |
    You are a test assistant.
  query: |
    Answer: {{input}}
"""
        provider = _provider_with_prompts()
        names = provider.add_prompt_from_str(yaml_str, overwrite_existing_prompt=True)
        assert "test_prompt_provider_p1" in names
        messages = provider.get_prompt("test_prompt_provider_p1")
        assert isinstance(messages, list)
        assert len(messages) >= 1


class TestPromptProviderAddPrompt:
    """add_prompt and overwrite behavior."""

    def test_add_prompt_twice_without_overwrite_raises(self):
        provider = _provider_with_prompts()
        template = {"system": "You are helpful.", "query": "Reply."}
        provider.add_prompt("dup_prompt", template, 1, overwrite_existing_prompt=False)
        with pytest.raises(ValueError) as exc_info:
            provider.add_prompt("dup_prompt", template, 1, overwrite_existing_prompt=False)
        assert "already exists" in str(exc_info.value).lower()

    def test_add_prompt_with_overwrite_succeeds(self):
        provider = _provider_with_prompts()
        template = {"system": "First.", "query": "Q1."}
        provider.add_prompt("overwrite_me", template, 1, overwrite_existing_prompt=True)
        provider.add_prompt(
            "overwrite_me", {"system": "Second.", "query": "Q2."}, 1, overwrite_existing_prompt=True
        )
        messages = provider.get_prompt("overwrite_me")
        assert messages is not None

    def test_add_prompt_with_invalid_template_structure_raises(self):
        """Template missing required keys (system, query) => rejet."""
        provider = _provider_with_prompts()
        with pytest.raises((ValueError, Exception)) as exc_info:
            provider.add_prompt("bad_prompt", {"only_key": "x"}, 1)
        assert "system" in str(exc_info.value).lower() or "query" in str(exc_info.value).lower() or "prompt" in str(exc_info.value).lower()
