"""
Tests for Structuredassembly when the LLM or validation fails.

Goals:
  - Failures must surface with clear error messages (prompt name, base_model name, details).
  - Document current behaviour: no retry on failure (so we can add retry later and test it).
"""
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.components.base.assembly import StructuredAssembly


class _DummyModel(BaseModel):
    """Minimal model for tests."""
    value: str


class TestStructuredAssemblyFailurePropagation:
    """When generator.run() or validation fails, error is propagated with clear message."""

    @pytest.fixture
    def gen(self):
        """Patch PromptProvider and ChatPromptBuilder so run() reaches generator.run() (and prompt_messages is set)."""
        with patch("src.components.base.assembly.PromptProvider") as mock_pp, patch(
            "src.components.base.assembly.ChatPromptBuilder"
        ) as mock_cpb:
            mock_pp.return_value.get_prompt.return_value = "Hello {{ documents }}"
            mock_cpb.return_value.run.return_value = {"prompt": []}
            sg = StructuredAssembly(
                prompt_name="test_prompt",
                base_model=_DummyModel,
                fixed_inputs=None,
            )
            yield sg

    def test_when_generator_raises_exception_propagates_with_context(self, gen):
        """Generator raises (e.g. API error) => Exception with base_model name and original error."""
        gen.generator.run = MagicMock(side_effect=RuntimeError("LLM timeout"))
        with pytest.raises(Exception) as exc_info:
            gen.run(documents=None, inputs={})
        msg = str(exc_info.value)
        assert "DummyModel" in msg or "_DummyModel" in msg
        assert "LLM timeout" in msg

    def test_when_generator_raises_no_retry_by_default(self, gen):
        """Current behaviour: generator.run is called once, no retry on failure."""
        gen.generator.run = MagicMock(side_effect=RuntimeError("fail"))
        with pytest.raises(Exception):
            gen.run(documents=None, inputs={})
        assert gen.generator.run.call_count == 1

    def test_validation_error_from_generator_raises_value_error_with_context(self, gen):
        """ValidationError from generator (e.g. LLM returned invalid JSON) => ValueError with prompt/model."""
        from pydantic import ValidationError
        try:
            _DummyModel.model_validate({})  # missing required 'value'
        except ValidationError as e:
            gen.generator.run = MagicMock(side_effect=e)
        with pytest.raises(ValueError) as exc_info:
            gen.run(documents=None, inputs={})
        msg = str(exc_info.value)
        assert "test_prompt" in msg or "DummyModel" in msg

    def test_reply_not_basemodel_raises_clear_exception(self, gen):
        """When generator returns reply that is not a BaseModel instance => Exception with type info."""
        gen.generator.run = MagicMock(return_value={"reply": "not a BaseModel"})
        with pytest.raises(Exception) as exc_info:
            gen.run(documents=None, inputs={})
        msg = str(exc_info.value)
        assert "BaseModel" in msg or "str" in msg
