from unittest.mock import MagicMock, patch

import pytest

from document_common.components.base.assembly import (
    ResponseGenerator,
    AsyncResponseGenerator,
    StructuredAssembly,
    TextAssembly,
)
from haystack.dataclasses.chat_message import ChatMessage


class TestResponseGenerator:
    @patch("document_common.components.base.assembly.OpenAI")
    def test_init(self, mock_openai, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.llm = MagicMock(
            model="gpt-4",
            api_key="test-key",
            temperature=0.5,
            max_output_tokens=500,
            timeout=30.0,
            base_url="http://localhost:8000/v1",
            top_p=0.95,
        )
        monkeypatch.setattr("document_common.components.base.assembly.config", mock_config)

        gen = ResponseGenerator()
        assert gen.model == "gpt-4"
        assert gen.temperature == 0.5
        assert gen.max_output_tokens == 500

    @patch("document_common.components.base.assembly.OpenAI")
    def test_run_unstructured(self, mock_openai, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.llm = MagicMock(
            model="gpt-4",
            api_key="test-key",
            temperature=0.0,
            max_output_tokens=1000,
            timeout=30.0,
            base_url="http://localhost:8000/v1",
            top_p=0.95,
        )
        monkeypatch.setattr("document_common.components.base.assembly.config", mock_config)

        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello world"
        mock_choice.logprobs = None
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        gen = ResponseGenerator()
        messages = [ChatMessage.from_user("hi")]
        result = gen.run(prompt=messages)
        assert result["reply"] == "Hello world"

    @patch("document_common.components.base.assembly.OpenAI")
    def test_run_log_prob(self, mock_openai, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.llm = MagicMock(
            model="gpt-4",
            api_key="test-key",
            temperature=0.0,
            max_output_tokens=1000,
            timeout=30.0,
            base_url="http://localhost:8000/v1",
            top_p=0.95,
        )
        monkeypatch.setattr("document_common.components.base.assembly.config", mock_config)

        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello"
        mock_choice.logprobs = {"content": []}
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        gen = ResponseGenerator()
        messages = [ChatMessage.from_user("hi")]
        result = gen.run(prompt=messages, log_prob=True)
        assert result["reply"] == "Hello"
        assert "log_prob" in result

    def test_run_structured_with_custom_model(self, mock_logger):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        mock_doc = MagicMock()
        mock_doc.content = '{"name": "test"}'
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=mock_doc)]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion

        with patch("document_common.components.base.assembly.OpenAI", return_value=mock_client):
            gen = ResponseGenerator()
            messages = [ChatMessage.from_user("hi")]
            result = gen.run(prompt=messages, base_model=TestModel)
            assert isinstance(result["reply"], TestModel)
            assert result["reply"].name == "test"

    def test_run_structured_parse_failure(self, mock_logger):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        mock_doc = MagicMock()
        mock_doc.content = "not json"
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=mock_doc)]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion

        with patch("document_common.components.base.assembly.OpenAI", return_value=mock_client):
            gen = ResponseGenerator()
            messages = [ChatMessage.from_user("hi")]
            with pytest.raises(Exception):
                gen.run(prompt=messages, base_model=TestModel)


class TestAsyncResponseGenerator:
    @patch("document_common.components.base.assembly.AsyncOpenAI")
    def test_init(self, mock_async_openai, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.llm = MagicMock(
            model="gpt-4",
            api_key="test-key",
            temperature=0.5,
            max_output_tokens=500,
            timeout=30.0,
            base_url="http://localhost:8000/v1",
            top_p=0.95,
        )
        monkeypatch.setattr("document_common.components.base.assembly.config", mock_config)

        gen = AsyncResponseGenerator()
        assert gen.model == "gpt-4"

    @pytest.mark.asyncio
    @patch("document_common.components.base.assembly.AsyncOpenAI")
    async def test_run_unstructured(self, mock_async_openai, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.llm = MagicMock(
            model="gpt-4",
            api_key="test-key",
            temperature=0.0,
            max_output_tokens=1000,
            timeout=30.0,
            base_url="http://localhost:8000/v1",
            top_p=0.95,
        )
        monkeypatch.setattr("document_common.components.base.assembly.config", mock_config)

        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Async reply"
        mock_completion.choices = [mock_choice]

        async def fake_create(*args, **kwargs):
            return mock_completion

        mock_client.chat.completions.create = fake_create
        mock_async_openai.return_value = mock_client

        gen = AsyncResponseGenerator()
        messages = [ChatMessage.from_user("hi")]
        result = await gen.run_async(prompt=messages)
        assert result["reply"] == "Async reply"


class TestStructuredAssembly:
    @patch("document_common.components.base.assembly.PromptProvider")
    @patch("document_common.components.base.assembly.ResponseGenerator")
    def test_init(self, mock_gen_cls, mock_prompt_provider, mock_logger):
        mock_prompt_provider.return_value.get_prompt.return_value = [ChatMessage.from_system("sys")]
        mock_gen_cls.return_value = MagicMock()

        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        sg = StructuredAssembly(prompt_name="test_prompt", base_model=TestModel)
        assert sg.prompt_name == "test_prompt"
        assert sg.base_model == TestModel

    @patch("document_common.components.base.assembly.PromptProvider")
    @patch("document_common.components.base.assembly.ResponseGenerator")
    def test_run_with_documents(self, mock_gen_cls, mock_prompt_provider, mock_logger):
        mock_prompt_provider.return_value.get_prompt.return_value = [ChatMessage.from_system("sys")]
        mock_gen = MagicMock()
        mock_gen.run.return_value = {"reply": MagicMock()}
        mock_gen_cls.return_value = mock_gen

        from pydantic import BaseModel
        from haystack import Document

        class TestModel(BaseModel):
            name: str

        sg = StructuredAssembly(prompt_name="test", base_model=TestModel)
        docs = [Document(content="test")]
        sg.generator.run = mock_gen
        mock_gen.run.return_value = {"reply": MagicMock(spec=TestModel, name="test")}

        result = sg.run(documents=docs, inputs={"key": "value"})
        assert "structured_reply" in result

    @patch("document_common.components.base.assembly.PromptProvider")
    @patch("document_common.components.base.assembly.ResponseGenerator")
    def test_run_overrides_prompt_name(self, mock_gen_cls, mock_prompt_provider, mock_logger):
        mock_prompt_provider.return_value.get_prompt.return_value = [ChatMessage.from_system("sys")]

        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        sg = StructuredAssembly(prompt_name="default", base_model=TestModel)
        sg.generator = MagicMock()
        sg.generator.run.return_value = {"reply": MagicMock()}
        mock_prompt_provider.return_value.get_prompt.return_value = [ChatMessage.from_system("sys")]

        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        sg.base_model = TestModel
        sg.generator.run.return_value = {"reply": MagicMock(spec=TestModel)}
        sg.run(prompt_name="override", inputs={})

        mock_prompt_provider.return_value.get_prompt.assert_called_with("override")


class TestTextAssembly:
    @patch("document_common.components.base.assembly.PromptProvider")
    @patch("document_common.components.base.assembly.ResponseGenerator")
    def test_init(self, mock_gen_cls, mock_prompt_provider, mock_logger):
        tg = TextAssembly(prompt_name="test")
        assert tg.prompt_name == "test"

    @patch("document_common.components.base.assembly.PromptProvider")
    @patch("document_common.components.base.assembly.ResponseGenerator")
    def test_run_with_list_documents(self, mock_gen_cls, mock_prompt_provider, mock_logger):
        mock_prompt_provider.return_value.get_prompt.return_value = [ChatMessage.from_system("sys")]
        mock_gen = MagicMock()
        mock_gen.run.return_value = {"reply": "text output"}

        tg = TextAssembly(prompt_name="test")
        tg.generator = mock_gen
        from haystack import Document
        result = tg.run(documents=[Document(content="test")])
        assert result["llm_answer"] == "text output"

    @patch("document_common.components.base.assembly.PromptProvider")
    @patch("document_common.components.base.assembly.ResponseGenerator")
    def test_run_with_single_document(self, mock_gen_cls, mock_prompt_provider, mock_logger):
        mock_prompt_provider.return_value.get_prompt.return_value = [ChatMessage.from_system("sys")]
        mock_gen = MagicMock()
        mock_gen.run.return_value = {"reply": "single doc output"}

        tg = TextAssembly(prompt_name="test")
        tg.generator = mock_gen
        from haystack import Document
        result = tg.run(documents=Document(content="test"))
        assert result["llm_answer"] == "single doc output"

    @patch("document_common.components.base.assembly.PromptProvider")
    @patch("document_common.components.base.assembly.ResponseGenerator")
    def test_run_overrides_prompt_name(self, mock_gen_cls, mock_prompt_provider, mock_logger):
        mock_prompt_provider.return_value.get_prompt.return_value = [ChatMessage.from_system("sys")]
        mock_gen = MagicMock()
        mock_gen.run.return_value = {"reply": "output"}

        tg = TextAssembly(prompt_name="default")
        tg.generator = mock_gen
        result = tg.run(prompt_name="override")
        assert result["llm_answer"] == "output"

    @patch("document_common.components.base.assembly.PromptProvider")
    @patch("document_common.components.base.assembly.ResponseGenerator")
    def test_run_with_fixed_inputs(self, mock_gen_cls, mock_prompt_provider, mock_logger):
        mock_prompt_provider.return_value.get_prompt.return_value = [ChatMessage.from_system("sys")]
        mock_gen = MagicMock()
        mock_gen.run.return_value = {"reply": "output"}

        tg = TextAssembly(prompt_name="test", fixed_inputs={"key": "val"})
        tg.generator = mock_gen
        result = tg.run(inputs={"other": "data"})
        assert result["llm_answer"] == "output"
