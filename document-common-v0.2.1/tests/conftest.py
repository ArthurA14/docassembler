import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Clear config singleton state before each test."""
    from config import config
    if hasattr(config, "_instance"):
        config._instance = None
    if hasattr(config, "_initialized"):
        config._initialized = False


@pytest.fixture(autouse=True)
def _clear_prompt_provider():
    """Clear PromptProvider singleton before each test."""
    from document_common.components.base.prompt import PromptProvider
    PromptProvider._instance = None


@pytest.fixture
def mock_logger(monkeypatch):
    """Return a mock logger for testing."""
    mock = MagicMock()
    monkeypatch.setattr("document_common.utils.logger.Logger", MagicMock(return_value=mock))
    return mock


@pytest.fixture
def mock_openai_client(monkeypatch):
    """Mock the OpenAI client to avoid API calls."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"name": "test", "context": "test context"}'
    mock_choice.logprobs = None
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    monkeypatch.setattr("document_common.components.base.assembly.OpenAI", MagicMock(return_value=mock_client))
    return mock_client


@pytest.fixture
def mock_async_openai_client(monkeypatch):
    """Mock the AsyncOpenAI client to avoid API calls."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"name": "test", "context": "test context"}'
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create = MagicMock(return_value=mock_completion)
    # Make it an awaitable
    async def fake_async(*args, **kwargs):
        return mock_completion
    mock_client.chat.completions.create = fake_async
    monkeypatch.setattr("document_common.components.base.assembly.AsyncOpenAI", MagicMock(return_value=mock_client))
    return mock_client


@pytest.fixture
def mock_config():
    """Mock the global config to return safe defaults."""
    mock_cfg = MagicMock()
    mock_cfg.get_current_config.return_value = MagicMock(
        llm=MagicMock(
            model="gpt-4",
            api_key="test-key",
            temperature=0.0,
            max_output_tokens=1000,
            timeout=30.0,
            base_url="http://localhost:8000/v1",
            top_p=0.95,
        ),
        embedder=MagicMock(
            type="api",
            url="http://localhost:8000/v1/embeddings",
            dimension=1024,
            model="tei",
        ),
        chunking=MagicMock(
            type="recursive",
            recursive={"chunk_size": 500},
        ),
        reranker=MagicMock(
            document_mode=MagicMock(enabled=False),
            api=MagicMock(
                url="http://localhost:8000/rerank",
                model="rerank-model",
                timeout_seconds=10,
                max_retries=2,
                retry_delay=1.0,
            ),
        ),
        entity_extraction=MagicMock(
            method="combined_scoring",
            prompt_name="entity_dedup",
            laj_prompt_name="entity_laj",
            combined_scoring_config=MagicMock(
                fuzzy_weight=0.4,
                name_weight=0.3,
                context_weight=0.3,
                combined_threshold=0.7,
            ),
            step_scoring_config=MagicMock(
                name_embedding_threshold=0.8,
                context_embedding_threshold=0.8,
                fuzzy_threshold=85.0,
            ),
        ),
        popcorn_ner=MagicMock(
            url="http://localhost:8000",
            endpoints=MagicMock(health="/health", ner="/extract"),
            request=MagicMock(timeout_s=60, retries=2, backoff_s=0.5),
            payload=MagicMock(default_lang="eng", text_field="text", lang_field="lang"),
        ),
    )
    return mock_cfg


@pytest.fixture
def haystack_document():
    """Create a simple Haystack Document for testing."""
    from haystack import Document
    return Document(content="test content", meta={"key": "value"})


@pytest.fixture
def mock_document_store():
    """Create a mock Haystack DocumentStore."""
    mock_ds = MagicMock()
    mock_ds.filter_documents.return_value = []
    mock_ds.write_documents.return_value = None
    mock_ds.delete_documents.return_value = None
    return mock_ds


@pytest.fixture
def mock_storage():
    """Create a mock PromptStorage for testing PromptService."""
    mock = MagicMock()
    mock.get_prompt.return_value = None
    mock.list_all_prompts.return_value = []
    mock.delete_prompt.return_value = False
    return mock


@pytest.fixture
def mock_template_storage():
    """Create a mock TemplateStorage for testing TemplateService."""
    mock = MagicMock()
    mock.get_template.return_value = None
    mock.list_all_templates.return_value = []
    mock.template_exists.return_value = False
    mock.delete_template.return_value = None
    return mock
