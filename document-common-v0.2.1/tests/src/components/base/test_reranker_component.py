from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from document_common.components.base.reranker import DocumentReranker, RerankMatch, RerankerAPIClient


class TestRerankMatch:
    def test_valid_rerank_match(self):
        m = RerankMatch(index=0, score=0.95, text="hello", method="reranker")
        assert m.index == 0
        assert m.score == 0.95
        assert m.text == "hello"
        assert m.method == "reranker"


class TestRerankerAPIClient:
    def test_init_with_explicit_overrides(self, mock_config, monkeypatch):
        """Test that explicit parameters override config values."""
        mock_config.get_current_config.return_value.reranker = MagicMock(
            api=MagicMock(
                url="http://config-url:8000/rerank",
                api_key_env_var="config-key",
                model="config-model",
                timeout_seconds=30,
                max_retries=5,
                retry_delay=5.0,
            )
        )
        monkeypatch.setattr("document_common.components.base.reranker.config", mock_config)

        client = RerankerAPIClient(
            api_url="http://override:9000/rerank",
            api_key="override-key",
            model="override-model",
            timeout=15,
            max_retries=3,
            retry_delay=2.0,
        )
        assert client.api_url == "http://override:9000/rerank"
        assert client.api_key == "override-key"
        assert client.model == "override-model"
        assert client.timeout == 15
        assert client.max_retries == 3
        assert client.retry_delay == 2.0
        assert client.length_limit == 70000

    def test_init_uses_config_when_no_overrides(self, mock_config, monkeypatch):
        """Test that config values are used when no overrides are provided."""
        mock_config.get_current_config.return_value.reranker = MagicMock(
            api=MagicMock(
                url="http://config-url:8000/rerank",
                api_key_env_var="config-key",
                model="config-model",
                timeout_seconds=30,
                max_retries=5,
                retry_delay=5.0,
            )
        )
        monkeypatch.setattr("document_common.components.base.reranker.config", mock_config)

        client = RerankerAPIClient()
        assert client.api_url == "http://config-url:8000/rerank"
        assert client.api_key == "config-key"
        assert client.model == "config-model"
        assert client.timeout == 30
        assert client.max_retries == 5
        assert client.retry_delay == 5.0

    def test_init_partial_overrides(self, mock_config, monkeypatch):
        """Test that partial overrides work (some from config, some explicit)."""
        mock_config.get_current_config.return_value.reranker = MagicMock(
            api=MagicMock(
                url="http://config-url:8000/rerank",
                api_key_env_var="config-key",
                model="config-model",
                timeout_seconds=30,
                max_retries=5,
                retry_delay=5.0,
            )
        )
        monkeypatch.setattr("document_common.components.base.reranker.config", mock_config)

        client = RerankerAPIClient(model="override-model")
        assert client.api_url == "http://config-url:8000/rerank"  # from config
        assert client.api_key == "config-key"  # from config
        assert client.model == "override-model"  # overridden
        assert client.timeout == 30  # from config
        assert client.max_retries == 5  # from config

    def test_rerank_empty_texts(self, mock_config, monkeypatch):
        """Test that empty texts returns empty list."""
        mock_config.get_current_config.return_value.reranker = MagicMock(
            api=MagicMock(
                url="http://localhost:8000/rerank",
                api_key_env_var="test-key",
                model="rerank-model",
                timeout_seconds=10,
                max_retries=2,
                retry_delay=1.0,
            )
        )
        monkeypatch.setattr("document_common.components.base.reranker.config", mock_config)

        client = RerankerAPIClient()
        result = client.rerank(query="test", texts=[])
        assert result == []

    @patch("document_common.components.base.reranker.requests.post")
    def test_rerank_success(self, mock_post, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.reranker = MagicMock(
            api=MagicMock(
                url="http://localhost:8000/rerank",
                api_key_env_var="test-key",
                model="rerank-model",
                timeout_seconds=10,
                max_retries=2,
                retry_delay=1.0,
            )
        )
        monkeypatch.setattr("document_common.components.base.reranker.config", mock_config)

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.5}]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = RerankerAPIClient()
        result = client.rerank(query="test", texts=["doc1", "doc2"])
        assert result == [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.5}]

    @patch("document_common.components.base.reranker.requests.post")
    def test_rerank_timeout(self, mock_post, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.reranker = MagicMock(
            api=MagicMock(
                url="http://localhost:8000/rerank",
                api_key_env_var="test-key",
                model="rerank-model",
                timeout_seconds=10,
                max_retries=0,
                retry_delay=0.01,
            )
        )
        monkeypatch.setattr("document_common.components.base.reranker.config", mock_config)

        mock_post.side_effect = Exception("timeout")

        client = RerankerAPIClient()
        result = client.rerank(query="test", texts=["doc1"])
        assert result is None

    @patch("document_common.components.base.reranker.requests.post")
    def test_rerank_truncates_long_texts(self, mock_post, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.reranker = MagicMock(
            api=MagicMock(
                url="http://localhost:8000/rerank",
                api_key_env_var="test-key",
                model="rerank-model",
                timeout_seconds=10,
                max_retries=2,
                retry_delay=0.01,
            )
        )
        monkeypatch.setattr("document_common.components.base.reranker.config", mock_config)

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"index": 0, "relevance_score": 0.9}]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        long_text = "word " * 40000  # Very long text
        client = RerankerAPIClient()
        result = client.rerank(query="test", texts=[long_text, "short"])
        assert result is not None

    @patch("document_common.components.base.reranker.requests.post")
    def test_rerank_with_explicit_override(self, mock_post, mock_config, monkeypatch):
        """Test rerank with explicitly passed params (no config needed)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"index": 0, "relevance_score": 0.85}]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = RerankerAPIClient(
            api_url="http://explicit:8000/rerank",
            api_key="explicit-key",
            model="explicit-model",
            timeout=5,
            max_retries=1,
            retry_delay=0.5,
        )
        result = client.rerank(query="test", texts=["doc1"])
        assert result == [{"index": 0, "relevance_score": 0.85}]
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["timeout"] == 5


class TestDocumentReranker:
    def test_init_disabled(self, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.reranker.document_mode.enabled = False
        monkeypatch.setattr("document_common.components.base.reranker.config", mock_config)

        reranker = DocumentReranker()
        assert reranker.enabled is False

    def test_run_enabled_no_documents(self, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.reranker.document_mode.enabled = False
        monkeypatch.setattr("document_common.components.base.reranker.config", mock_config)

        reranker = DocumentReranker()
        result = reranker.run(query="test", documents=[])
        assert result["documents"] == []

    def test_run_disabled_returns_original(self, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.reranker.document_mode.enabled = False
        monkeypatch.setattr("document_common.components.base.reranker.config", mock_config)

        reranker = DocumentReranker()
        from haystack import Document
        docs = [Document(content="doc1"), Document(content="doc2")]
        result = reranker.run(query="test", documents=docs)
        assert len(result["documents"]) == 2

    @patch("document_common.components.base.reranker.requests.post")
    def test_run_with_successful_rerank(self, mock_post, mock_config, monkeypatch):
        doc_mode = MagicMock()
        doc_mode.enabled = True
        doc_mode.top_k_output = 2
        doc_mode.top_k_input = 100
        doc_mode.min_score = 0.0
        doc_mode.normalize_scores = False
        doc_mode.fallback_to_embedding = False
        mock_config.get_current_config.return_value.reranker.document_mode = doc_mode
        mock_config.get_current_config.return_value.reranker.api = MagicMock(
            url="http://localhost:8000/rerank",
            api_key_env_var="test-key",
            model="rerank-model",
            timeout_seconds=10,
            max_retries=2,
            retry_delay=1.0,
        )
        monkeypatch.setattr("document_common.components.base.reranker.config", mock_config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.7},
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        from haystack import Document
        docs = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
        reranker = DocumentReranker()
        assert reranker.enabled is True, f"Reranker should be enabled, got enabled={reranker.enabled}"
        result = reranker.run(query="test", documents=docs)
        assert len(result["documents"]) == 2
        assert result["documents"][0].score == 0.9

    @patch("document_common.components.base.reranker.requests.post")
    def test_run_fallback_on_api_failure(self, mock_post, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.reranker.document_mode.enabled = True
        mock_config.get_current_config.return_value.reranker.document_mode.top_k_output = 2
        mock_config.get_current_config.return_value.reranker.document_mode.min_score = 0.0
        mock_config.get_current_config.return_value.reranker.document_mode.normalize_scores = False
        mock_config.get_current_config.return_value.reranker.document_mode.fallback_to_embedding = True
        mock_config.get_current_config.return_value.reranker.api = MagicMock(
            url="http://localhost:8000/rerank",
            api_key_env_var="test-key",
            model="rerank-model",
            timeout_seconds=10,
            max_retries=2,
            retry_delay=1.0,
        )
        monkeypatch.setattr("document_common.components.base.reranker.config", mock_config)

        mock_post.side_effect = Exception("API failure")

        from haystack import Document
        docs = [Document(content="doc1", score=0.5), Document(content="doc2", score=0.8)]
        reranker = DocumentReranker()
        result = reranker.run(query="test", documents=docs)
        assert len(result["documents"]) == 2

    @patch("document_common.components.base.reranker.requests.post")
    def test_run_min_score_filters(self, mock_post, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.reranker.document_mode.enabled = True
        mock_config.get_current_config.return_value.reranker.document_mode.top_k_output = 10
        mock_config.get_current_config.return_value.reranker.document_mode.min_score = 0.8
        mock_config.get_current_config.return_value.reranker.document_mode.normalize_scores = False
        mock_config.get_current_config.return_value.reranker.document_mode.fallback_to_embedding = False
        mock_config.get_current_config.return_value.reranker.api = MagicMock(
            url="http://localhost:8000/rerank",
            api_key_env_var="test-key",
            model="rerank-model",
            timeout_seconds=10,
            max_retries=2,
            retry_delay=1.0,
        )
        monkeypatch.setattr("document_common.components.base.reranker.config", mock_config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.5},
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        from haystack import Document
        docs = [Document(content="doc1"), Document(content="doc2")]
        reranker = DocumentReranker()
        result = reranker.run(query="test", documents=docs)
        assert len(result["documents"]) == 1
        assert result["documents"][0].score == 0.9

    def test_run_custom_top_k_output(self, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.reranker.document_mode.enabled = False
        monkeypatch.setattr("document_common.components.base.reranker.config", mock_config)

        from haystack import Document
        docs = [Document(content="doc1")]
        reranker = DocumentReranker()
        result = reranker.run(query="test", documents=docs, top_k_output=5)
        assert len(result["documents"]) == 1