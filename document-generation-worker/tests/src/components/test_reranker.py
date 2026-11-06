from unittest.mock import MagicMock, patch

from haystack import Document

from src.components.base.reranker import DocumentReranker, RerankerAPIClient
from src.components.base.text_reranker import TextPairReranker
from src.utils.config.utils import DotDict
from src.utils.reranker_utils import SimpleCache, fuzzy_match_texts, normalize_scores


class TestRerankerAPIClient:
    def test_successful_rerank(self):
        client = RerankerAPIClient(
            api_url="http://test.api/v1/rerank",
            api_key="test-key",
            model="test-model",
            timeout=10,
            max_retries=2,
            retry_delay=0.1,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.80},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            results = client.rerank(query="test query", texts=["text1", "text2"])

        assert len(results) == 2
        assert results[0]["relevance_score"] == 0.95

    def test_api_timeout(self):
        client = RerankerAPIClient(
            api_url="http://test.api/v1/rerank",
            api_key="test-key",
            model="test-model",
            timeout=1,
            max_retries=1,
            retry_delay=0.1,
        )

        with patch("requests.post", side_effect=Exception("Timeout")):
            results = client.rerank(query="test", texts=["text"])

        assert results is None


class TestDocumentReranker:
    @patch("src.components.base.reranker.config")
    @patch("os.getenv")
    def test_rerank_documents_success(self, mock_getenv, mock_config):
        mock_getenv.return_value = "test-key"
        mock_config.get_current_config.return_value.reranker = DotDict({
            "api": {
                "url": "http://test/v1/rerank",
                "api_key_env_var": "QWEN_API_KEY",
                "model": "test-model",
                "timeout_seconds": 10,
                "max_retries": 2,
                "retry_delay": 1.0,
            },
            "document_mode": {
                "enabled": True,
                "top_k_input": 100,
                "top_k_output": 50,
                "min_score": 0.0,
                "normalize_scores": True,
                "fallback_to_embedding": True,
            },
        })

        reranker = DocumentReranker()
        reranker.enabled = True

        mock_api_response = [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.80},
        ]

        reranker.api_client.rerank = MagicMock(return_value=mock_api_response)

        docs = [
            Document(content="low relevance doc", score=0.5),
            Document(content="high relevance doc", score=0.6),
        ]

        result = reranker.run(query="test query", documents=docs)

        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "high relevance doc"
        assert result["documents"][0].score == 1.0

    @patch("src.components.base.reranker.config")
    @patch("os.getenv")
    def test_fallback_on_api_failure(self, mock_getenv, mock_config):
        mock_getenv.return_value = "test-key"
        mock_config.get_current_config.return_value.reranker = DotDict({
            "api": {
                "url": "http://test/v1/rerank",
                "api_key_env_var": "QWEN_API_KEY",
                "model": "test-model",
                "timeout_seconds": 10,
                "max_retries": 2,
                "retry_delay": 1.0,
            },
            "document_mode": {
                "enabled": True,
                "top_k_input": 100,
                "top_k_output": 50,
                "min_score": 0.0,
                "normalize_scores": True,
                "fallback_to_embedding": True,
            },
        })

        reranker = DocumentReranker()
        reranker.enabled = True
        reranker.api_client.rerank = MagicMock(return_value=None)

        docs = [
            Document(content="doc1", score=0.5),
            Document(content="doc2", score=0.9),
        ]

        result = reranker.run(query="test", documents=docs)

        assert len(result["documents"]) == 2
        assert result["documents"][0].score == 0.9


class TestTextPairReranker:
    @patch("src.components.base.reranker.config")
    @patch("os.getenv")
    def test_score_text_pairs(self, mock_getenv, mock_config):
        mock_getenv.return_value = "test-key"
        mock_config.get_current_config.return_value.reranker = DotDict({
            "api": {
                "url": "http://test/v1/rerank",
                "api_key_env_var": "QWEN_API_KEY",
                "model": "test-model",
                "timeout_seconds": 10,
                "max_retries": 2,
                "retry_delay": 1.0,
            },
            "citation_mode": {
                "enabled": True,
                "batch_size": 32,
                "base_threshold": 0.75,
                "normalize_scores": False,
                "adaptive_retry": {"enabled": True, "max_retries": 2, "threshold_decrement": 0.1, "min_threshold": 0.5},
                "fuzzy_fallback": {
                    "enabled": True,
                    "threshold": 85.0,
                    "scorer": "token_sort_ratio",
                    "min_length": 10,
                },
                "cache": {"enabled": False},
            },
        })

        reranker = TextPairReranker()
        reranker.enabled = True

        mock_api_response = [
            {"index": 0, "relevance_score": 0.85},
            {"index": 1, "relevance_score": 0.78},
        ]
        reranker.api_client.rerank = MagicMock(return_value=mock_api_response)

        candidates = ["matching text", "another match"]
        result = reranker.run(query="test query", candidates=candidates)

        assert len(result["matches"]) == 2
        assert result["matches"][0].score == 0.85
        assert result["matches"][0].method == "reranker"

    @patch("src.components.base.reranker.config")
    @patch("os.getenv")
    def test_fuzzy_fallback(self, mock_getenv, mock_config):
        mock_getenv.return_value = "test-key"
        mock_config.get_current_config.return_value.reranker = DotDict({
            "api": {
                "url": "http://test/v1/rerank",
                "api_key_env_var": "QWEN_API_KEY",
                "model": "test-model",
                "timeout_seconds": 10,
                "max_retries": 2,
                "retry_delay": 1.0,
            },
            "citation_mode": {
                "enabled": True,
                "batch_size": 32,
                "base_threshold": 0.75,
                "normalize_scores": False,
                "adaptive_retry": {"enabled": False},
                "fuzzy_fallback": {
                    "enabled": True,
                    "threshold": 60.0,
                    "scorer": "token_sort_ratio",
                    "min_length": 5,
                },
                "cache": {"enabled": False},
            },
        })

        reranker = TextPairReranker()
        reranker.enabled = True
        reranker.api_client.rerank = MagicMock(return_value=[])

        candidates = ["test query exact match", "unrelated text"]
        result = reranker.run(query="test query", candidates=candidates)

        assert len(result["matches"]) >= 1
        assert result["matches"][0].method == "fuzzy"


class TestRerankUtils:
    def test_normalize_scores(self):
        scores = [0.5, 0.75, 1.0, 0.25]
        normalized = normalize_scores(scores)

        assert min(normalized) == 0.0
        assert max(normalized) == 1.0
        assert len(normalized) == len(scores)

    def test_fuzzy_match_texts(self):
        query = "the quick brown fox"
        candidates = ["the quick brown fox jumps", "completely different", "quick brown fox"]

        matches = fuzzy_match_texts(query, candidates, threshold=70.0, scorer="token_sort_ratio")

        assert len(matches) >= 2
        assert all(score >= 0.7 for _, score, _ in matches)
        assert all(method == "fuzzy" for _, _, method in matches)

    def test_simple_cache(self):
        cache = SimpleCache(max_size=2, ttl_seconds=3600)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"


class TestTextPairRerankerForSentences:
    """Tests for the new run_for_sentences() batch method."""

    @patch("src.components.base.reranker.config")
    @patch("os.getenv")
    def test_run_for_sentences_basic(self, mock_getenv, mock_config):
        """Test basic batch reranking for multiple sentences."""
        from src.components.citations.preprocessor import Sentence
        from src.schemas.citations import SourceMetadata

        mock_getenv.return_value = "test-key"
        mock_config.get_current_config.return_value.reranker = DotDict({
            "api": {
                "url": "http://test/v1/rerank",
                "api_key_env_var": "QWEN_API_KEY",
                "model": "test-model",
                "timeout_seconds": 10,
                "max_retries": 2,
                "retry_delay": 1.0,
            },
            "citation_mode": {
                "enabled": True,
                "batch_size": 32,
                "base_threshold": 0.75,
                "normalize_scores": False,
                "adaptive_retry": {"enabled": False},
                "fuzzy_fallback": {"enabled": False},
                "cache": {"enabled": False},
            },
        })

        reranker = TextPairReranker()
        reranker.enabled = True

        # Mock API response for 2 sentences
        def mock_rerank(query, texts):
            if "machine" in query.lower():
                return [{"index": 0, "relevance_score": 0.85}]
            else:
                return [{"index": 1, "relevance_score": 0.80}]

        reranker.api_client.rerank = MagicMock(side_effect=mock_rerank)

        sentences = [
            Sentence(
                text="Machine learning requires data",
                normalized_text="machine learning requires data",
                start_index=0,
                end_index=30,
                length=30,
            ),
            Sentence(
                text="Neural networks are complex",
                normalized_text="neural networks are complex",
                start_index=31,
                end_index=58,
                length=27,
            ),
        ]

        sources = [
            SourceMetadata(chunk_id="s1", file_id="f1", content="ML needs datasets", type="original"),
            SourceMetadata(chunk_id="s2", file_id="f2", content="DL models are hard", type="original"),
        ]

        result = reranker.run_for_sentences(sentences, sources, top_k=5)

        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result
        assert len(result[0]) > 0  # Sentence 0 has matches
        assert len(result[1]) > 0  # Sentence 1 has matches

    @patch("src.components.base.reranker.config")
    @patch("os.getenv")
    def test_run_for_sentences_top_k_limiting(self, mock_getenv, mock_config):
        """Test that run_for_sentences respects top-K limit."""
        from src.components.citations.preprocessor import Sentence
        from src.schemas.citations import SourceMetadata

        mock_getenv.return_value = "test-key"
        mock_config.get_current_config.return_value.reranker = DotDict({
            "api": {
                "url": "http://test/v1/rerank",
                "api_key_env_var": "QWEN_API_KEY",
                "model": "test-model",
                "timeout_seconds": 10,
                "max_retries": 2,
                "retry_delay": 1.0,
            },
            "citation_mode": {
                "enabled": True,
                "batch_size": 32,
                "base_threshold": 0.5,  # Low threshold to get many matches
                "normalize_scores": False,
                "adaptive_retry": {"enabled": False},
                "fuzzy_fallback": {"enabled": False},
                "cache": {"enabled": False},
            },
        })

        reranker = TextPairReranker()
        reranker.enabled = True

        # Mock API to return many matches
        mock_api_response = [
            {"index": i, "relevance_score": 0.95 - (i * 0.05)} for i in range(20)
        ]
        reranker.api_client.rerank = MagicMock(return_value=mock_api_response)

        sentences = [
            Sentence(
                text="Test query", normalized_text="test query", start_index=0, end_index=10, length=10
            )
        ]

        sources = [SourceMetadata(chunk_id=f"s{i}", file_id=f"f{i}", content=f"text{i}", type="original") for i in range(20)]

        result = reranker.run_for_sentences(sentences, sources, top_k=5)

        # Should only have 5 matches even though API returned 20
        assert len(result[0]) <= 5
        # Top matches should have highest scores
        if len(result[0]) > 1:
            scores = [m.score for m in result[0]]
            assert scores == sorted(scores, reverse=True)

    @patch("src.components.base.reranker.config")
    @patch("os.getenv")
    def test_run_for_sentences_no_matches(self, mock_getenv, mock_config):
        """Test handling when no matches are found."""
        from src.components.citations.preprocessor import Sentence
        from src.schemas.citations import SourceMetadata

        mock_getenv.return_value = "test-key"
        mock_config.get_current_config.return_value.reranker = DotDict({
            "api": {
                "url": "http://test/v1/rerank",
                "api_key_env_var": "QWEN_API_KEY",
                "model": "test-model",
                "timeout_seconds": 10,
                "max_retries": 2,
                "retry_delay": 1.0,
            },
            "citation_mode": {
                "enabled": True,
                "batch_size": 32,
                "base_threshold": 0.9,  # High threshold to prevent matches
                "normalize_scores": False,
                "adaptive_retry": {"enabled": False},
                "fuzzy_fallback": {"enabled": False},
                "cache": {"enabled": False},
            },
        })

        reranker = TextPairReranker()
        reranker.enabled = True
        reranker.api_client.rerank = MagicMock(return_value=[{"index": 0, "relevance_score": 0.5}])

        sentences = [
            Sentence(
                text="Test", normalized_text="test", start_index=0, end_index=4, length=4
            )
        ]
        sources = [SourceMetadata(chunk_id="s1", file_id="f1", content="text", type="original")]

        result = reranker.run_for_sentences(sentences, sources, top_k=5)

        # Should return empty list for this sentence (score below threshold)
        assert 0 in result
        assert isinstance(result[0], list)
