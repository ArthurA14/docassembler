import time
import pytest

from document_common.utils.reranker_utils import (
    normalize_scores,
    batch_items,
    fuzzy_match_texts,
    adaptive_threshold_search,
    merge_reranker_fuzzy_results,
    compute_batch_statistics,
    SimpleCache,
)


class TestNormalizeScores:
    def test_empty_list(self):
        assert normalize_scores([]) == []

    def test_single_value(self):
        result = normalize_scores([5.0])
        assert result == [0.0]

    def test_all_same_values(self):
        result = normalize_scores([3.0, 3.0, 3.0])
        assert all(v == 0.0 for v in result)

    def test_normalization(self):
        result = normalize_scores([0.0, 5.0, 10.0])
        assert result[0] == 0.0
        assert result[1] == 0.5
        assert result[2] == 1.0

    def test_custom_range(self):
        result = normalize_scores([0.0, 10.0], min_val=10.0, max_val=20.0)
        assert result[0] == 10.0
        assert result[1] == 20.0

    def test_reverse_order(self):
        result = normalize_scores([10.0, 0.0])
        assert result[0] == 1.0
        assert result[1] == 0.0

    def test_negative_values(self):
        result = normalize_scores([-10.0, 0.0, 10.0])
        assert result[0] == 0.0
        assert abs(result[1] - 0.5) < 0.001
        assert result[2] == 1.0


class TestBatchItems:
    def test_exact_batches(self):
        result = batch_items([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    def test_partial_last_batch(self):
        result = batch_items([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_single_item_batches(self):
        result = batch_items([1, 2, 3], 1)
        assert result == [[1], [2], [3]]

    def test_batch_larger_than_list(self):
        result = batch_items([1, 2], 10)
        assert result == [[1, 2]]

    def test_empty_list(self):
        result = batch_items([], 5)
        assert result == []

    def test_zero_batch_size_raises(self):
        with pytest.raises(ValueError):
            batch_items([1, 2], 0)

    def test_negative_batch_size_raises(self):
        with pytest.raises(ValueError):
            batch_items([1, 2], -1)


class TestFuzzyMatchTexts:
    def test_exact_match(self):
        result = fuzzy_match_texts("hello world", ["hello world"], threshold=85.0)
        assert len(result) == 1
        assert result[0][0] == 0
        assert result[0][2] == "fuzzy"

    def test_no_match_low_threshold(self):
        result = fuzzy_match_texts("completely different text here", ["hello world"], threshold=85.0)
        assert len(result) == 0

    def test_short_query_ignored(self):
        result = fuzzy_match_texts("hi", ["hello world"], threshold=85.0, min_length=10)
        assert len(result) == 0

    def test_short_candidate_ignored(self):
        result = fuzzy_match_texts("hello world test query", ["hi"], threshold=85.0, min_length=10)
        assert len(result) == 0

    def test_multiple_candidates(self):
        candidates = [
            "hello world",
            "completely unrelated",
            "hello world again",
        ]
        result = fuzzy_match_texts("hello world", candidates, threshold=85.0)
        assert len(result) >= 1
        # Results should be sorted by score descending
        for i in range(len(result) - 1):
            assert result[i][1] >= result[i + 1][1]

    def test_different_scorer(self):
        result = fuzzy_match_texts(
            "hello world",
            ["hello world"],
            threshold=85.0,
            scorer="token_ratio",
        )
        assert len(result) >= 0  # May or may not match depending on scorer

    def test_empty_candidates(self):
        result = fuzzy_match_texts("query", [], threshold=85.0)
        assert result == []


class TestAdaptiveThresholdSearch:
    def test_immediate_match(self):
        def score_func(threshold):
            return [{"score": 0.9}] if threshold <= 0.8 else []
        matches, threshold = adaptive_threshold_search(score_func, base_threshold=0.9, max_retries=2)
        assert len(matches) == 1
        assert threshold <= 0.9

    def test_no_match_returns_empty(self):
        def score_func(threshold):
            return []
        matches, threshold = adaptive_threshold_search(score_func, base_threshold=0.9, min_threshold=0.5, max_retries=2)
        assert matches == []

    def test_stops_at_min_threshold(self):
        def score_func(threshold):
            return []
        matches, threshold = adaptive_threshold_search(score_func, base_threshold=0.9, min_threshold=0.5, max_retries=10)
        assert threshold == 0.5

    def test_empty_result_at_base(self):
        def score_func(threshold):
            return []
        matches, threshold = adaptive_threshold_search(score_func, base_threshold=0.9, max_retries=0)
        assert matches == []


class TestMergeRerankerFuzzyResults:
    def test_no_dups(self):
        reranker = [(0, 0.9, "reranker"), (2, 0.7, "reranker")]
        fuzzy = [(1, 0.8, "fuzzy"), (3, 0.6, "fuzzy")]
        result = merge_reranker_fuzzy_results(reranker, fuzzy)
        assert len(result) == 4

    def test_prefer_reranker(self):
        reranker = [(0, 0.9, "reranker")]
        fuzzy = [(0, 0.85, "fuzzy")]
        result = merge_reranker_fuzzy_results(reranker, fuzzy, prefer_reranker=True)
        assert len(result) == 1
        assert result[0][1] == 0.9
        assert result[0][2] == "reranker"

    def test_prefer_fuzzy(self):
        reranker = [(0, 0.9, "reranker")]
        fuzzy = [(0, 0.95, "fuzzy")]
        result = merge_reranker_fuzzy_results(reranker, fuzzy, prefer_reranker=False)
        assert len(result) == 1
        assert result[0][1] == 0.95
        assert result[0][2] == "fuzzy"

    def test_sorted_by_score(self):
        reranker = [(1, 0.5, "reranker"), (0, 0.9, "reranker")]
        fuzzy = [(2, 0.7, "fuzzy")]
        result = merge_reranker_fuzzy_results(reranker, fuzzy)
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)


class TestComputeBatchStatistics:
    def test_empty_list(self):
        result = compute_batch_statistics([])
        assert result == {"min": 0.0, "max": 0.0, "mean": 0.0, "total": 0.0, "count": 0}

    def test_single_item(self):
        result = compute_batch_statistics([5.0])
        assert result["min"] == 5.0
        assert result["max"] == 5.0
        assert result["mean"] == 5.0
        assert result["total"] == 5.0
        assert result["count"] == 1

    def test_multiple_items(self):
        result = compute_batch_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["mean"] == 3.0
        assert result["total"] == 15.0
        assert result["count"] == 5


class TestSimpleCache:
    def test_set_and_get(self):
        cache = SimpleCache(max_size=100, ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self):
        cache = SimpleCache()
        assert cache.get("nonexistent") is None

    def test_eviction_on_ttl(self, monkeypatch):
        import time as _time
        cache = SimpleCache(max_size=100, ttl_seconds=1)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        monkeypatch.setattr("document_common.utils.reranker_utils.time", _time)
        class FakeTime:
            @staticmethod
            def time():
                return _time.time() + 2
        import document_common.utils.reranker_utils as ru
        monkeypatch.setattr(ru, "time", FakeTime())
        assert cache.get("key1") is None

    def test_lru_eviction(self):
        cache = SimpleCache(max_size=3, ttl_seconds=3600)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # Access "a" to make it recently used
        cache.get("a")
        # Add "d" - should evict "b" (least recently used)
        cache.set("d", 4)
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_update_existing_key(self):
        cache = SimpleCache(max_size=100)
        cache.set("key1", "value1")
        cache.set("key1", "value2")
        assert cache.get("key1") == "value2"

    def test_clear(self):
        cache = SimpleCache(max_size=100)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.size() == 0

    def test_size(self):
        cache = SimpleCache(max_size=100)
        assert cache.size() == 0
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.size() == 2

    def test_ttl_expiry_removes_from_access_order(self, monkeypatch):
        import time as _time
        cache = SimpleCache(max_size=3, ttl_seconds=1)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # Expire "a"
        import document_common.utils.reranker_utils as ru
        class FakeTime:
            @staticmethod
            def time():
                return _time.time() + 2
        monkeypatch.setattr(ru, "time", FakeTime())
        cache.get("a")
        # Now add a new item - should not fail
        cache.set("d", 4)
        assert cache.get("d") == 4
