import time
from typing import Any, Callable, Dict, List, Tuple

from rapidfuzz import fuzz


def normalize_scores(scores: List[float], min_val: float = 0.0, max_val: float = 1.0) -> List[float]:
    """
    Normalize scores to a specific range.

    Args:
        scores: List of scores to normalize
        min_val: Minimum value for normalized range
        max_val: Maximum value for normalized range

    Returns:
        List of normalized scores
    """
    if not scores:
        return []

    score_min = min(scores)
    score_max = max(scores)

    if score_max == score_min:
        return [min_val] * len(scores)

    normalized = [min_val + (score - score_min) * (max_val - min_val) / (score_max - score_min) for score in scores]
    return normalized


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split items into batches of specified size.

    Args:
        items: List of items to batch
        batch_size: Maximum size of each batch

    Returns:
        List of batches
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def fuzzy_match_texts(
    query: str,
    candidates: List[str],
    threshold: float = 85.0,
    scorer: str = "token_sort_ratio",
    min_length: int = 10,
) -> List[Tuple[int, float, str]]:
    """
    Perform fuzzy string matching between query and candidates using RapidFuzz.

    Args:
        query: Query text to match against
        candidates: List of candidate texts
        threshold: Minimum fuzzy match score (0-100)
        scorer: RapidFuzz scoring function name
        min_length: Minimum text length to consider

    Returns:
        List of (index, score, method) tuples for matches above threshold
    """
    if len(query.strip()) < min_length:
        return []

    scorer_func = getattr(fuzz, scorer, fuzz.token_sort_ratio)

    matches = []
    for idx, candidate in enumerate(candidates):
        if len(candidate.strip()) < min_length:
            continue

        score = scorer_func(query, candidate)
        if score >= threshold:
            matches.append((idx, score / 100.0, "fuzzy"))

    return sorted(matches, key=lambda x: x[1], reverse=True)


def adaptive_threshold_search(
    score_func: Callable,
    base_threshold: float,
    decrement: float = 0.1,
    min_threshold: float = 0.5,
    max_retries: int = 2,
) -> Tuple[List[Any], float]:
    """
    Execute score function with adaptive threshold lowering until matches found.

    Args:
        score_func: Function that returns matches given a threshold
        base_threshold: Starting threshold value
        decrement: Amount to lower threshold per retry
        min_threshold: Minimum allowed threshold
        max_retries: Maximum retry attempts

    Returns:
        Tuple of (matches, final_threshold_used)
    """
    current_threshold = base_threshold
    attempts = 0

    while attempts <= max_retries:
        matches = score_func(current_threshold)

        if matches or current_threshold <= min_threshold:
            return matches, current_threshold

        current_threshold = max(current_threshold - decrement, min_threshold)
        attempts += 1

    return [], current_threshold


def merge_reranker_fuzzy_results(
    reranker_matches: List[Tuple[int, float, str]],
    fuzzy_matches: List[Tuple[int, float, str]],
    prefer_reranker: bool = True,
) -> List[Tuple[int, float, str]]:
    """
    Merge results from reranker and fuzzy matching, avoiding duplicates.

    Args:
        reranker_matches: Matches from reranker (index, score, method)
        fuzzy_matches: Matches from fuzzy matching (index, score, method)
        prefer_reranker: If True, use reranker score when index appears in both

    Returns:
        Merged list of (index, score, method) tuples sorted by score
    """
    seen_indices = {}

    for idx, score, method in reranker_matches:
        seen_indices[idx] = (score, method)

    for idx, score, method in fuzzy_matches:
        if idx not in seen_indices or not prefer_reranker:
            seen_indices[idx] = (score, method)

    merged = [(idx, score, method) for idx, (score, method) in seen_indices.items()]
    return sorted(merged, key=lambda x: x[1], reverse=True)


def compute_batch_statistics(batch_times: List[float]) -> Dict[str, float]:
    """
    Compute statistics for batch processing times.

    Args:
        batch_times: List of batch processing times in seconds

    Returns:
        Dictionary with min, max, mean, and total time
    """
    if not batch_times:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "total": 0.0, "count": 0}

    return {
        "min": min(batch_times),
        "max": max(batch_times),
        "mean": sum(batch_times) / len(batch_times),
        "total": sum(batch_times),
        "count": len(batch_times),
    }


class SimpleCache:
    """Simple LRU cache with TTL for source sentences"""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_order: List[str] = []

    def get(self, key: str) -> Any:
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]

        if time.time() - timestamp > self.ttl_seconds:
            self._evict(key)
            return None

        self._access_order.remove(key)
        self._access_order.append(key)

        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = (value, time.time())
        self._access_order.append(key)

    def _evict(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        return len(self._cache)
