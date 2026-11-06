import time
from typing import Dict, List

from haystack import component
from document_common.components.base.reranker import RerankerAPIClient, RerankMatch
from document_common.schemas.citations import SourceMetadata
from document_common.utils.logger import Logger
from document_common.utils.reranker_utils import (
    SimpleCache,
    adaptive_threshold_search,
    batch_items,
    compute_batch_statistics,
    fuzzy_match_texts,
    normalize_scores,
)

from config import config


@component
class TextPairReranker:
    """
    Haystack component for scoring text pairs using reranker model.

    Used in citation matching  to find source chunks relevant to
    generated text with adaptive threshold retry and fuzzy fallback.
    """

    def __init__(self):
        self.logger = Logger().get_logger("reranker.text_pair")
        reranker_config = config.get_current_config().reranker

        if not reranker_config.citation_mode.get("enabled", True):
            self.logger.warning("Text pair reranker is disabled in configuration")
            self.enabled = False
            return

        self.enabled = True
        api_config = reranker_config.api
        cit_config = reranker_config.citation_mode

        self.api_client = RerankerAPIClient(
            api_url=api_config.get("url"),
            api_key="api_key",
            model=api_config.get("model"),
            timeout=api_config.get("timeout_seconds", 10),
            max_retries=api_config.get("max_retries", 2),
            retry_delay=api_config.get("retry_delay", 1.0),
        )

        self.batch_size = cit_config.get("batch_size", 32)
        self.base_threshold = cit_config.get("base_threshold", 0.70)
        self.normalize = cit_config.get("normalize_scores", True)

        # Adaptive retry settings
        adaptive_config = cit_config.get("adaptive_retry", {})
        self.adaptive_enabled = adaptive_config.get("enabled", True)
        self.adaptive_max_retries = adaptive_config.get("max_retries", 2)
        self.threshold_decrement = adaptive_config.get("threshold_decrement", 0.1)
        self.min_threshold = adaptive_config.get("min_threshold", 0.5)

        # Fuzzy fallback settings
        fuzzy_config = cit_config.get("fuzzy_fallback", {})
        self.fuzzy_enabled = fuzzy_config.get("enabled", True)
        self.fuzzy_threshold = fuzzy_config.get("threshold", 85.0)
        self.fuzzy_scorer = fuzzy_config.get("scorer", "token_sort_ratio")
        self.fuzzy_min_length = fuzzy_config.get("min_length", 10)

        # Cache settings
        cache_config = cit_config.get("cache", {})
        if cache_config.get("enabled", True):
            self.cache = SimpleCache(
                max_size=cache_config.get("max_size", 10000), ttl_seconds=cache_config.get("ttl_seconds", 3600)
            )
        else:
            self.cache = SimpleCache(max_size=0, ttl_seconds=0)  # Disable cache with empty instance

        self.logger.debug(f"TextPairReranker initialized (batch: {self.batch_size}, threshold: {self.base_threshold})")

    @component.output_types(matches=List[RerankMatch])
    def run(self, query: str, candidates: List[str]) -> Dict[str, List[RerankMatch]]:
        """
        Score query text against candidate texts with adaptive threshold and fallback.

        Args:
            query: Generated text to match
            candidates: List of source text candidates

        Returns:
            Dict with 'matches' key containing RerankMatch objects sorted by score
        """
        if not self.enabled:
            self.logger.debug("Reranker disabled, returning empty matches")
            return {"matches": []}

        if not candidates:
            return {"matches": []}

        # Check cache
        cache_key = f"{query}:{len(candidates)}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.logger.debug(f"Cache hit for query (candidates: {len(candidates)})")
                return {"matches": cached}

        # Batch score candidates
        matches = self._score_with_adaptive_threshold(query, candidates)

        # Convert to RerankMatch objects
        match_objects = [
            RerankMatch(index=idx, score=score, text=candidates[idx], method=method)
            for idx, score, method in matches
            if score > 0.5
        ]

        # Cache results
        if self.cache:
            self.cache.set(cache_key, match_objects)

        self.logger.debug(f"Found {len(match_objects)} matches for query (candidates: {len(candidates)})")
        return {"matches": match_objects}

    def run_for_sentences(
        self,
        sentences: List["Sentence"],
        source_documents: List[SourceMetadata],
        top_k: int = 5,
    ) -> Dict[int, List[RerankMatch]]:

        """
        Batch rerank sources for multiple sentences efficiently.

        For each sentence, finds the top-K most relevant sources using the reranker.
        This method is more efficient than calling run() multiple times because it
        allows for potential future batch optimization.

        Args:
            sentences: List of Sentence objects with text and character offsets
            source_documents: List of SourceMetadata objects with content
            top_k: Maximum number of sources to return per sentence

        Returns:
            Dict mapping sentence_idx (int) → List[RerankMatch] (top-K filtered matches per sentence)
            - Dictionary keys are indices in the sentences list
            - Dictionary values are sorted lists of top-K matches by relevance score

        Example:
            >>> sentences = [
            ...     Sentence(text="Machine learning needs data", normalized_text="...", start_index=0, end_index=26, length=26),
            ...     Sentence(text="Neural networks are complex", normalized_text="...", start_index=27, end_index=54, length=27),
            ... ]
            >>> sources = [SourceMetadata(chunk_id="s1", content="ML requires datasets", ...), ...]
            >>> result = reranker.run_for_sentences(sentences, sources, top_k=5)
            >>> result[0]  # Top 5 sources for sentence 0
            [RerankMatch(index=0, score=0.92, ...), RerankMatch(index=3, score=0.87, ...), ...]
        """
        if not self.enabled:
            self.logger.debug("Reranker disabled, returning empty results for sentences")
            return {idx: [] for idx in range(len(sentences))}

        if not sentences or not source_documents:
            self.logger.debug("Empty sentences or sources, returning empty results")
            return {idx: [] for idx in range(len(sentences))}

        # Extract source texts once (reused for all sentences)
        source_texts = [doc.content for doc in source_documents]

        # Result dictionary: sentence_idx → list of RerankMatch
        results: Dict[int, List[RerankMatch]] = {}

        start_time = time.time()
        total_matches = 0

        # Process each sentence
        for sentence_idx, sentence in enumerate(sentences):
            try:
                # Skip empty sentences
                if not sentence.normalized_text.strip():
                    self.logger.debug(f"[sentence-{sentence_idx}] Skipping empty sentence")
                    results[sentence_idx] = []
                    continue

                # Rerank sources for this sentence
                matches = self._score_with_adaptive_threshold(query=sentence.normalized_text, candidates=source_texts)

                # Filter to top-K
                filtered_matches = sorted(matches, key=lambda x: x[1], reverse=True)[:top_k]

                # Convert to RerankMatch objects
                match_objects = [
                    RerankMatch(index=idx, score=score, text=source_texts[idx], method=method)
                    for idx, score, method in filtered_matches
                    if score > 0.5  # Still apply base threshold
                ]

                results[sentence_idx] = match_objects
                total_matches += len(match_objects)

                self.logger.debug(
                    f"[sentence-{sentence_idx}] Found {len(match_objects)} top-K matches "
                    f"from {len(matches)} total (text: '{sentence.text[:40]}...')"
                )

            except Exception as e:
                self.logger.warning(
                    f"[sentence-{sentence_idx}] Error reranking for sentence: {e}. "
                    f"Returning empty matches for this sentence."
                )
                results[sentence_idx] = []

        elapsed = time.time() - start_time

        self.logger.debug(
            f"Batch reranking completed for {len(sentences)} sentences: "
            f"{total_matches} total matches in {elapsed:.3f}s "
            f"(avg {elapsed / len(sentences):.3f}s per sentence)"
        )

        return results

    def _score_with_adaptive_threshold(self, query: str, candidates: List[str]) -> List[tuple]:
        """Score candidates with adaptive threshold retry"""

        def score_at_threshold(threshold: float) -> List[tuple]:
            return self._score_candidates(query, candidates, threshold)

        if self.adaptive_enabled:
            matches, final_threshold = adaptive_threshold_search(
                score_func=score_at_threshold,
                base_threshold=self.base_threshold,
                decrement=self.threshold_decrement,
                min_threshold=self.min_threshold,
                max_retries=self.adaptive_max_retries,
            )

            if final_threshold < self.base_threshold:
                self.logger.debug(
                    f"Adaptive retry lowered threshold to {final_threshold:.2f} (found {len(matches)} matches)"
                )
        else:
            matches = score_at_threshold(self.base_threshold)

        # Fuzzy fallback ONLY if reranker is disabled (not if it simply found no matches)
        if not matches and not self.enabled:
            self.logger.warning("Reranker returned no matches but is disabled, using fuzzy fallback")
            fuzzy_matches = fuzzy_match_texts(
                query=query,
                candidates=candidates,
                threshold=self.fuzzy_threshold,
                scorer=self.fuzzy_scorer,
                min_length=self.fuzzy_min_length,
            )

            if fuzzy_matches:
                self.logger.debug(f"Fuzzy fallback found {len(fuzzy_matches)} matches")
                return fuzzy_matches

        return matches

    def _score_candidates(self, query: str, candidates: List[str], threshold: float) -> List[tuple]:
        """Score candidates in batches and filter by threshold"""
        all_matches = []
        batches = batch_items(candidates, self.batch_size)
        batch_times = []

        for batch_idx, batch in enumerate(batches):
            start_time = time.time()
            results = self.api_client.rerank(query=query, texts=batch)
            batch_time = time.time() - start_time
            batch_times.append(batch_time)

            if results is None:
                self.logger.warning(f"Batch {batch_idx + 1}/{len(batches)} failed")
                continue

            # Calculate global indices and filter by threshold
            base_idx = batch_idx * self.batch_size
            for result in results:
                local_idx = result.get("index", 0)
                score = result.get("relevance_score", 0.0)
                global_idx = base_idx + local_idx

                if score >= threshold:
                    all_matches.append((global_idx, score, "reranker"))

        # Log batch statistics
        if batch_times:
            stats = compute_batch_statistics(batch_times)
            self.logger.debug(
                f"Batch processing: {stats['count']} batches, avg {stats['mean']:.3f}s, total {stats['total']:.3f}s"
            )

        # Normalize scores if requested
        if self.normalize and all_matches:
            scores = [score for _, score, _ in all_matches]
            normalized = normalize_scores(scores)
            all_matches = [(idx, norm_score, method) for (idx, _, method), norm_score in zip(all_matches, normalized)]

        return sorted(all_matches, key=lambda x: x[1], reverse=True)