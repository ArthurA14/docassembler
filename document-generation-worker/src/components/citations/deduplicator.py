from enum import Enum
from typing import List, Set, Tuple

from rapidfuzz import fuzz
from document_common.schemas.citations import Citation
from document_common.utils.logger import Logger


class DeduplicationMethod(str, Enum):
    """Deduplication methods for citations"""

    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class CitationDeduplicator:
    """
    Deduplicates citations within a group using various methods.

    Supports exact matching, fuzzy text matching, semantic similarity,
    and hybrid approaches to identify and remove duplicate citations.
    """

    def __init__(
        self,
        method: DeduplicationMethod = DeduplicationMethod.FUZZY,
        fuzzy_threshold: float = 85.0,
        semantic_threshold: float = 0.95,
        fuzzy_scorer: str = "token_sort_ratio",
    ):
        """
        Initialize the citation deduplicator.

        Args:
            method: Deduplication method to use
            fuzzy_threshold: Similarity threshold for fuzzy matching (0-100)
            semantic_threshold: Similarity threshold for semantic matching (0.0-1.0)
            fuzzy_scorer: RapidFuzz scorer to use (token_sort_ratio, ratio, partial_ratio)
        """
        self.logger = Logger().get_logger("citation_deduplicator")
        self.method = method
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.fuzzy_scorer = fuzzy_scorer

        # Map scorer names to functions
        self.scorer_map = {
            "ratio": fuzz.ratio,
            "partial_ratio": fuzz.partial_ratio,
            "token_sort_ratio": fuzz.token_sort_ratio,
            "token_set_ratio": fuzz.token_set_ratio,
        }

        self.total_processed = 0
        self.total_duplicates_removed = 0

        self.logger.debug(
            f"CitationDeduplicator initialized (method: {method.value}, "
            f"fuzzy_threshold: {fuzzy_threshold}, semantic_threshold: {semantic_threshold})"
        )

    def deduplicate(self, citations: List[Citation]) -> List[Citation]:
        """
        Deduplicate a list of citations.

        Args:
            citations: List of citations to deduplicate

        Returns:
            Deduplicated list of citations
        """
        if not citations:
            return []

        # Track metrics before early returns
        self.total_processed += len(citations)

        if len(citations) == 1:
            return citations

        if self.method == DeduplicationMethod.EXACT:
            result = self._deduplicate_exact(citations)
        elif self.method == DeduplicationMethod.FUZZY:
            result = self._deduplicate_fuzzy(citations)
        elif self.method == DeduplicationMethod.SEMANTIC:
            result = self._deduplicate_semantic(citations)
        elif self.method == DeduplicationMethod.HYBRID:
            result = self._deduplicate_hybrid(citations)
        else:
            self.logger.warning(f"Unknown deduplication method: {self.method}, using exact")
            result = self._deduplicate_exact(citations)

        duplicates_removed = len(citations) - len(result)
        self.total_duplicates_removed += duplicates_removed

        if duplicates_removed > 0:
            self.logger.debug(
                f"Removed {duplicates_removed} duplicates from {len(citations)} citations "
                f"using {self.method.value} method"
            )

        return result

    def _deduplicate_exact(self, citations: List[Citation]) -> List[Citation]:
        """
        Deduplicate using exact matching of quote text and source.

        Args:
            citations: List of citations

        Returns:
            Deduplicated list
        """
        seen: Set[Tuple[str, str]] = set()
        unique_citations = []

        for citation in citations:
            # Deduplicate by generated span text and source
            key = (citation.span_text, citation.file_id)

            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)

        return unique_citations

    def _deduplicate_fuzzy(self, citations: List[Citation]) -> List[Citation]:
        """
        Deduplicate using fuzzy text matching.

        Args:
            citations: List of citations

        Returns:
            Deduplicated list
        """
        if len(citations) <= 1:
            return citations

        # Get scorer function
        scorer = self.scorer_map.get(self.fuzzy_scorer, fuzz.token_sort_ratio)

        unique_citations = []
        skip_indices: Set[int] = set()

        for i, citation_a in enumerate(citations):
            if i in skip_indices:
                continue

            # Check against all subsequent citations
            for j in range(i + 1, len(citations)):
                if j in skip_indices:
                    continue

                citation_b = citations[j]

                # If same source, check text similarity
                if citation_a.file_id == citation_b.file_id:
                    similarity = scorer(citation_a.span_text, citation_b.span_text)

                    if similarity >= self.fuzzy_threshold:
                        # Keep the one with higher confidence
                        if citation_b.confidence > citation_a.confidence:
                            skip_indices.add(i)
                            break  # Skip citation_a
                        else:
                            skip_indices.add(j)  # Skip citation_b

            if i not in skip_indices:
                unique_citations.append(citation_a)

        return unique_citations

    def _deduplicate_semantic(self, citations: List[Citation]) -> List[Citation]:
        """
        Deduplicate using semantic similarity scores.

        Args:
            citations: List of citations

        Returns:
            Deduplicated list
        """
        if len(citations) <= 1:
            return citations

        unique_citations = []
        skip_indices: Set[int] = set()

        for i, citation_a in enumerate(citations):
            if i in skip_indices:
                continue

            for j in range(i + 1, len(citations)):
                if j in skip_indices:
                    continue

                citation_b = citations[j]

                # If same source and high similarity, consider duplicate
                if citation_a.file_id == citation_b.file_id:
                    # Use similarity_score as proxy for semantic similarity
                    # Assuming both have similar similarity to the generated text
                    sim_diff = abs(citation_a.similarity_score - citation_b.similarity_score)

                    # If similarity scores are very close, likely duplicates
                    if sim_diff < (1.0 - self.semantic_threshold):
                        # Also check offset overlap
                        overlap = self._calculate_offset_overlap(citation_a, citation_b)

                        if overlap > 0.8:  # 80% overlap
                            # Keep the one with higher confidence
                            if citation_b.confidence > citation_a.confidence:
                                skip_indices.add(i)
                                break
                            else:
                                skip_indices.add(j)

            if i not in skip_indices:
                unique_citations.append(citation_a)

        return unique_citations

    def _deduplicate_hybrid(self, citations: List[Citation]) -> List[Citation]:
        """
        Deduplicate using a hybrid approach combining multiple methods.

        Args:
            citations: List of citations

        Returns:
            Deduplicated list
        """
        # First pass: exact deduplication
        citations = self._deduplicate_exact(citations)

        if len(citations) <= 1:
            return citations

        # Second pass: fuzzy deduplication with higher threshold
        original_threshold = self.fuzzy_threshold
        self.fuzzy_threshold = min(90.0, self.fuzzy_threshold + 5.0)
        citations = self._deduplicate_fuzzy(citations)
        self.fuzzy_threshold = original_threshold

        if len(citations) <= 1:
            return citations

        # Third pass: semantic deduplication
        citations = self._deduplicate_semantic(citations)

        return citations

    def _calculate_offset_overlap(self, citation_a: Citation, citation_b: Citation) -> float:
        """
        Calculate offset overlap between two citations.

        Args:
            citation_a: First citation
            citation_b: Second citation

        Returns:
            Overlap ratio (0.0-1.0)
        """
        # Calculate overlap in generated text span
        overlap_start = max(citation_a.span_start, citation_b.span_start)
        overlap_end = min(citation_a.span_end, citation_b.span_end)

        if overlap_end <= overlap_start:
            return 0.0

        overlap_length = overlap_end - overlap_start

        # Calculate as ratio of smaller citation
        len_a = citation_a.span_end - citation_a.span_start
        len_b = citation_b.span_end - citation_b.span_start
        min_length = min(len_a, len_b)

        if min_length == 0:
            return 0.0

        return overlap_length / min_length

    def get_metrics(self) -> dict:
        """
        Get deduplication metrics.

        Returns:
            Dict with deduplication statistics
        """
        dedup_rate = self.total_duplicates_removed / self.total_processed if self.total_processed > 0 else 0

        return {
            "total_processed": self.total_processed,
            "total_duplicates_removed": self.total_duplicates_removed,
            "deduplication_rate": dedup_rate,
            "method": self.method.value,
        }

    def reset_metrics(self) -> None:
        """Reset deduplication metrics."""
        self.total_processed = 0
        self.total_duplicates_removed = 0
        self.logger.debug("Deduplication metrics reset")
