from enum import Enum
from typing import Dict, List, Optional

from document_common.schemas.citations import Citation
from document_common.utils.logger import Logger


class RankingStrategy(str, Enum):
    """Ranking strategies for citations"""

    CONFIDENCE = "confidence"
    SOURCE_PRIORITY = "source_priority"
    TEMPORAL = "temporal"
    COMBINED = "combined"


class CitationRanker:
    """
    Ranks citations using configurable strategies.

    Supports confidence-based, source priority, temporal, and combined
    ranking methods to determine primary and supporting citations.
    """

    def __init__(
        self,
        strategy: RankingStrategy = RankingStrategy.COMBINED,
        source_priorities: Optional[Dict[str, float]] = None,
        combined_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the citation ranker.

        Args:
            strategy: Ranking strategy to use
            source_priorities: Dict mapping source_id to priority score (0.0-1.0)
            combined_weights: Dict with weights for combined strategy
                             (confidence, similarity, source_priority, recency)
        """
        self.logger = Logger().get_logger("citation_ranker")
        self.strategy = strategy
        self.source_priorities = source_priorities or {}

        # Default weights for combined strategy
        self.combined_weights = combined_weights or {
            "confidence": 0.50,
            "similarity": 0.30,
            "source_priority": 0.15,
            "recency": 0.05,
        }

        self.total_ranked = 0

        self.logger.info(
            f"CitationRanker initialized (strategy: {strategy.value}, "
            f"source_priorities: {len(self.source_priorities)} sources)"
        )

    def rank(self, citations: List[Citation], preserve_order: bool = False) -> List[Citation]:
        """
        Rank citations according to the configured strategy.

        Args:
            citations: List of citations to rank
            preserve_order: If True, use original order for temporal ranking

        Returns:
            Sorted list of citations (highest rank first)
        """
        if not citations:
            return []

        if len(citations) == 1:
            return citations

        self.total_ranked += len(citations)

        if self.strategy == RankingStrategy.CONFIDENCE:
            result = self._rank_by_confidence(citations)
        elif self.strategy == RankingStrategy.SOURCE_PRIORITY:
            result = self._rank_by_source_priority(citations)
        elif self.strategy == RankingStrategy.TEMPORAL:
            result = self._rank_by_temporal(citations, preserve_order)
        elif self.strategy == RankingStrategy.COMBINED:
            result = self._rank_by_combined(citations, preserve_order)
        else:
            self.logger.warning(f"Unknown ranking strategy: {self.strategy}, using confidence")
            result = self._rank_by_confidence(citations)

        self.logger.debug(f"Ranked {len(citations)} citations using {self.strategy.value} strategy")

        return result

    def _rank_by_confidence(self, citations: List[Citation]) -> List[Citation]:
        """
        Rank citations by confidence score (descending).

        Args:
            citations: List of citations

        Returns:
            Sorted list by confidence
        """
        return sorted(citations, key=lambda c: c.confidence, reverse=True)

    def _rank_by_source_priority(self, citations: List[Citation]) -> List[Citation]:
        """
        Rank citations by source priority, then confidence.

        Args:
            citations: List of citations

        Returns:
            Sorted list by source priority
        """

        def priority_score(citation: Citation) -> tuple:
            # Get priority from mapping, default to 0.5 (medium priority)
            priority = self.source_priorities.get(citation.file_id, 0.5)
            # Return tuple for multi-key sort: (priority desc, confidence desc)
            return (-priority, -citation.confidence)

        return sorted(citations, key=priority_score)

    def _rank_by_temporal(self, citations: List[Citation], preserve_order: bool = False) -> List[Citation]:
        """
        Rank citations by temporal ordering (most recent first).

        If preserve_order is True, assumes input list is already temporally ordered.
        Otherwise, uses confidence as a tiebreaker.

        Args:
            citations: List of citations
            preserve_order: Use original list order

        Returns:
            Sorted list by temporal order
        """
        if preserve_order:
            # Reverse to get most recent first
            return list(reversed(citations))

        # If no explicit temporal info, use confidence as proxy
        # (assuming higher confidence correlates with more recent/reliable sources)
        return self._rank_by_confidence(citations)

    def _rank_by_combined(self, citations: List[Citation], preserve_order: bool = False) -> List[Citation]:
        """
        Rank citations using weighted combination of multiple factors.

        Combines confidence, similarity score, source priority, and recency
        according to configured weights.

        Args:
            citations: List of citations
            preserve_order: Use original order for recency calculation

        Returns:
            Sorted list by combined score
        """

        def combined_score(citation_with_index: tuple) -> float:
            index, citation = citation_with_index

            # Confidence component (0.0-1.0)
            confidence_component = citation.confidence * self.combined_weights["confidence"]

            # Similarity component (0.0-1.0)
            similarity_component = citation.similarity_score * self.combined_weights["similarity"]

            # Source priority component (0.0-1.0)
            source_priority = self.source_priorities.get(citation.chunk_id, 0.5)
            source_component = source_priority * self.combined_weights["source_priority"]

            # Recency component (0.0-1.0)
            # More recent = higher score (later index = higher recency)
            if preserve_order and len(citations) > 1:
                recency_score = index / (len(citations) - 1)
            else:
                recency_score = 0.5  # Default to middle value

            recency_component = recency_score * self.combined_weights["recency"]

            # Total weighted score
            total = confidence_component + similarity_component + source_component + recency_component

            return total

        # Enumerate to track original order for recency
        citations_with_index = list(enumerate(citations))

        # Sort by combined score (descending)
        sorted_with_index = sorted(citations_with_index, key=combined_score, reverse=True)

        # Extract citations
        return [citation for _, citation in sorted_with_index]

    def set_source_priority(self, source_id: str, priority: float) -> None:
        """
        Set priority for a specific source.

        Args:
            source_id: Source identifier
            priority: Priority score (0.0-1.0, higher is better)
        """
        if not 0.0 <= priority <= 1.0:
            self.logger.warning(f"Priority {priority} out of range [0.0, 1.0], clamping")
            priority = max(0.0, min(1.0, priority))

        self.source_priorities[source_id] = priority
        self.logger.debug(f"Set priority for source {source_id}: {priority}")

    def set_source_priorities(self, priorities: Dict[str, float]) -> None:
        """
        Set priorities for multiple sources.

        Args:
            priorities: Dict mapping source_id to priority score
        """
        for source_id, priority in priorities.items():
            self.set_source_priority(source_id, priority)

    def get_source_priority(self, source_id: str) -> float:
        """
        Get priority for a source.

        Args:
            source_id: Source identifier

        Returns:
            Priority score (0.0-1.0), defaults to 0.5 if not set
        """
        return self.source_priorities.get(source_id, 0.5)

    def update_combined_weights(self, weights: Dict[str, float]) -> None:
        """
        Update weights for combined ranking strategy.

        Args:
            weights: Dict with keys: confidence, similarity, source_priority, recency
        """
        # Validate weights sum to approximately 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"Combined weights sum to {total_weight}, not 1.0. Weights will be normalized.")
            # Normalize weights
            weights = {k: v / total_weight for k, v in weights.items()}

        self.combined_weights.update(weights)
        self.logger.info(f"Updated combined weights: {self.combined_weights}")

    def get_metrics(self) -> dict:
        """
        Get ranking metrics.

        Returns:
            Dict with ranking statistics
        """
        return {
            "total_ranked": self.total_ranked,
            "strategy": self.strategy.value,
            "num_source_priorities": len(self.source_priorities),
        }

    def reset_metrics(self) -> None:
        """Reset ranking metrics."""
        self.total_ranked = 0
        self.logger.info("Ranking metrics reset")
