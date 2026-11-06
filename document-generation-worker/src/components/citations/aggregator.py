from collections import defaultdict
from typing import Dict, List, Optional

from haystack import component
from document_common.schemas.citations import Citation, CitationStack, SupportingCitation
from document_common.utils.logger import Logger

from config import config
from src.components.citations.deduplicator import (
    CitationDeduplicator,
    DeduplicationMethod,
)
from src.components.citations.ranker import CitationRanker, RankingStrategy
from src.utils.confidence_boosting import (
    assess_stack_quality,
    calculate_confidence_with_method,
)


@component
class CitationAggregator:
    """
    Haystack component for aggregating citations into citation stacks.

    Groups citations by generated text span, ranks them by confidence/priority,
    creates stacks with primary + supporting structure, calculates boosted
    stack confidence, and enforces size limits.
    """

    def __init__(
        self,
        ranker: Optional[CitationRanker] = None,
        deduplicator: Optional[CitationDeduplicator] = None,
        max_stack_size: Optional[int] = None,
        min_stack_confidence: Optional[float] = None,
        boost_factor: Optional[float] = None,
        boost_method: str = "standard",
        ranking_strategy: Optional[str] = None,
        deduplication_method: Optional[str] = None,
        fuzzy_threshold: float = 85.0,
        source_priorities: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the citation aggregator.

        Args:
            ranker: Optional CitationRanker dependency (if None, will be created)
            deduplicator: Optional CitationDeduplicator dependency (if None, will be created)
            max_stack_size: Maximum supporting citations per stack (default from config)
            min_stack_confidence: Minimum confidence for stacks (default from config)
            boost_factor: Confidence boost per supporting citation (default from config)
            boost_method: Method for confidence boosting (standard, progressive, diversity, quality)
            ranking_strategy: Strategy for ranking citations (confidence, source_priority, temporal, combined)
            deduplication_method: Method for deduplication (exact, fuzzy, semantic, hybrid)
            fuzzy_threshold: Threshold for fuzzy deduplication (0-100)
            source_priorities: Dict mapping source_id to priority score (0.0-1.0)
        """
        self.logger = Logger().get_logger("citation_aggregator")

        # Load configuration
        citation_config = config.get_current_config().citations
        stacking_config = citation_config.stacking

        self.max_stack_size = max_stack_size or stacking_config.get("max_stack_size", 5)
        self.min_stack_confidence = min_stack_confidence or stacking_config.get("min_stack_confidence", 0.80)
        self.boost_factor = boost_factor or stacking_config.get("boost_factor", 0.05)
        self.boost_method = boost_method
        self.max_boost = 0.25  # Maximum total boost

        # Use provided dependencies or create new ones
        if ranker is not None:
            self.ranker = ranker
        else:
            ranking_strategy_enum = self._parse_ranking_strategy(
                ranking_strategy or stacking_config.get("ranking_strategy", "combined")
            )
            self.ranker = CitationRanker(strategy=ranking_strategy_enum, source_priorities=source_priorities or {})

        if deduplicator is not None:
            self.deduplicator = deduplicator
        else:
            dedup_method_enum = self._parse_deduplication_method(
                deduplication_method or stacking_config.get("deduplication_method", "fuzzy")
            )
            self.deduplicator = CitationDeduplicator(
                method=dedup_method_enum,
                fuzzy_threshold=fuzzy_threshold,
                semantic_threshold=0.95,
            )

        # Metrics
        self.total_citations_processed = 0
        self.total_stacks_created = 0
        self.total_single_citation_stacks = 0
        self.total_low_confidence_stacks = 0

        ranker_info = type(self.ranker).__name__
        dedup_info = type(self.deduplicator).__name__
        self.logger.debug(
            f"CitationAggregator initialized (max_stack_size: {self.max_stack_size}, "
            f"ranker: {ranker_info}, deduplicator: {dedup_info})"
        )

    @component.output_types(citation_stacks=List[CitationStack])
    def run(
        self, citations: List[Citation], generated_text: Optional[str] = None, preserve_order: bool = False
    ) -> Dict[str, List[CitationStack]]:
        """
        Aggregate citations into citation stacks.

        Args:
            citations: List of citations to aggregate
            generated_text: Full generated text for span extraction (optional)
            preserve_order: Preserve original order for temporal ranking

        Returns:
            Dict with 'citation_stacks' key containing list of CitationStack objects
        """
        if not citations:
            self.logger.debug("No citations to aggregate")
            return {"citation_stacks": []}

        self.total_citations_processed += len(citations)

        # Step 1: Group citations by generated text span
        grouped_citations = self._group_by_span(citations)

        self.logger.debug(
            f"Grouped {len(citations)} citations into {len(grouped_citations)} spans "
            f"(avg {len(citations) / len(grouped_citations):.1f} citations per span)"
        )

        # Step 2: Process each group into a citation stack
        stacks = []
        for span_key, citation_group in grouped_citations.items():
            stack = self._create_citation_stack(citation_group, generated_text, preserve_order)
            if stack:
                stacks.append(stack)

        self.total_stacks_created += len(stacks)

        self.logger.debug(
            f"Created {len(stacks)} citation stacks from {len(citations)} citations "
            f"(avg {len(citations) / len(stacks) if stacks else 0:.1f} citations per stack)"
        )

        return {"citation_stacks": stacks}

    def _group_by_span(self, citations: List[Citation]) -> Dict[tuple, List[Citation]]:
        """
        Group citations by generated text span.

        Citations are grouped if they reference the same generated text location.
        This ensures that all citations supporting the same generated text span
        are aggregated together into a single CitationStack.

        Args:
            citations: List of citations to group

        Returns:
            Dict mapping span_key (start, end) to list of citations for that span
        """
        # Group citations by generated text span
        # All citations with the same generated span location are grouped together
        grouped: Dict[tuple, List[Citation]] = defaultdict(list)

        for citation in citations:
            # Group by generated text span location
            # All citations with the same span are supporting citations for that text
            span_key = (citation.span_start, citation.span_end)
            grouped[span_key].append(citation)

        return grouped

    def _create_citation_stack(
        self, citations: List[Citation], generated_text: Optional[str], preserve_order: bool
    ) -> Optional[CitationStack]:
        """
        Create a citation stack from a group of citations.

        Args:
            citations: List of citations for the same span
            generated_text: Full generated text
            preserve_order: Preserve citation order for temporal ranking

        Returns:
            CitationStack or None if creation fails
        """
        if not citations:
            return None

        # Step 1: Deduplicate citations
        # Apply deduplication based on configured method.
        # Note: When deduplication is enabled, truly duplicate citations are removed.
        # This is correct behavior - duplicates should not count as supporting citations.
        # If you want to keep similar citations as supporting evidence, configure
        # deduplication to use a stricter method (e.g., "exact" only) or disable it.
        unique_citations = self.deduplicator.deduplicate(citations)

        if not unique_citations:
            self.logger.warning("No citations remained after deduplication")
            return None

        # Step 2: Rank citations
        ranked_citations = self.ranker.rank(unique_citations, preserve_order=preserve_order)

        # Step 3: Select primary and supporting citations
        primary_citation = ranked_citations[0]
        supporting_citations = [
            SupportingCitation(
                chunk_id=sup.chunk_id,
                file_id=sup.file_id,
                confidence=sup.confidence,
                similarity_score=sup.similarity_score,
                citation_label=sup.citation_label,
            )
            for sup in ranked_citations[1 : self.max_stack_size + 1]
        ]

        # Step 4: Calculate stack confidence with boosting
        stack_confidence = calculate_confidence_with_method(
            primary_citation=primary_citation,
            supporting_citations=supporting_citations,
            method=self.boost_method,
            boost_factor=self.boost_factor,
            max_boost=self.max_boost,
        )

        # Step 5: Assess stack quality
        quality_assessment = assess_stack_quality(primary_citation, supporting_citations)

        # Log quality issues
        if quality_assessment["has_low_confidence"]:
            self.logger.warning(f"Stack has low confidence primary citation: {primary_citation.confidence:.2f}")
            self.total_low_confidence_stacks += 1

        if quality_assessment["weak_support"]:
            self.total_single_citation_stacks += 1

        # Step 6: Extract generated text span information from Citation
        # The Citation now contains generated_span_* fields populated by the pipeline
        # with actual generated text span information from the Sentence object.
        # This eliminates the need for unreliable string searching and supports
        # all citation types including paraphrases.

        span_text = primary_citation.span_text
        span_start_index = primary_citation.span_start
        span_end_index = primary_citation.span_end

        self.logger.debug(
            f"Creating citation stack for span [{span_start_index}:{span_end_index}]: "
            f"'{span_text[:50]}...' ({primary_citation.citation_label.value})"
        )

        # Step 7: Create CitationStack
        try:
            stack = CitationStack(
                span_text=span_text,
                span_start=span_start_index,
                span_end=span_end_index,
                primary_citation=primary_citation,
                supporting_citations=supporting_citations,
                stack_confidence=stack_confidence,
            )

            # Check minimum confidence threshold
            if stack.stack_confidence < self.min_stack_confidence:
                self.logger.debug(
                    f"Stack confidence {stack.stack_confidence:.2f} below minimum "
                    f"{self.min_stack_confidence:.2f}, but including anyway"
                )

            return stack

        except Exception as e:
            self.logger.error(f"Failed to create citation stack: {e}")
            return None

    def _parse_ranking_strategy(self, strategy: str) -> RankingStrategy:
        """Parse ranking strategy string to enum."""
        try:
            return RankingStrategy(strategy.lower())
        except ValueError:
            self.logger.warning(f"Unknown ranking strategy '{strategy}', using 'combined'")
            return RankingStrategy.COMBINED

    def _parse_deduplication_method(self, method: str) -> DeduplicationMethod:
        """Parse deduplication method string to enum."""
        try:
            return DeduplicationMethod(method.lower())
        except ValueError:
            self.logger.warning(f"Unknown deduplication method '{method}', using 'fuzzy'")
            return DeduplicationMethod.FUZZY

    def set_source_priority(self, source_id: str, priority: float) -> None:
        """
        Set priority for a specific source.

        Args:
            source_id: Source identifier
            priority: Priority score (0.0-1.0)
        """
        self.ranker.set_source_priority(source_id, priority)

    def set_source_priorities(self, priorities: Dict[str, float]) -> None:
        """
        Set priorities for multiple sources.

        Args:
            priorities: Dict mapping source_id to priority
        """
        self.ranker.set_source_priorities(priorities)

    def get_metrics(self) -> dict:
        """
        Get aggregation metrics.

        Returns:
            Dict with aggregation statistics
        """
        ranker_metrics = self.ranker.get_metrics()
        dedup_metrics = self.deduplicator.get_metrics()

        avg_citations_per_stack = (
            self.total_citations_processed / self.total_stacks_created if self.total_stacks_created > 0 else 0
        )

        return {
            "total_citations_processed": self.total_citations_processed,
            "total_stacks_created": self.total_stacks_created,
            "total_single_citation_stacks": self.total_single_citation_stacks,
            "total_low_confidence_stacks": self.total_low_confidence_stacks,
            "avg_citations_per_stack": avg_citations_per_stack,
            "ranker_metrics": ranker_metrics,
            "deduplicator_metrics": dedup_metrics,
        }

    def reset_metrics(self) -> None:
        """Reset all aggregation metrics."""
        self.total_citations_processed = 0
        self.total_stacks_created = 0
        self.total_single_citation_stacks = 0
        self.total_low_confidence_stacks = 0
        self.ranker.reset_metrics()
        self.deduplicator.reset_metrics()
        self.logger.debug("Aggregation metrics reset")
