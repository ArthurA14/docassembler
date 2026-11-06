"""
Reranker filtering stage for the citation extraction pipeline.

Filters sources per sentence using semantic similarity scoring.
This reduces the number of sources passed to the classifier by keeping
only the top-K most relevant sources for each sentence.
"""

from typing import Any, Dict, List, Optional

from document_common.components.base.batched_operations import BatchedOperations
from document_common.components.base.reranker import RerankMatch
from document_common.schemas.citations import SourceMetadata
from document_common.utils.logger import Logger

from src.components.base.text_reranker import TextPairReranker
from src.pipeline.citation.stages.base import PipelineStageInput
from src.utils.citations.execution import ComponentExecutor


class SourceSentenceReranker:
    """Filters sources per sentence using semantic reranker scoring."""

    def __init__(
        self,
        reranker: TextPairReranker,
        enabled: bool = True,
        min_reranker_score: float = 0.60,
        top_k_sources: int = 5,
        fallback_to_all_sources: bool = True,
        batch_size: int = 128,
    ):
        """
        Initialize the reranker filtering stage.

        Args:
            reranker: TextPairReranker component for scoring
            enabled: If False, returns all sources (passthrough)
            min_reranker_score: Minimum score to keep a source
            top_k_sources: Maximum sources to return per sentence
            fallback_to_all_sources: If reranker fails, use all sources instead of none
        """
        self.reranker = reranker
        self.enabled = enabled
        self.min_reranker_score = min_reranker_score
        self.top_k_sources = top_k_sources
        self.fallback_to_all_sources = fallback_to_all_sources

        # NEW: Initialize BatchedOperations for internal batch processing
        self._batcher = BatchedOperations(batch_size=batch_size, show_progress=True)

        self.logger = Logger().get_logger("source_filtering_stage")

    def execute(self, inputs: PipelineStageInput, sentences: List[Any]) -> Optional[Dict[int, List[SourceMetadata]]]:
        """
        Execute reranker filtering.

        Args:
            inputs: Pipeline input data
            sentences: Preprocessed sentences

        Returns:
            Dict mapping sentence_idx → filtered List[SourceMetadata]
            or None on error
        """
        executor = ComponentExecutor("RerankerFiltering", inputs.request_id)

        try:
            # Passthrough if disabled
            if not self.enabled:
                executor.log_info("Reranker filtering disabled, using all sources")
                filtered_sources = {idx: inputs.source_documents for idx in range(len(sentences))}
                inputs.execution_metrics.record_execution(executor)
                return filtered_sources

            # Validate inputs
            if not sentences or not inputs.source_documents:
                executor.log_warning("Empty sentences or sources, skipping filtering")
                filtered_sources = {idx: inputs.source_documents for idx in range(len(sentences))}
                inputs.execution_metrics.record_execution(executor)
                return filtered_sources

            executor.log_info(
                f"Filtering {len(inputs.source_documents)} sources for "
                f"{len(sentences)} sentences (top_k={self.top_k_sources}, "
                f"min_score={self.min_reranker_score})"
            )

            # Call reranker to score all source-sentence pairs
            rerank_results = self.reranker.run_for_sentences(
                sentences=sentences,
                source_documents=inputs.source_documents,
                top_k=self.top_k_sources,
            )

            # Convert RerankMatch results to SourceMetadata filtered mapping
            filtered_sources = self._build_filtered_sources_mapping(
                rerank_results, inputs.source_documents, len(sentences)
            )

            # Track filtering statistics
            stats = self._calculate_filtering_stats(filtered_sources, len(inputs.source_documents), len(sentences))

            executor.log_info(
                f"Filtered sources: avg {stats['avg_sources_per_sentence']:.1f} "
                f"per sentence (min={stats['min_sources']}, max={stats['max_sources']}, "
                f"sentences_with_sources={stats['sentences_with_sources']}/{len(sentences)})"
            )

            inputs.execution_metrics.record_execution(executor)
            return filtered_sources

        except Exception as e:
            executor.handle_error(e, "Reranker filtering failed")
            inputs.errors.extend(executor.errors)
            inputs.execution_metrics.record_execution(executor)

            # Fallback: return all sources for all sentences
            if self.fallback_to_all_sources:
                executor.log_warning("Falling back to all sources")
                return {idx: inputs.source_documents for idx in range(len(sentences))}
            else:
                executor.log_warning("Returning empty filtered sources")
                return {idx: [] for idx in range(len(sentences))}

    def _build_filtered_sources_mapping(
        self,
        rerank_results: Dict[int, List[RerankMatch]],
        all_sources: List[SourceMetadata],
        num_sentences: int,
    ) -> Dict[int, List[SourceMetadata]]:
        """
        Convert RerankMatch results to SourceMetadata mapping.

        Args:
            rerank_results: Dict[sentence_idx → List[RerankMatch]]
            all_sources: Original source documents list
            num_sentences: Total number of sentences

        Returns:
            Dict[sentence_idx → List[SourceMetadata]] filtered by score and top_k
        """
        filtered_sources: Dict[int, List[SourceMetadata]] = {}

        for sentence_idx in range(num_sentences):
            matches = rerank_results.get(sentence_idx, [])

            if not matches:
                # No matches for this sentence
                if self.fallback_to_all_sources:
                    filtered_sources[sentence_idx] = all_sources
                else:
                    filtered_sources[sentence_idx] = []
                continue

            # Filter matches by minimum score and top_k limit
            filtered_matches = [m for m in matches if m.score >= self.min_reranker_score][: self.top_k_sources]

            if not filtered_matches:
                # Scores below threshold
                if self.fallback_to_all_sources:
                    filtered_sources[sentence_idx] = all_sources
                else:
                    filtered_sources[sentence_idx] = []
                continue

            # Map RerankMatch indices back to SourceMetadata objects
            sentence_sources = []
            for match in filtered_matches:
                if 0 <= match.index < len(all_sources):
                    sentence_sources.append(all_sources[match.index])

            if not sentence_sources and self.fallback_to_all_sources:
                filtered_sources[sentence_idx] = all_sources
            else:
                filtered_sources[sentence_idx] = sentence_sources

        return filtered_sources

    def _calculate_filtering_stats(
        self,
        filtered_sources: Dict[int, List[SourceMetadata]],
        total_sources: int,
        total_sentences: int,
    ) -> Dict[str, Any]:
        """
        Calculate filtering statistics for logging.

        Args:
            filtered_sources: Filtered mapping
            total_sources: Original source count
            total_sentences: Total sentence count

        Returns:
            Dictionary with filtering statistics
        """
        sources_per_sentence = [len(filtered_sources.get(idx, [])) for idx in range(total_sentences)]

        sentences_with_sources = sum(1 for count in sources_per_sentence if count > 0)

        return {
            "avg_sources_per_sentence": (sum(sources_per_sentence) / total_sentences if total_sentences > 0 else 0),
            "min_sources": min(sources_per_sentence) if sources_per_sentence else 0,
            "max_sources": max(sources_per_sentence) if sources_per_sentence else 0,
            "sentences_with_sources": sentences_with_sources,
            "total_original_sources": total_sources,
            "total_after_filtering": sum(sources_per_sentence),
            "reduction_percentage": (
                100 * (1 - sum(sources_per_sentence) / (total_sources * total_sentences))
                if (total_sources * total_sentences) > 0
                else 0
            ),
        }
