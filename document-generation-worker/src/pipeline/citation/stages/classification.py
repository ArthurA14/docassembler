"""
Classification stage for the citation extraction pipeline.

Classifies sentence-source pairs using batch processing with LLM and fallback.
"""

import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

from document_common.components.base.embedder import TextEmbedder
from document_common.schemas.citations import Citation
from document_common.utils.logger import Logger

from src.components.base.text_reranker import TextPairReranker
from src.components.citations.classifier.classification import CitationClassifier
from src.pipeline.citation.stages.base import PipelineStageInput
from src.utils.citations.execution import ComponentExecutor


class ClassificationStage:
    """Classifies citations through batch reranking and LLM classification."""

    # Batch size for concurrent similarity score computation
    SIMILARITY_BATCH_SIZE = 64

    def __init__(
        self,
        classifier: CitationClassifier,
        reranker: Optional[TextPairReranker] = None,
        mvp_mode: bool = True,
    ):
        """
        Initialize the classification stage.

        Args:
            classifier: CitationClassifier component
            reranker: Optional TextPairReranker for top-K filtering
            mvp_mode: If True, bypasses reranker (MVP mode)
        """
        self.classifier = classifier
        self.reranker = reranker
        self.logger = Logger().get_logger("classification_stage")

        # Initialize text embedder for similarity scoring
        self.text_embedder = TextEmbedder()

        # Initialize event loop for async operations
        self._loop = None

    def execute(
        self,
        inputs: PipelineStageInput,
        sentences: List[Any],
        state: Optional[dict] = None,
    ) -> Optional[List[Citation]]:
        """
        Execute classification.

        Args:
            inputs: Pipeline input data
            sentences: Preprocessed sentences
            state: Optional orchestrator state dict containing filtered_sources from SourceSentenceReranker

        Returns:
            List of Citation objects or None on error
        """
        executor = ComponentExecutor("Classification", inputs.request_id)

        try:
            # Calculate total pairs for logging
            total_pairs_before_filtering = len(sentences) * len(inputs.source_documents)

            executor.log_info(
                f"Starting batch classification ({len(sentences)} sentences, {len(inputs.source_documents)} sources)"
            )

            # Get filtered sources from orchestrator state if available
            filtered_sources_map = state.get("filtered_sources") if state else None

            # Phase 2: Build classification pairs for batch processing
            classification_pairs = []
            pair_metadata = []
            batch_pairs = []  # Initialize for batch similarity computation

            for sentence_idx, sentence in enumerate(sentences):
                if not getattr(sentence, "normalized_text", "").strip():
                    continue

                # Determine sources for this sentence
                if filtered_sources_map is not None and sentence_idx in filtered_sources_map:
                    # Use per-sentence filtered sources from SourceSentenceReranker
                    sources_for_sentence = filtered_sources_map[sentence_idx]
                else:
                    # Backward compatible: use all sources
                    sources_for_sentence = inputs.source_documents

                # Build pairs for this sentence against its sources
                for source_idx, source_doc in enumerate(sources_for_sentence):
                    text = getattr(sentence, "normalized_text", "")
                    source_content = source_doc.content

                    if not text.strip() or not source_content.strip():
                        # Skip empty pairs with zero similarity
                        classification_pairs.append({
                            "text": text,
                            "source": source_content,
                            "score": 0.0,
                        })
                        pair_metadata.append({
                            "sentence_idx": sentence_idx,
                            "sentence": sentence,
                            "source_doc": source_doc,
                            "similarity_score": 0.0,
                        })
                    else:
                        # Collect pairs for batch similarity computation
                        batch_pairs.append({
                            "text": text,
                            "source": source_content,
                            "metadata": {
                                "sentence_idx": sentence_idx,
                                "sentence": sentence,
                                "source_doc": source_doc,
                            },
                        })

            # Phase 2.5: Compute similarity scores in batches asynchronously
            if batch_pairs:
                executor.log_info(f"Computing similarity scores for {len(batch_pairs)} pairs using async batching")

                # Run async similarity score computation
                similarity_results = self._compute_similarity_scores_async(batch_pairs)

                # Populate classification_pairs and pair_metadata with results
                for pair, (similarity_score, original_metadata) in zip(batch_pairs, similarity_results):
                    classification_pairs.append({
                        "text": pair["text"],
                        "source": pair["source"],
                        "score": similarity_score,
                    })

                    pair_metadata.append({
                        "sentence_idx": original_metadata["sentence_idx"],
                        "sentence": original_metadata["sentence"],
                        "source_doc": original_metadata["source_doc"],
                        "similarity_score": similarity_score,
                    })

            if not classification_pairs:
                executor.log_warning("No classification pairs created")
                inputs.warnings.append("No valid sentence-source pairs for classification")
                inputs.execution_metrics.record_execution(executor)
                return []

            # Phase 3: Run batch classification
            executor.log_info(f"Processing {len(classification_pairs)} classification pairs")

            batch_results = self.classifier.batch_classify(classification_pairs)
            citations: List[Citation] = []

            for result, metadata in zip(batch_results, pair_metadata):
                try:
                    if result.get("error"):
                        inputs.warnings.append(
                            f"Classification failed for sentence {metadata['sentence_idx']}: {result['error']}"
                        )
                        continue

                    classification = result.get("classification")
                    if not classification:
                        continue

                    # Skip independent classifications
                    if classification.classification.value == "independent":
                        continue

                    # Create Citation object
                    citation = Citation(
                        span_text=getattr(metadata["sentence"], "text", ""),
                        span_start=getattr(metadata["sentence"], "start_index", 0),
                        span_end=getattr(metadata["sentence"], "end_index", 0),
                        file_id=metadata["source_doc"].file_id,
                        chunk_id=metadata["source_doc"].chunk_id,
                        citation_label=classification.classification,
                        confidence=classification.confidence,
                        similarity_score=metadata["similarity_score"],
                    )

                    citations.append(citation)

                except Exception as e:
                    executor.log_warning(f"Failed to create citation from result: {str(e)}")
                    inputs.warnings.append(f"Citation creation failed: {str(e)}")

            # Log filtering reduction statistics if reranker was used
            if filtered_sources_map is not None:
                actual_pairs = len(classification_pairs)
                reduction_percentage = (
                    100 * (1 - actual_pairs / total_pairs_before_filtering) if total_pairs_before_filtering > 0 else 0
                )
                executor.log_info(
                    f"Classification complete: {len(citations)} citations created | "
                    f"Filtering reduction: {total_pairs_before_filtering} → {actual_pairs} pairs "
                    f"({reduction_percentage:.1f}% reduction)"
                )
            else:
                executor.log_info(f"Classification complete: {len(citations)} citations created")

            inputs.execution_metrics.record_execution(executor)
            return citations

        except Exception as e:
            executor.handle_error(e, "Classification stage failed")
            inputs.errors.extend(executor.errors)
            inputs.execution_metrics.record_execution(executor)
            return None

    def similarity_score(self, sentence: str, source: str) -> float:
        """
        Calculate similarity score between sentence and source using embedding API.

        Uses cosine similarity between embeddings of sentence and source text.
        The text embedder is configured to use the same model and dimension as the
        system-wide embedding API (Qwen/Qwen3-Embedding-4B with 768 dimensions).

        Args:
            sentence: Generated text to compare
            source: Source text to compare against

        Returns:
            float: Cosine similarity score between 0 and 1
        """
        try:
            # Get embeddings for both texts using the same mode for consistency
            sentence_embedding_result = self.text_embedder.run(sentence, mode="document")
            source_embedding_result = self.text_embedder.run(source, mode="document")

            sentence_embedding = sentence_embedding_result["embedding"]
            source_embedding = source_embedding_result["embedding"]

            # Calculate cosine similarity
            # Dot product of normalized vectors equals cosine similarity
            dot_product = sum(a * b for a, b in zip(sentence_embedding, source_embedding))
            sentence_norm = sum(a * a for a in sentence_embedding) ** 0.5
            source_norm = sum(b * b for b in source_embedding) ** 0.5

            if sentence_norm == 0 or source_norm == 0:
                return 0.0

            cosine_similarity = dot_product / (sentence_norm * source_norm)

            # Clamp to [0, 1] range
            return max(0.0, min(1.0, cosine_similarity))

        except Exception as e:
            self.logger.warning(f"Failed to calculate similarity score: {e}")
            return 0.5  # Fallback value

    def _compute_similarity_scores_async(self, batch_pairs: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Compute similarity scores for multiple pairs asynchronously using batching.

        This method batches API calls to the embedding service for improved performance.

        Args:
            batch_pairs: List of pairs containing text, source, and metadata

        Returns:
            List of tuples (similarity_score, metadata) for each pair
        """
        try:
            # Check if we're in a running event loop
            try:
                _ = asyncio.get_running_loop()
                # We're in an existing loop - use run_coroutine_threadsafe
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._async_compute_similarity_scores(batch_pairs))
                    return future.result()
            except RuntimeError:
                # No running loop - safe to use asyncio.run()
                return asyncio.run(self._async_compute_similarity_scores(batch_pairs))

        except Exception as e:
            self.logger.error(f"Failed to compute similarity scores asynchronously: {e}")
            # Fallback to sequential computation
            return [(self.similarity_score(pair["text"], pair["source"]), pair["metadata"]) for pair in batch_pairs]

    async def _async_compute_similarity_scores(
        self, batch_pairs: List[Dict[str, Any]]
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Internal async function for computing similarity scores in batches.

        Processes pairs in batches to avoid overloading the embedding API,
        while maintaining parallel execution within each batch.
        """

        def compute_single_score(pair: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
            """Compute similarity score for a single pair (synchronous)"""
            try:
                similarity_score = self.similarity_score(pair["text"], pair["source"])
                return (similarity_score, pair["metadata"])
            except Exception as e:
                self.logger.warning(f"Error computing similarity score: {str(e)}")
                return (0.5, pair["metadata"])  # Fallback value

        # Get the current event loop
        loop = asyncio.get_running_loop()

        # Process pairs in batches to avoid overloading API
        all_results = []
        num_batches = (len(batch_pairs) + self.SIMILARITY_BATCH_SIZE - 1) // self.SIMILARITY_BATCH_SIZE

        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.SIMILARITY_BATCH_SIZE
            batch_end = min(batch_start + self.SIMILARITY_BATCH_SIZE, len(batch_pairs))
            batch_data = batch_pairs[batch_start:batch_end]

            self.logger.debug(
                f"Processing similarity batch {batch_idx + 1}/{num_batches} "
                f"({len(batch_data)} pairs, indices {batch_start}-{batch_end - 1})"
            )

            # Execute batch computations concurrently in thread pool
            batch_tasks = [loop.run_in_executor(None, compute_single_score, pair) for pair in batch_data]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions in batch results
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Error in similarity batch {batch_idx + 1}, pair {i}: {str(result)}")
                    all_results.append((0.5, batch_data[i]["metadata"]))  # Fallback
                else:
                    all_results.append(result)

        self.logger.debug(f"Completed async similarity computation: {len(all_results)} results")
        return all_results
