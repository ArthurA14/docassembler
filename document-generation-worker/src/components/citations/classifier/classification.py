"""
Citation classification component using BatchedOperations for efficient batch processing.

This module provides a Haystack component for classifying citation relationships
using LLM with structured output. Now uses BatchedOperations for elegant batching
while maintaining perfect Haystack component compatibility.
"""

import time
from typing import Any, Dict, List, Optional

from haystack import component
from pydantic import ValidationError
from document_common.components.base.batched_operations import BatchedOperations
from document_common.components.base.assembly import StructuredAssembly
from document_common.schemas.citations import CitationLabel, ClassificationResult
from document_common.utils.logger import Logger

from config import config
from src.components.citations.classifier.fallback import FallbackClassifier


@component
class CitationClassifier:
    """
    Haystack component for classifying citation relationships using LLM with structured output.

    Uses structured output mode to classify citations as DIRECT_QUOTE,
    CLOSE_PARAPHRASE, LOOSE_PARAPHRASE, or INDEPENDENT. Implements retry logic with
    error feedback and falls back to rule-based classification on repeated failures.

    Features:
    - BatchedOperations integration for efficient batch processing
    - Perfect Haystack @component compatibility (no metaclass conflicts)
    - Configurable batch size and progress tracking
    - Enhanced error handling and retry logic
    - Comprehensive metrics collection
    - Backward compatibility with existing APIs

    Example:
        ```python
        # Single classification
        classifier = CitationClassifier()
        result = classifier.run("generated text", "source text", 0.85)

        # Batch classification
        pairs = [
            {"text": "text1", "source": "source1", "score": 0.8},
            {"text": "text2", "source": "source2", "score": 0.7},
        ]
        results = classifier.batch_classify(pairs)
        ```
    """

    def __init__(
        self,
        prompt_name: str = "citation.classification",
        max_retries: int = 3,
        confidence_threshold: float = 0.70,
        batch_size: int = 128,
    ):
        """
        Initialize the citation classifier.

        Args:
            prompt_name: Name of prompt template in PromptProvider
            max_retries: Maximum retry attempts on LLM failure
            confidence_threshold: Minimum confidence to accept classification
            batch_size: Number of pairs to process concurrently in batches
        """
        self.logger = Logger().get_logger("citation_classifier")
        citation_config = config.get_current_config().citations.classification

        self.prompt_name = prompt_name
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold or citation_config.get("confidence_threshold", 0.70)

        # Initialize structured assembly component
        self.generator = StructuredAssembly(
            prompt_name=prompt_name,
            base_model=ClassificationResult,
            fixed_inputs=None,
        )

        # Initialize fallback classifier
        self.fallback_classifier = FallbackClassifier(
            quote_threshold=citation_config.get("quote_match_threshold", 0.95),
            close_paraphrase_threshold=citation_config.get("close_paraphrase_threshold", 0.85),
            loose_paraphrase_threshold=citation_config.get("loose_paraphrase_threshold", 0.70),
        )

        # NEW: Initialize BatchedOperations for internal batch processing
        self._batcher = BatchedOperations(batch_size=batch_size, show_progress=True)

        # Metrics tracking
        self.total_classifications = 0
        self.llm_successes = 0
        self.fallback_uses = 0
        self.retry_counts = []

        self.logger.debug(
            f"CitationClassifier initialized (prompt: {prompt_name}, " +
            f"max_retries: {max_retries}, confidence_threshold: {self.confidence_threshold}, " +
            f"batch_size: {batch_size})"
        )

    @component.output_types(classification=ClassificationResult)
    def run(self, generated_text: str, source_text: str, similarity_score: float) -> Dict[str, ClassificationResult]:
        """
        Classify the citation relationship between generated and source text.

        Args:
            generated_text: Text from generated document
            source_text: Text from source document
            similarity_score: Semantic similarity score (0.0-1.0)

        Returns:
            Dict with 'classification' key containing ClassificationResult
        """
        self.total_classifications += 1
        start_time = time.time()

        # Try LLM classification with retry logic
        classification = self._classify_with_llm_retry(generated_text, source_text, similarity_score)

        # If LLM fails, use fallback
        if classification is None:
            self.logger.warning("LLM classification failed after retries, using fallback")
            self.fallback_uses += 1
            classification = self.fallback_classifier.classify(generated_text, source_text, similarity_score)

        # Check confidence threshold
        if classification.confidence < self.confidence_threshold:
            self.logger.debug(
                f"Classification confidence {classification.confidence:.2f} below threshold " +
                f"{self.confidence_threshold:.2f}, marking as low confidence"
            )

        elapsed = time.time() - start_time
        self.logger.debug(
            f"Classification: {classification.classification.value} " +
            f"(confidence: {classification.confidence:.2f}, latency: {elapsed:.3f}s)"
        )

        return {"classification": classification}

    def _classify_with_llm_retry(
        self, generated_text: str, source_text: str, similarity_score: float
    ) -> Optional[ClassificationResult]:
        """
        Attempt LLM classification with retry logic and error feedback.

        Args:
            generated_text: Generated document text
            source_text: Source document text
            similarity_score: Semantic similarity score

        Returns:
            ClassificationResult or None if all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Prepare inputs for prompt
                inputs = {
                    "generated_text": generated_text,
                    "source_text": source_text,
                    "similarity_score": f"{similarity_score:.2f}",
                }

                # Add error feedback on retries
                if last_error and attempt > 0:
                    inputs["error_feedback"] = (
                        f"Previous attempt failed with error: {last_error}. "
                        "Please ensure your response is valid JSON matching the required schema."
                    )
                    self.logger.debug(f"Retry {attempt + 1}/{self.max_retries} with error feedback")

                # Call LLM with structured output
                result = self.generator.run(inputs=inputs)
                classification = result["structured_reply"]

                # Validate it's a ClassificationResult
                if not isinstance(classification, ClassificationResult):
                    raise ValueError(f"Expected ClassificationResult, got {type(classification)}")

                # Validate classification type is valid
                if classification.classification not in CitationLabel:
                    raise ValueError(f"Invalid classification type: {classification.classification}")

                # Success
                self.llm_successes += 1
                self.retry_counts.append(attempt)

                if attempt > 0:
                    self.logger.debug(f"LLM classification succeeded on retry {attempt + 1}")

                return classification

            except ValidationError as e:
                last_error = f"Validation error: {str(e)}"
                self.logger.warning(f"Attempt {attempt + 1} validation failed: {e}")

            except ValueError as e:
                last_error = f"Value error: {str(e)}"
                self.logger.warning(f"Attempt {attempt + 1} value error: {e}")

            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                self.logger.error(f"Attempt {attempt + 1} unexpected error: {e}")

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                wait_time = 0.5 * (2**attempt)  # 0.5s, 1s, 2s
                time.sleep(wait_time)

        self.logger.error(f"LLM classification failed after {self.max_retries} attempts: {last_error}")
        self.retry_counts.append(self.max_retries)
        return None

    def batch_classify(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify multiple (text, source, score) pairs concurrently.

        NOW USES: BatchedOperations for elegant batching while maintaining exact API compatibility.
        This method provides the same interface as the original implementation but uses
        the internal BatchedOperations instance for improved error handling and maintainability.

        Args:
            pairs: List of dictionaries with 'text', 'source', and 'score' keys

        Returns:
            List of dictionaries with 'classification' and 'error' keys, maintaining
            the exact same format as the original implementation

        Example:
            ```python
            classifier = CitationClassifier()
            pairs = [
                {"text": "generated_text", "source": "source_text", "score": 0.8},
                {"text": "another_text", "source": "another_source", "score": 0.7},
            ]
            results = classifier.batch_classify(pairs)
            # results format: [{"classification": ClassificationResult(...), "error": None}, ...]
            ```
        """
        if not pairs:
            self.logger.debug("No pairs to classify")
            return []

        self.logger.debug(f"Starting batch classification of {len(pairs)} pairs using BatchedOperations")

        # Prepare arguments for BatchedOperations
        positional_args_list = []
        keyword_args_list = []

        for pair in pairs:
            # Extract arguments from pair
            generated_text = pair.get("text", "")
            source_text = pair.get("source", "")
            similarity_score = pair.get("score", 0.0)

            # Add to batch lists
            positional_args_list.append([generated_text, source_text, similarity_score])
            keyword_args_list.append(None)  # No additional keyword arguments

        # Use BatchedOperations to batch the calls
        results = self._batcher.batch_call(
            func=self.run, positional_args_list=positional_args_list, keyword_args_list=keyword_args_list
        )

        # Convert results to expected format
        formatted_results = []
        for result in results:
            if isinstance(result, Exception):
                formatted_results.append({
                    "classification": None,
                    "error": str(result),
                })
            else:
                formatted_results.append({
                    "classification": result.get("classification"),
                    "error": None,
                })

        # Log completion statistics
        successful = sum(1 for r in formatted_results if r.get("error") is None)
        failed = len(formatted_results) - successful

        self.logger.debug(
            f"Batch classification complete: {successful} successful, {failed} failed " +
            f"(total pairs: {len(pairs)}, using BatchedOperations with batch_size={self._batcher.batch_size})"
        )

        return formatted_results

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get classification metrics for monitoring.

        Returns:
            Dict with classification statistics, now including batch processing metrics
        """
        avg_retries = sum(self.retry_counts) / len(self.retry_counts) if self.retry_counts else 0

        # Original classification metrics
        original_metrics = {
            "total_classifications": self.total_classifications,
            "llm_successes": self.llm_successes,
            "fallback_uses": self.fallback_uses,
            "llm_success_rate": self.llm_successes / self.total_classifications
            if self.total_classifications > 0
            else 0,
            "fallback_rate": self.fallback_uses / self.total_classifications if self.total_classifications > 0 else 0,
            "avg_retries": avg_retries,
        }

        # Add batch processing metrics
        batch_stats = self._batcher.get_stats()

        return {
            **original_metrics,
            "batch_processing": {
                "batch_size": batch_stats["batch_size"],
                "supports_async": batch_stats["supports_async"],
                "supports_sync": batch_stats["supports_sync"],
                "error_handling": batch_stats["error_handling"],
            },
        }

    def reset_metrics(self) -> None:
        """Reset all classification metrics."""
        self.total_classifications = 0
        self.llm_successes = 0
        self.fallback_uses = 0
        self.retry_counts = []
        self.logger.debug("Classification metrics reset")

    def get_batch_stats(self) -> Dict[str, Any]:
        """
        Get detailed batch processing statistics.

        Returns:
            Dict with detailed batch processor statistics and performance metrics
        """
        return self._batcher.get_stats()
