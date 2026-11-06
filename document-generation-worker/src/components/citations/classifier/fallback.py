from typing import Dict

from document_common.schemas.citations import CitationLabel, ClassificationResult
from document_common.utils.logger import Logger


class FallbackClassifier:
    """
    Rule-based fallback classifier for citation classification.

    Uses similarity score thresholds to classify citations when LLM fails.
    Provides deterministic classification based on configured thresholds.
    """

    def __init__(
        self,
        quote_threshold: float = 0.95,
        close_paraphrase_threshold: float = 0.85,
        loose_paraphrase_threshold: float = 0.70,
    ):
        self.logger = Logger().get_logger("fallback_classifier")
        self.quote_threshold = quote_threshold
        self.close_paraphrase_threshold = close_paraphrase_threshold
        self.loose_paraphrase_threshold = loose_paraphrase_threshold

        self.logger.debug(
            "FallbackClassifier initialized (thresholds: " +
            f"quote={quote_threshold}, close={close_paraphrase_threshold}, " +
            f"loose={loose_paraphrase_threshold})"
        )

    def classify(self, generated_text: str, source_text: str, similarity_score: float) -> ClassificationResult:
        """
        Classify citation relationship using rule-based similarity thresholds.

        Args:
            generated_text: Text from generated document
            source_text: Text from source document
            similarity_score: Semantic similarity score (0.0-1.0)

        Returns:
            ClassificationResult with type, confidence, and reasoning
        """
        # Normalize similarity score to 0-1 range
        normalized_score = max(0.0, min(1.0, similarity_score))

        # Apply threshold-based classification
        if normalized_score >= self.quote_threshold:
            classification = CitationLabel.DIRECT_QUOTE
            confidence = min(0.95, normalized_score)
            reasoning = (
                f"Similarity score {normalized_score:.2f} exceeds direct quote threshold "
                f"{self.quote_threshold:.2f}, indicating near-identical text."
            )

        elif normalized_score >= self.close_paraphrase_threshold:
            classification = CitationLabel.CLOSE_PARAPHRASE
            confidence = min(0.90, normalized_score)
            reasoning = (
                f"Similarity score {normalized_score:.2f} indicates close paraphrase "
                f"(threshold: {self.close_paraphrase_threshold:.2f}), preserving core meaning "
                "with different wording."
            )

        elif normalized_score >= self.loose_paraphrase_threshold:
            classification = CitationLabel.LOOSE_PARAPHRASE
            confidence = min(0.85, normalized_score)
            reasoning = (
                f"Similarity score {normalized_score:.2f} suggests loose paraphrase "
                f"(threshold: {self.loose_paraphrase_threshold:.2f}), related ideas with "
                "significant differences."
            )

        else:
            classification = CitationLabel.INDEPENDENT
            confidence = max(0.70, 1.0 - normalized_score)
            reasoning = (
                f"Similarity score {normalized_score:.2f} below all thresholds, "
                "indicating independent content with no meaningful citation relationship."
            )

        self.logger.debug(
            f"Fallback classification: {classification.value} " +
            f"(score: {normalized_score:.2f}, confidence: {confidence:.2f})"
        )

        return ClassificationResult(classification=classification, confidence=confidence, reasoning=reasoning)

    def update_thresholds(
        self,
        quote_threshold: float = None,
        close_paraphrase_threshold: float = None,
        loose_paraphrase_threshold: float = None,
    ) -> None:
        """
        Update classification thresholds dynamically.

        Args:
            quote_threshold: New DIRECT_QUOTE threshold
            close_paraphrase_threshold: New CLOSE_PARAPHRASE threshold
            loose_paraphrase_threshold: New LOOSE_PARAPHRASE threshold
        """
        if quote_threshold is not None:
            self.quote_threshold = quote_threshold
        if close_paraphrase_threshold is not None:
            self.close_paraphrase_threshold = close_paraphrase_threshold
        if loose_paraphrase_threshold is not None:
            self.loose_paraphrase_threshold = loose_paraphrase_threshold

        self.logger.debug(
            f"Thresholds updated: quote={self.quote_threshold}, " +
            f"close={self.close_paraphrase_threshold}, " +
            f"loose={self.loose_paraphrase_threshold}"
        )

    def get_thresholds(self) -> Dict[str, float]:
        """Get current threshold configuration."""
        return {
            "quote_threshold": self.quote_threshold,
            "close_paraphrase_threshold": self.close_paraphrase_threshold,
            "loose_paraphrase_threshold": self.loose_paraphrase_threshold,
        }
