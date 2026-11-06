from unittest.mock import MagicMock, patch

import pytest

from src.components.citations.classifier.classification import CitationClassifier
from src.components.citations.classifier.fallback import FallbackClassifier
from src.schemas.citations import CitationLabel, ClassificationResult


class TestFallbackClassifier:
    def test_direct_quote_classification(self):
        classifier = FallbackClassifier(
            quote_threshold=0.95,
            close_paraphrase_threshold=0.85,
            loose_paraphrase_threshold=0.70,
        )

        result = classifier.classify(
            generated_text="Machine learning requires large datasets.",
            source_text="Machine learning requires large datasets.",
            similarity_score=0.98,
        )

        assert result.classification == CitationLabel.DIRECT_QUOTE
        assert result.confidence >= 0.90
        assert "direct quote" in result.reasoning.lower()

    def test_close_paraphrase_classification(self):
        classifier = FallbackClassifier()

        result = classifier.classify(
            generated_text="Neural networks need substantial training data.",
            source_text="Machine learning requires large datasets.",
            similarity_score=0.87,
        )

        assert result.classification == CitationLabel.CLOSE_PARAPHRASE
        assert result.confidence >= 0.80
        assert "close paraphrase" in result.reasoning.lower()

    def test_loose_paraphrase_classification(self):
        classifier = FallbackClassifier()

        result = classifier.classify(
            generated_text="Data quality is important for ML models.",
            source_text="Machine learning requires large datasets.",
            similarity_score=0.75,
        )

        assert result.classification == CitationLabel.LOOSE_PARAPHRASE
        assert result.confidence >= 0.70
        assert "loose paraphrase" in result.reasoning.lower()

    def test_independent_classification(self):
        classifier = FallbackClassifier()

        result = classifier.classify(
            generated_text="The Renaissance was a cultural movement.",
            source_text="Machine learning requires large datasets.",
            similarity_score=0.15,
        )

        assert result.classification == CitationLabel.INDEPENDENT
        assert result.confidence >= 0.70
        assert "independent" in result.reasoning.lower()

    def test_boundary_conditions(self):
        classifier = FallbackClassifier(
            quote_threshold=0.95,
            close_paraphrase_threshold=0.85,
            loose_paraphrase_threshold=0.70,
        )

        # Exactly at quote threshold
        result = classifier.classify("test", "test", 0.95)
        assert result.classification == CitationLabel.DIRECT_QUOTE

        # Just below quote threshold
        result = classifier.classify("test", "test", 0.94)
        assert result.classification == CitationLabel.CLOSE_PARAPHRASE

        # Exactly at close threshold
        result = classifier.classify("test", "test", 0.85)
        assert result.classification == CitationLabel.CLOSE_PARAPHRASE

        # Just below close threshold
        result = classifier.classify("test", "test", 0.84)
        assert result.classification == CitationLabel.LOOSE_PARAPHRASE

    def test_update_thresholds(self):
        classifier = FallbackClassifier(
            quote_threshold=0.95,
            close_paraphrase_threshold=0.85,
            loose_paraphrase_threshold=0.70,
        )

        classifier.update_thresholds(quote_threshold=0.90, close_paraphrase_threshold=0.80)

        thresholds = classifier.get_thresholds()
        assert thresholds["quote_threshold"] == 0.90
        assert thresholds["close_paraphrase_threshold"] == 0.80
        assert thresholds["loose_paraphrase_threshold"] == 0.70

    def test_score_normalization(self):
        classifier = FallbackClassifier()

        # Score > 1.0 should be clamped
        result = classifier.classify("test", "test", 1.5)
        assert result.classification in CitationLabel

        # Score < 0.0 should be clamped
        result = classifier.classify("test", "test", -0.5)
        assert result.classification == CitationLabel.INDEPENDENT


class TestCitationClassifier:
    @patch("src.components.citations.classifier.classification.config")
    def test_initialization(self, mock_config):
        mock_config.get_current_config.return_value.citations.classification = {
            "confidence_threshold": 0.75,
            "quote_match_threshold": 0.95,
            "close_paraphrase_threshold": 0.85,
            "loose_paraphrase_threshold": 0.70,
        }

        classifier = CitationClassifier(
            prompt_name="test_prompt",
            max_retries=2,
            confidence_threshold=0.80,
        )

        assert classifier.prompt_name == "test_prompt"
        assert classifier.max_retries == 2
        assert classifier.confidence_threshold == 0.80

    @patch("src.components.citations.classifier.classification.config")
    def test_successful_llm_classification(self, mock_config):
        mock_config.get_current_config.return_value.citations.classification = {
            "confidence_threshold": 0.70,
            "quote_match_threshold": 0.95,
            "close_paraphrase_threshold": 0.85,
            "loose_paraphrase_threshold": 0.70,
        }

        classifier = CitationClassifier()

        # Mock successful LLM response
        mock_result = ClassificationResult(
            classification=CitationLabel.DIRECT_QUOTE,
            confidence=0.95,
            reasoning="Texts are identical.",
        )

        classifier.generator.run = MagicMock(return_value={"structured_reply": mock_result})

        result = classifier.run(
            generated_text="Test text",
            source_text="Test text",
            similarity_score=0.98,
        )

        assert result["classification"].classification == CitationLabel.DIRECT_QUOTE
        assert result["classification"].confidence == pytest.approx(0.95)
        assert classifier.llm_successes == 1
        assert classifier.fallback_uses == 0

    @patch("src.components.citations.classifier.classification.config")
    def test_llm_retry_logic(self, mock_config):
        mock_config.get_current_config.return_value.citations.classification = {
            "confidence_threshold": 0.70,
            "quote_match_threshold": 0.95,
            "close_paraphrase_threshold": 0.85,
            "loose_paraphrase_threshold": 0.70,
        }

        classifier = CitationClassifier(max_retries=3)

        # Mock first two attempts fail, third succeeds
        mock_result = ClassificationResult(
            classification=CitationLabel.CLOSE_PARAPHRASE,
            confidence=0.88,
            reasoning="Close semantic match.",
        )

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Mock error")
            return {"structured_reply": mock_result}

        classifier.generator.run = MagicMock(side_effect=side_effect)

        result = classifier.run(
            generated_text="Neural networks need data.",
            source_text="ML algorithms require datasets.",
            similarity_score=0.87,
        )

        assert result["classification"].classification == CitationLabel.CLOSE_PARAPHRASE
        assert call_count == 3
        assert classifier.llm_successes == 1

    @patch("src.components.citations.classifier.classification.config")
    @patch("src.components.citations.classifier.classification.time")
    def test_fallback_on_llm_failure(self, mock_time, mock_config):
        mock_time.time.return_value = 0
        mock_time.sleep = MagicMock()

        mock_config.get_current_config.return_value.citations.classification = {
            "confidence_threshold": 0.70,
            "quote_match_threshold": 0.95,
            "close_paraphrase_threshold": 0.85,
            "loose_paraphrase_threshold": 0.70,
        }

        classifier = CitationClassifier(max_retries=2)

        # Mock all LLM attempts fail
        classifier.generator.run = MagicMock(side_effect=Exception("API failure"))

        result = classifier.run(
            generated_text="Test text",
            source_text="Test text",
            similarity_score=0.98,
        )

        # Should use fallback classifier
        assert result["classification"].classification == CitationLabel.DIRECT_QUOTE
        assert classifier.fallback_uses == 1
        assert classifier.llm_successes == 0

    @patch("src.components.citations.classifier.classification.config")
    def test_confidence_threshold_check(self, mock_config):
        mock_config.get_current_config.return_value.citations.classification = {
            "confidence_threshold": 0.80,
            "quote_match_threshold": 0.95,
            "close_paraphrase_threshold": 0.85,
            "loose_paraphrase_threshold": 0.70,
        }

        classifier = CitationClassifier(confidence_threshold=0.80)

        # Mock low confidence result
        mock_result = ClassificationResult(
            classification=CitationLabel.LOOSE_PARAPHRASE,
            confidence=0.65,
            reasoning="Low confidence match.",
        )

        classifier.generator.run = MagicMock(return_value={"structured_reply": mock_result})

        result = classifier.run(
            generated_text="Text A",
            source_text="Text B",
            similarity_score=0.70,
        )

        # Result should still be returned but confidence is below threshold
        assert result["classification"].confidence < 0.80

    @patch("src.components.citations.classifier.classification.config")
    def test_get_metrics(self, mock_config):
        mock_config.get_current_config.return_value.citations.classification = {
            "confidence_threshold": 0.70,
            "quote_match_threshold": 0.95,
            "close_paraphrase_threshold": 0.85,
            "loose_paraphrase_threshold": 0.70,
        }

        classifier = CitationClassifier()

        # Simulate some classifications
        classifier.total_classifications = 10
        classifier.llm_successes = 7
        classifier.fallback_uses = 3
        classifier.retry_counts = [0, 1, 0, 2, 0, 1, 0, 3, 3, 3]

        metrics = classifier.get_metrics()

        assert metrics["total_classifications"] == 10
        assert metrics["llm_successes"] == 7
        assert metrics["fallback_uses"] == 3
        assert metrics["llm_success_rate"] == 0.7
        assert metrics["fallback_rate"] == 0.3
        assert metrics["avg_retries"] == 1.3

    @patch("src.components.citations.classifier.classification.config")
    def test_reset_metrics(self, mock_config):
        mock_config.get_current_config.return_value.citations.classification = {
            "confidence_threshold": 0.70,
            "quote_match_threshold": 0.95,
            "close_paraphrase_threshold": 0.85,
            "loose_paraphrase_threshold": 0.70,
        }

        classifier = CitationClassifier()

        classifier.total_classifications = 10
        classifier.llm_successes = 7
        classifier.fallback_uses = 3

        classifier.reset_metrics()

        assert classifier.total_classifications == 0
        assert classifier.llm_successes == 0
        assert classifier.fallback_uses == 0
        assert len(classifier.retry_counts) == 0


class TestIntegration:
    @patch("src.components.citations.classifier.classification.config")
    def test_all_classification_types(self, mock_config):
        mock_config.get_current_config.return_value.citations.classification = {
            "confidence_threshold": 0.70,
            "quote_match_threshold": 0.95,
            "close_paraphrase_threshold": 0.85,
            "loose_paraphrase_threshold": 0.70,
        }

        classifier = CitationClassifier()

        test_cases = [
            (0.98, CitationLabel.DIRECT_QUOTE),
            (0.87, CitationLabel.CLOSE_PARAPHRASE),
            (0.72, CitationLabel.LOOSE_PARAPHRASE),
            (0.20, CitationLabel.INDEPENDENT),
        ]

        for score, expected_type in test_cases:
            mock_result = ClassificationResult(
                classification=expected_type,
                confidence=score,
                reasoning=f"Test classification for {expected_type.value}",
            )

            classifier.generator.run = MagicMock(return_value={"structured_reply": mock_result})

            result = classifier.run(
                generated_text="Test",
                source_text="Test",
                similarity_score=score,
            )

            assert result["classification"].classification == expected_type
