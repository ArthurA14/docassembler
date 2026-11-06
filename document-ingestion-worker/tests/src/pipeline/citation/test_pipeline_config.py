"""
Tests for CitationPipeline configuration handling.

Verifies that configuration from citations.yaml is properly passed to components.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.citation import CitationPipeline


class TestCitationPipelineConfiguration:
    """Test configuration propagation from YAML to components."""

    @patch("src.pipeline.citation.pipeline.config")
    def test_config_passed_to_preprocessor(self, mock_config):
        """Test that preprocessing config is passed to TextPreprocessor."""
        # Mock config
        mock_citations_config = MagicMock()
        mock_citations_config.preprocessing.get.return_value = 30  # Custom min_sentence_length
        mock_citations_config.classification.get.side_effect = lambda k, d: d
        mock_citations_config.stacking.get.side_effect = lambda k, d: d
        mock_citations_config.validation.get.side_effect = lambda k, d: d
        mock_citations_config.get.side_effect = lambda k, d: d

        mock_config.get_current_config.return_value.citations = mock_citations_config

        # Create pipeline
        with patch("src.pipeline.citation.pipeline.TextPreprocessor") as MockPreprocessor:
            pipeline = CitationPipeline()

            # Verify TextPreprocessor was called with config value
            MockPreprocessor.assert_called_once_with(min_sentence_length=30)

    @patch("src.pipeline.citation.pipeline.config")
    def test_config_passed_to_classifier(self, mock_config):
        """Test that classification config is passed to CitationClassifier."""
        # Mock config with custom values
        mock_citations_config = MagicMock()
        mock_citations_config.preprocessing.get.side_effect = lambda k, d: d
        mock_citations_config.classification.get.side_effect = {
            "prompt_name": "custom_prompt",
            "max_retries": 5,
            "confidence_threshold": 0.85,
        }.get
        mock_citations_config.stacking.get.side_effect = lambda k, d: d
        mock_citations_config.validation.get.side_effect = lambda k, d: d
        mock_citations_config.get.side_effect = lambda k, d: d

        mock_config.get_current_config.return_value.citations = mock_citations_config

        # Create pipeline
        with patch("src.pipeline.citation.pipeline.CitationClassifier") as MockClassifier:
            with patch("src.pipeline.citation.pipeline.TextPreprocessor"):
                pipeline = CitationPipeline()

            # Verify CitationClassifier was called with config values
            MockClassifier.assert_called_once_with(
                prompt_name="custom_prompt",
                max_retries=5,
                confidence_threshold=0.85,
            )

    @patch("src.pipeline.citation.pipeline.config")
    def test_config_passed_to_aggregator(self, mock_config):
        """Test that stacking config is passed to CitationAggregator."""
        # Mock config with custom values
        mock_citations_config = MagicMock()
        mock_citations_config.preprocessing.get.side_effect = lambda k, d: d
        mock_citations_config.classification.get.side_effect = lambda k, d: d
        mock_citations_config.stacking.get.side_effect = {
            "max_stack_size": 10,
            "min_stack_confidence": 0.90,
            "boost_factor": 0.10,
            "boost_method": "progressive",
            "ranking_strategy": "confidence",
            "deduplication_method": "semantic",
            "fuzzy_threshold": 90.0,
        }.get
        mock_citations_config.validation.get.side_effect = lambda k, d: d
        mock_citations_config.get.side_effect = lambda k, d: d

        mock_config.get_current_config.return_value.citations = mock_citations_config

        # Create pipeline
        with patch("src.pipeline.citation.pipeline.CitationAggregator") as MockAggregator:
            with patch("src.pipeline.citation.pipeline.TextPreprocessor"):
                with patch("src.pipeline.citation.pipeline.CitationClassifier"):
                    pipeline = CitationPipeline()

            # Verify CitationAggregator was called with config values
            MockAggregator.assert_called_once_with(
                max_stack_size=10,
                min_stack_confidence=0.90,
                boost_factor=0.10,
                boost_method="progressive",
                ranking_strategy="confidence",
                deduplication_method="semantic",
                fuzzy_threshold=90.0,
            )

    @patch("src.pipeline.citation.pipeline.config")
    def test_config_passed_to_validator(self, mock_config):
        """Test that validation config is passed to CitationValidator."""
        # Mock config with custom values
        mock_citations_config = MagicMock()
        mock_citations_config.preprocessing.get.side_effect = lambda k, d: d
        mock_citations_config.classification.get.side_effect = lambda k, d: d
        mock_citations_config.stacking.get.side_effect = lambda k, d: d
        mock_citations_config.validation.get.side_effect = {
            "require_exact_quotes": False,
            "allow_whitespace_normalize": False,
            "max_offset_drift": 20,
        }.get
        mock_citations_config.get.side_effect = lambda k, d: d

        mock_config.get_current_config.return_value.citations = mock_citations_config

        # Create pipeline
        with patch("src.pipeline.citation.pipeline.CitationValidator") as MockValidator:
            with patch("src.pipeline.citation.pipeline.TextPreprocessor"):
                with patch("src.pipeline.citation.pipeline.CitationClassifier"):
                    with patch("src.pipeline.citation.pipeline.CitationAggregator"):
                        pipeline = CitationPipeline()

            # Verify CitationValidator was called with config values
            MockValidator.assert_called_once_with(
                validate_quote_match=False,
                validate_type_consistency=True,
                validate_confidence=True,
                allow_whitespace_normalize=False,
                max_offset_drift=20,
                fail_on_error=False,  # Default value
            )

    @patch("src.pipeline.citation.pipeline.config")
    def test_mvp_mode_from_config(self, mock_config):
        mock_citations_config = MagicMock()
        mock_citations_config.preprocessing.get.side_effect = lambda k, d: d
        mock_citations_config.classification.get.side_effect = lambda k, d: d
        mock_citations_config.stacking.get.side_effect = lambda k, d: d
        mock_citations_config.validation.get.side_effect = lambda k, d: d

        mock_config.get_current_config.return_value.citations = mock_citations_config
        with patch("src.pipeline.citation.pipeline.TextPreprocessor"):
            with patch("src.pipeline.citation.pipeline.CitationClassifier"):
                with patch("src.pipeline.citation.pipeline.CitationAggregator"):
                    with patch("src.pipeline.citation.pipeline.CitationValidator"):
                        with patch("src.pipeline.citation.pipeline.TextPairReranker"):
                            pipeline = CitationPipeline()

        # Should be in production mode (mvp_mode = False) based on config
        assert pipeline.mvp_mode == False

    @patch("src.pipeline.citation.pipeline.config")
    def test_mvp_mode_constructor_override(self, mock_config):
        """Test that constructor parameter overrides config"""
        mock_citations_config = MagicMock()
        mock_citations_config.preprocessing.get.side_effect = lambda k, d: d
        mock_citations_config.classification.get.side_effect = lambda k, d: d
        mock_citations_config.stacking.get.side_effect = lambda k, d: d
        mock_citations_config.validation.get.side_effect = lambda k, d: d

        mock_config.get_current_config.return_value.citations = mock_citations_config

        with patch("src.pipeline.citation.pipeline.TextPreprocessor"):
            with patch("src.pipeline.citation.pipeline.CitationClassifier"):
                with patch("src.pipeline.citation.pipeline.CitationAggregator"):
                    with patch("src.pipeline.citation.pipeline.CitationValidator"):
                        pipeline = CitationPipeline()

        # Should be in MVP mode despite config saying production
        assert pipeline.mvp_mode == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
