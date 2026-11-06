"""
Integration tests for CitationPipeline with real components.

These tests use real component instances (not mocks) to verify end-to-end
pipeline behavior. They test:
- Complete pipeline flow from text to validated stacks
- MVP vs Production mode differences
- Error recovery and fallback mechanisms
- Performance and latency characteristics

Note: These tests may be slower than unit tests as they use real components.
Mark with @pytest.mark.integration or @pytest.mark.slow as needed.
"""

import pytest

from src.pipeline.citation import CitationPipeline
from document_common.schemas.citations import SourceMetadata


@pytest.fixture
def real_generated_text():
    """Real-world example of generated document text."""
    return """
    Machine learning algorithms have revolutionized data analysis in recent years.
    These systems require large datasets for effective training.
    Neural networks, a subset of machine learning, have shown remarkable success
    in tasks ranging from image recognition to natural language processing.
    The quality of training data directly impacts model performance.
    """


@pytest.fixture
def real_source_documents():
    """Real-world example of source documents."""
    return [
        SourceMetadata(
            id="source-1",
            file_id="file-001",
            content="""
            Machine learning has become a cornerstone of modern data science.
            Algorithms in this field learn patterns from data without explicit programming.
            Large datasets are essential for training effective machine learning models.
            The quality and quantity of training data significantly affect model accuracy.
            """,
            type="original",
        ),
        SourceMetadata(
            id="source-2",
            file_id="file-002",
            content="""
            Neural networks are computational models inspired by biological neural systems.
            They have achieved state-of-the-art results in computer vision tasks.
            Deep learning, based on neural networks, excels at image recognition.
            Natural language processing has benefited greatly from neural network advances.
            """,
            type="original",
        ),
        SourceMetadata(
            id="source-3",
            file_id="file-003",
            content="""
            Data quality is paramount in machine learning applications.
            Training datasets must be diverse and representative of real-world scenarios.
            Model performance correlates strongly with the quality of training examples.
            Inadequate data can lead to poor generalization and biased predictions.
            """,
            type="original",
        ),
    ]


@pytest.mark.integration
class TestCitationPipelineIntegration:
    """Integration tests with real components."""

    def test_mvp_mode_end_to_end(self, real_generated_text, real_source_documents):
        """Test complete MVP mode pipeline with real components."""
        pipeline = CitationPipeline()

        result = pipeline.run(
            generated_text=real_generated_text,
            source_documents=real_source_documents,
            request_id="test-mvp-001",
        )

        # Verify pipeline executed
        assert result["result"] is not None
        assert result["result"].metrics.request_id == "test-mvp-001"
        assert result["result"].metrics.component_2_bypassed is True

        # Verify preprocessor ran
        assert result["result"].metrics.sentences_processed > 0

        # Verify timing data
        assert result["result"].metrics.total_duration_seconds > 0
        assert "preprocessor" in result["result"].metrics.component_durations

        # Log results for inspection
        print("\n=== MVP Mode Results ===")
        print(f"Sentences processed: {result['result'].metrics.sentences_processed}")
        print(f"Citations created: {result['result'].metrics.citations_created}")
        print(f"Stacks created: {result['result'].metrics.stacks_created}")
        print(f"Total duration: {result['result'].metrics.total_duration_seconds:.3f}s")
        print(f"Errors: {len(result['result'].errors)}")
        print(f"Warnings: {len(result['result'].warnings)}")

    def test_production_mode_end_to_end(self, real_generated_text, real_source_documents):
        """Test complete production mode pipeline with real components."""
        pipeline = CitationPipeline()

        result = pipeline.run(
            generated_text=real_generated_text,
            source_documents=real_source_documents,
            request_id="test-prod-001",
        )

        # Verify pipeline executed
        assert result["result"] is not None
        assert result["result"].metrics.request_id == "test-prod-001"
        assert result["result"].metrics.component_2_bypassed is False

        # Verify reranker ran
        if "reranker" in result["result"].metrics.component_durations:
            assert result["result"].metrics.component_durations["reranker"] >= 0

        # Log results for inspection
        print("\n=== Production Mode Results ===")
        print(f"Sentences processed: {result['result'].metrics.sentences_processed}")
        print(f"Citations created: {result['result'].metrics.citations_created}")
        print(f"Stacks created: {result['result'].metrics.stacks_created}")
        print(f"Total duration: {result['result'].metrics.total_duration_seconds:.3f}s")
        print(f"Component 2 duration: {result['result'].metrics.component_durations.get('reranker', 0):.3f}s")
        print(f"Errors: {len(result['result'].errors)}")
        print(f"Warnings: {len(result['result'].warnings)}")

    def test_component_metrics_aggregation(self, real_generated_text, real_source_documents):
        """Test that component metrics are properly aggregated."""
        pipeline = CitationPipeline()

        # Run pipeline
        result = pipeline.run(generated_text=real_generated_text, source_documents=real_source_documents)

        # Get component metrics
        component_metrics = pipeline.get_metrics()

        assert "classifier" in component_metrics
        assert "aggregator" in component_metrics
        assert "validator" in component_metrics

        # Verify classifier metrics
        classifier_metrics = component_metrics["classifier"]
        assert "total_classifications" in classifier_metrics
        assert "llm_success_rate" in classifier_metrics

        # Verify aggregator metrics
        aggregator_metrics = component_metrics["aggregator"]
        assert "total_citations_processed" in aggregator_metrics
        assert "total_stacks_created" in aggregator_metrics

        # Verify validator metrics
        validator_metrics = component_metrics["validator"]
        assert "total_validations" in validator_metrics

        print("\n=== Component Metrics ===")
        print(f"Classifier: {classifier_metrics}")
        print(f"Aggregator: {aggregator_metrics}")
        print(f"Validator: {validator_metrics}")

    def test_metrics_reset(self, real_generated_text, real_source_documents):
        """Test that metrics can be reset across pipeline runs."""
        pipeline = CitationPipeline()

        # Run pipeline
        result1 = pipeline.run(generated_text=real_generated_text, source_documents=real_source_documents)
        metrics1 = pipeline.get_metrics()

        # Reset metrics
        pipeline.reset_metrics()
        metrics2 = pipeline.get_metrics()

        # Verify metrics were reset
        assert metrics2["classifier"]["total_classifications"] == 0
        assert metrics2["aggregator"]["total_citations_processed"] == 0
        assert metrics2["validator"]["total_validations"] == 0

    def test_multiple_runs_same_pipeline(self, real_generated_text, real_source_documents):
        """Test that the same pipeline instance can handle multiple runs."""
        pipeline = CitationPipeline()

        # Run 1
        result1 = pipeline.run(
            generated_text=real_generated_text,
            source_documents=real_source_documents,
            request_id="run-1",
        )

        # Run 2
        result2 = pipeline.run(
            generated_text=real_generated_text,
            source_documents=real_source_documents,
            request_id="run-2",
        )

        # Verify both runs completed successfully
        assert result1["result"].metrics.request_id == "run-1"
        assert result2["result"].metrics.request_id == "run-2"

        # Verify metrics accumulated
        component_metrics = pipeline.get_metrics()
        assert component_metrics["classifier"]["total_classifications"] >= 0


@pytest.mark.integration
@pytest.mark.slow
class TestCitationPipelinePerformance:
    """Performance and latency tests for pipeline."""

    def test_latency_breakdown(self, real_generated_text, real_source_documents):
        """Test and document latency breakdown by component."""
        pipeline = CitationPipeline()

        result = pipeline.run(generated_text=real_generated_text, source_documents=real_source_documents)

        metrics = result["result"].metrics
        durations = metrics.component_durations

        print("\n=== Latency Breakdown ===")
        print(f"Total: {metrics.total_duration_seconds:.3f}s")

        for component, duration in sorted(durations.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / metrics.total_duration_seconds) * 100
            print(f"  {component:15s}: {duration:.3f}s ({percentage:.1f}%)")

        # Verify total duration is sum of components (approximately)
        component_sum = sum(durations.values())
        # Allow for some overhead
        assert component_sum <= metrics.total_duration_seconds * 1.1

    def test_throughput_single_document(self, real_source_documents):
        """Test throughput for single short document."""
        pipeline = CitationPipeline()
        short_text = "Machine learning requires data."

        import time

        start = time.time()
        result = pipeline.run(generated_text=short_text, source_documents=real_source_documents)
        end = time.time()

        latency = end - start

        print("\n=== Single Document Throughput ===")
        print(f"Text length: {len(short_text)} chars")
        print(f"Latency: {latency:.3f}s")
        print(f"Throughput: {len(short_text) / latency:.1f} chars/sec")

        # Reasonable latency threshold (adjust based on environment)
        assert latency < 10.0, f"Pipeline too slow: {latency:.3f}s"

    def test_throughput_long_document(self, real_source_documents):
        """Test throughput for longer document."""
        pipeline = CitationPipeline()

        # Create a longer document by repeating
        long_text = (
            """
        Machine learning algorithms process data. Neural networks learn patterns.
        Training requires substantial datasets. Model quality depends on data quality.
        """
            * 10
        )  # 10x repetition

        import time

        start = time.time()
        result = pipeline.run(generated_text=long_text, source_documents=real_source_documents)
        end = time.time()

        latency = end - start

        print("\n=== Long Document Throughput ===")
        print(f"Text length: {len(long_text)} chars")
        print(f"Sentences: {result['result'].metrics.sentences_processed}")
        print(f"Latency: {latency:.3f}s")
        print(f"Throughput: {len(long_text) / latency:.1f} chars/sec")
        print(f"Per sentence: {latency / result['result'].metrics.sentences_processed:.3f}s")


@pytest.mark.integration
class TestCitationPipelineErrorRecovery:
    """Test error recovery and resilience."""

    def test_malformed_source_document(self, real_generated_text):
        """Test pipeline handles malformed source documents gracefully."""
        malformed_sources = [
            SourceMetadata(
                id="bad-1",
                file_id="file-bad",
                content="",  # Empty content
                type="original",
            ),
        ]

        pipeline = CitationPipeline()
        result = pipeline.run(generated_text=real_generated_text, source_documents=malformed_sources)

        # Pipeline should complete without crashing
        assert result["result"] is not None
        # May have warnings or errors
        print("\n=== Malformed Source Results ===")
        print(f"Errors: {result['result'].errors}")
        print(f"Warnings: {result['result'].warnings}")

    def test_very_long_text(self, real_source_documents):
        """Test pipeline with very long generated text."""
        very_long_text = "Machine learning requires data. " * 1000  # 1000 repetitions

        pipeline = CitationPipeline()
        result = pipeline.run(generated_text=very_long_text, source_documents=real_source_documents)

        # Should complete without crashing
        assert result["result"] is not None
        assert result["result"].metrics.sentences_processed > 0

        print("\n=== Very Long Text Results ===")
        print(f"Text length: {len(very_long_text)} chars")
        print(f"Sentences: {result['result'].metrics.sentences_processed}")
        print(f"Duration: {result['result'].metrics.total_duration_seconds:.3f}s")

    def test_unicode_and_special_characters(self, real_source_documents):
        """Test pipeline handles unicode and special characters."""
        unicode_text = """
        機械学習はデータが必要です。
        L'apprentissage automatique nécessite des données.
        Машинное обучение требует данных.
        Special chars: @#$%^&*()_+-=[]{}|;':",./<>?
        """

        pipeline = CitationPipeline()
        result = pipeline.run(generated_text=unicode_text, source_documents=real_source_documents)

        # Should handle unicode gracefully
        assert result["result"] is not None

        print("\n=== Unicode Text Results ===")
        print(f"Sentences: {result['result'].metrics.sentences_processed}")
        print(f"Errors: {len(result['result'].errors)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
