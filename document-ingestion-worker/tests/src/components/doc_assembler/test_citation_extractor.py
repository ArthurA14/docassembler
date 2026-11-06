from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from src.components.doc_assembler.citation_generators import CitationExtractor
from src.components.doc_assembler.models import Section, SectionStatus
from document_common.schemas.citations import (
    Citation,
    CitationLabel,
    CitationPipelineResult,
    CitationStack,
    PipelineMetrics,
    SourceMetadata,
)


class TestCitationExtractor:
    """Unit tests for CitationExtractor."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store."""
        store = MagicMock()
        return store

    @pytest.fixture
    def mock_citation_pipeline(self):
        """Create a mock citation pipeline."""
        pipeline = MagicMock()
        return pipeline

    @pytest.fixture
    def extractor(self, mock_document_store, mock_citation_pipeline):
        """Create a CitationExtractor with mocked dependencies."""
        with patch("src.components.doc_assembler.citation_generators.CitationPipeline") as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_citation_pipeline
            extractor = CitationExtractor(document_store=mock_document_store)
            extractor.citation_pipeline = mock_citation_pipeline
            return extractor

    @pytest.fixture
    def sample_section(self):
        """Create a sample section with source IDs."""
        return Section(
            job_id="test-job-123",
            section_id="section-1",
            title="Test Section",
            name="test_section",
            order=1,
            generated_content="This is generated content about machine learning.",
            source_ids=[["doc-1", "doc-2"], ["doc-3"]],  # Hierarchical structure
            status=SectionStatus.COMPLETED,
        )

    @pytest.fixture
    def sample_documents(self):
        """Create sample source documents."""
        return [
            Document(
                id="doc-1",
                content="Machine learning is a field of AI.",
                meta={"source": "source1.pdf", "page": 1},
            ),
            Document(
                id="doc-2",
                content="Neural networks are used in ML.",
                meta={"source": "source2.pdf", "page": 5},
            ),
            Document(
                id="doc-3",
                content="Deep learning requires large datasets.",
                meta={"source": "source3.pdf", "page": 10},
            ),
        ]

    @pytest.fixture
    def sample_citation_result(self):
        """Create a sample citation result."""
        citation = Citation(
            source_id="doc-1",
            source_doc_id="doc-1",
            quote_text="Machine learning is a field of AI",
            start_index=0,
            end_index=33,
            citation_type=CitationLabel.CLOSE_PARAPHRASE,
            similarity_score=0.85,
            confidence=0.88,
        )
        citation_stack = CitationStack(
            span_text="machine learning",
            span_start_index=0,
            span_end_index=16,
            primary_citation=citation,
            supporting_citations=[],
            stack_confidence=0.88,
        )
        metrics = PipelineMetrics(
            request_id="test-request",
            total_duration_seconds=1.5,
            sentences_processed=1,
            citations_created=1,
            stacks_created=1,
        )
        return CitationPipelineResult(
            citation_stacks=[citation_stack],
            metrics=metrics,
        )

    def test_initialization(self, mock_document_store):
        """Test that CitationExtractor initializes correctly."""
        with patch("src.components.doc_assembler.citation_extractor.CitationPipeline") as mock_pipeline_class:
            extractor = CitationExtractor(document_store=mock_document_store)

            assert extractor.document_store == mock_document_store
            assert extractor.citation_pipeline is not None
            mock_pipeline_class.assert_called_once()

    def test_extract_for_section_success(self, extractor, sample_section, sample_documents, sample_citation_result):
        """Test successful citation extraction for a section."""
        # Mock document retrieval
        extractor._retrieve_source_documents = MagicMock(return_value=sample_documents)

        # Mock citation pipeline run - must return dict with 'result' key
        extractor.citation_pipeline.run = MagicMock(return_value={"result": sample_citation_result})

        # Extract citations
        result = extractor.extract_for_section(section=sample_section, request_id="test-request-1")

        # Verify section was updated
        assert result.citations == sample_citation_result.citation_stacks
        assert result.citation_extraction_status == "completed"

        # Verify pipeline was called with correct arguments
        extractor.citation_pipeline.run.assert_called_once()
        call_args = extractor.citation_pipeline.run.call_args
        assert call_args.kwargs["generated_text"] == sample_section.generated_content
        assert call_args.kwargs["request_id"] == "test-request-1"

    def test_extract_for_section_no_source_ids(self, extractor):
        """Test extraction when section has no source IDs."""
        section = Section(
            job_id="test-job-123",
            section_id="section-1",
            title="Test Section",
            name="test_section",
            order=1,
            generated_content="Content without sources.",
            source_ids=[],  # Empty source IDs
            status=SectionStatus.COMPLETED,
        )

        result = extractor.extract_for_section(section=section)

        assert result.citation_extraction_status == "skipped"
        assert result.citations is None

    def test_extract_for_section_no_content(self, extractor):
        """Test extraction when section has no generated content."""
        section = Section(
            job_id="test-job-123",
            section_id="section-1",
            title="Test Section",
            name="test_section",
            order=1,
            generated_content=None,
            source_ids=[["doc-1"]],
            status=SectionStatus.COMPLETED,
        )

        result = extractor.extract_for_section(section=section)

        assert result.citation_extraction_status == "skipped"
        assert result.citations is None

    def test_extract_for_section_pipeline_error(self, extractor, sample_section, sample_documents):
        """Test handling of pipeline errors during extraction."""
        # Mock document retrieval
        extractor._retrieve_source_documents = MagicMock(return_value=sample_documents)

        # Mock pipeline to raise an error
        extractor.citation_pipeline.run = MagicMock(side_effect=Exception("Pipeline error"))

        result = extractor.extract_for_section(section=sample_section)

        assert result.citation_extraction_status == "failed"
        assert result.citations is None

    def test_extract_for_section_no_documents_found(self, extractor, sample_section):
        """Test extraction when no source documents are found."""
        # Mock document retrieval to return empty list
        extractor._retrieve_source_documents = MagicMock(return_value=[])

        result = extractor.extract_for_section(section=sample_section)

        assert result.citation_extraction_status == "skipped"
        assert result.citations is None

    def test_retrieve_source_documents(self, extractor, sample_documents, mock_document_store):
        """Test retrieval of source documents from document store."""
        source_ids = ["doc-1", "doc-2", "doc-3"]

        # Mock document store filter
        mock_document_store.filter_documents = MagicMock(return_value=sample_documents)

        result = extractor._retrieve_source_documents(source_ids, request_id="test-request")

        assert len(result) == 3
        # Verify all documents were converted to SourceMetadata
        for source_meta in result:
            assert isinstance(source_meta, SourceMetadata)

        # Verify filter was called with correct OR structure
        mock_document_store.filter_documents.assert_called_once()
        call_kwargs = mock_document_store.filter_documents.call_args.kwargs
        filters = call_kwargs["filters"]
        assert filters["operator"] == "OR"
        assert "conditions" in filters
        assert len(filters["conditions"]) == len(source_ids)

    def test_retrieve_source_documents_error(self, extractor, mock_document_store):
        """Test handling of document store errors during retrieval."""
        source_ids = ["doc-1", "doc-2"]

        # Mock document store to raise an error
        mock_document_store.filter_documents = MagicMock(side_effect=Exception("Document store error"))

        result = extractor._retrieve_source_documents(source_ids, request_id="test-request")

        assert result == []

    def test_source_id_flattening(self, extractor, sample_section, sample_documents, sample_citation_result):
        """Test that hierarchical source IDs are correctly flattened and deduplicated."""
        # Add duplicate IDs to test deduplication
        sample_section.source_ids = [["doc-1", "doc-2"], ["doc-2", "doc-3"], ["doc-1"]]

        extractor._retrieve_source_documents = MagicMock(return_value=sample_documents)
        extractor.citation_pipeline.run = MagicMock(return_value={"result": sample_citation_result})

        extractor.extract_for_section(section=sample_section)

        # Verify that retrieve was called with deduplicated IDs
        call_args = extractor._retrieve_source_documents.call_args[0][0]
        assert len(call_args) == 3  # Should be deduplicated
        assert set(call_args) == {"doc-1", "doc-2", "doc-3"}

    # Removed test_convert_documents_to_source_metadata as this method doesn't exist in the implementation

    def test_empty_citation_result(self, extractor, sample_section, sample_documents):
        """Test handling when pipeline returns empty citation result."""
        extractor._retrieve_source_documents = MagicMock(return_value=sample_documents)

        # Mock pipeline to return result with no citation stacks
        empty_metrics = PipelineMetrics(
            request_id="test-request",
            total_duration_seconds=1.0,
            sentences_processed=1,
            citations_created=0,
            stacks_created=0,
        )
        empty_result = CitationPipelineResult(
            citation_stacks=[],
            metrics=empty_metrics,
        )
        extractor.citation_pipeline.run = MagicMock(return_value={"result": empty_result})

        result = extractor.extract_for_section(section=sample_section)

        assert result.citation_extraction_status == "completed"
        assert result.citations == []

    def test_request_id_propagation(self, extractor, sample_section, sample_documents, sample_citation_result):
        """Test that request_id is properly propagated through the extraction process."""
        extractor._retrieve_source_documents = MagicMock(return_value=sample_documents)
        extractor.citation_pipeline.run = MagicMock(return_value={"result": sample_citation_result})

        custom_request_id = "custom-request-123"
        extractor.extract_for_section(section=sample_section, request_id=custom_request_id)

        # Verify pipeline received the request_id
        call_args = extractor.citation_pipeline.run.call_args
        assert call_args.kwargs["request_id"] == custom_request_id


class TestIntegration:
    """Integration tests for citation extraction in document assembly."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store with realistic behavior."""
        store = MagicMock()

        # Create sample documents in the store
        documents = {
            "doc-1": Document(
                id="doc-1",
                content="Machine learning is a subset of artificial intelligence.",
                meta={"source": "ml_textbook.pdf", "page": 42},
            ),
            "doc-2": Document(
                id="doc-2",
                content="Neural networks consist of layers of interconnected nodes.",
                meta={"source": "nn_guide.pdf", "page": 15},
            ),
        }

        def filter_docs(filters):
            if filters.get("operator") == "OR" and "conditions" in filters:
                requested_ids = [cond["value"] for cond in filters["conditions"] if cond["field"] == "id"]
                return [documents[doc_id] for doc_id in requested_ids if doc_id in documents]
            return []

        store.filter_documents = MagicMock(side_effect=filter_docs)
        return store

    def test_full_extraction_workflow(self, mock_document_store):
        """Test complete citation extraction workflow with mocked pipeline."""
        with patch("src.components.doc_assembler.citation_extractor.CitationPipeline") as mock_pipeline_class:
            # Setup mock pipeline
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline

            # Setup mock citation result
            citation = Citation(
                source_id="doc-1",
                source_doc_id="doc-1",
                quote_text="Machine learning is a subset of artificial intelligence.",
                start_index=0,
                end_index=56,
                citation_type=CitationLabel.LOOSE_PARAPHRASE,
                similarity_score=0.75,
                confidence=0.82,
            )
            citation_stack = CitationStack(
                span_text="ML uses neural networks",
                span_start_index=0,
                span_end_index=23,
                primary_citation=citation,
                supporting_citations=[],
                stack_confidence=0.82,
            )
            metrics = PipelineMetrics(
                request_id="integration-test-1",
                total_duration_seconds=1.2,
                sentences_processed=1,
                citations_created=1,
                stacks_created=1,
            )
            citation_result = CitationPipelineResult(
                citation_stacks=[citation_stack],
                metrics=metrics,
            )
            mock_pipeline.run = MagicMock(return_value={"result": citation_result})

            # Create extractor
            extractor = CitationExtractor(document_store=mock_document_store)

            # Create test section
            section = Section(
                job_id="job-456",
                section_id="section-2",
                title="Machine Learning Overview",
                name="ml_overview",
                order=1,
                generated_content="ML uses neural networks for pattern recognition.",
                source_ids=[["doc-1", "doc-2"]],
                status=SectionStatus.COMPLETED,
            )

            # Extract citations
            result = extractor.extract_for_section(section=section, request_id="integration-test-1")

            # Verify results
            assert result.citation_extraction_status == "completed"
            assert len(result.citations) == 1
            assert result.citations[0].span_text == "ML uses neural networks"
            assert result.citations[0].primary_citation is not None

            # Verify pipeline was called
            mock_pipeline.run.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def mock_document_store(self):
        return MagicMock()

    def test_section_with_none_values(self, mock_document_store):
        """Test handling of sections with None values in various fields."""
        with patch("src.components.doc_assembler.citation_extractor.CitationPipeline"):
            extractor = CitationExtractor(document_store=mock_document_store)

            section = Section(
                job_id="test-job",
                section_id="section-1",
                title="Test",
                name="test_section",
                order=1,
                generated_content=None,
                source_ids=[],  # Empty list instead of None
                status=SectionStatus.COMPLETED,
            )

            result = extractor.extract_for_section(section=section)
            assert result.citation_extraction_status == "skipped"

    def test_section_with_empty_strings(self, mock_document_store):
        """Test handling of sections with empty string content."""
        with patch("src.components.doc_assembler.citation_extractor.CitationPipeline"):
            extractor = CitationExtractor(document_store=mock_document_store)

            section = Section(
                job_id="test-job",
                section_id="section-1",
                title="Test",
                name="test_section",
                order=1,
                generated_content="",
                source_ids=[["doc-1"]],
                status=SectionStatus.COMPLETED,
            )

            result = extractor.extract_for_section(section=section)
            assert result.citation_extraction_status == "skipped"

    def test_section_with_whitespace_only_content(self, mock_document_store):
        """Test handling of sections with whitespace-only content."""
        with patch("src.components.doc_assembler.citation_extractor.CitationPipeline"):
            extractor = CitationExtractor(document_store=mock_document_store)

            section = Section(
                job_id="test-job",
                section_id="section-1",
                title="Test",
                name="test_section",
                order=1,
                generated_content="   \n\t  ",
                source_ids=[["doc-1"]],
                status=SectionStatus.COMPLETED,
            )

            result = extractor.extract_for_section(section=section)
            assert result.citation_extraction_status == "skipped"

    def test_large_number_of_source_ids(self, mock_document_store):
        """Test handling of sections with many source documents."""
        with patch("src.components.doc_assembler.citation_extractor.CitationPipeline") as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline

            # Create many documents
            many_docs = [Document(id=f"doc-{i}", content=f"Content {i}", meta={}) for i in range(100)]
            mock_document_store.filter_documents = MagicMock(return_value=many_docs)

            test_metrics = PipelineMetrics(
                request_id="test-request",
                total_duration_seconds=0.5,
                sentences_processed=0,
                citations_created=0,
                stacks_created=0,
            )
            mock_pipeline.run = MagicMock(
                return_value={
                    "result": CitationPipelineResult(
                        citation_stacks=[],
                        metrics=test_metrics,
                    )
                }
            )

            extractor = CitationExtractor(document_store=mock_document_store)

            # Create section with many source IDs
            source_ids = [[f"doc-{i}"] for i in range(100)]
            section = Section(
                job_id="test-job",
                section_id="section-1",
                title="Test",
                name="test_section",
                order=1,
                generated_content="Test content",
                source_ids=source_ids,
                status=SectionStatus.COMPLETED,
            )

            result = extractor.extract_for_section(section=section)

            # Should handle successfully
            assert result.citation_extraction_status == "completed"
            mock_pipeline.run.assert_called_once()
