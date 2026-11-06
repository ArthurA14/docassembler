"""
Integration tests for citation extraction in document assembly.

These tests verify that citation extraction integrates correctly with
the document assembly workflow, using real components where possible.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from src.components.doc_assembler.citation_generators import CitationExtractor
from src.components.doc_assembler.models import Section, SectionStatus
from src.components.doc_assembler.document_writer import DocumentWriter
from src.components.doc_assembler.section_generators import SequentialSectionGenerator
from document_common.schemas.citations import Citation, CitationLabel, CitationPipelineResult, CitationStack, PipelineMetrics


class TestCitationExtractionIntegration:
    """Test citation extraction integration with document assembly components."""

    @pytest.fixture
    def document_store(self):
        """Create an in-memory document store with sample documents."""
        store = InMemoryDocumentStore()

        # Add sample source documents
        documents = [
            Document(
                id="doc-ml-001",
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                meta={
                    "source": "ml_textbook.pdf",
                    "page": 5,
                    "title": "Introduction to ML",
                },
            ),
            Document(
                id="doc-ml-002",
                content="Neural networks are computational models inspired by biological neural networks.",
                meta={
                    "source": "nn_guide.pdf",
                    "page": 12,
                    "title": "Neural Network Basics",
                },
            ),
            Document(
                id="doc-ml-003",
                content="Deep learning requires large amounts of training data to achieve good performance.",
                meta={
                    "source": "deep_learning.pdf",
                    "page": 25,
                    "title": "Deep Learning Requirements",
                },
            ),
        ]

        store.write_documents(documents)
        return store

    @pytest.fixture
    def section_store(self):
        """Create an in-memory document store for sections."""
        return InMemoryDocumentStore()

    @pytest.fixture
    def document_store(self):
        """Create an in-memory document store for documents."""
        return InMemoryDocumentStore()

    @pytest.fixture
    def document_writer(self, document_store, section_store):
        """Create a documentWriter instance."""
        return DocumentWriter(
            document_document_store=document_store,
            section_document_store=section_store,
        )

    @pytest.fixture
    def mock_citation_pipeline(self):
        """Create a mock citation pipeline with realistic behavior."""
        pipeline = MagicMock()

        def run_pipeline(generated_text, source_documents, request_id=None):
            # Simulate realistic citation extraction
            citation_stacks = []

            if "machine learning" in generated_text.lower():
                citation = Citation(
                    source_id="doc-ml-001",
                    source_doc_id="doc-ml-001",
                    quote_text="Machine learning is a subset of AI",
                    start_index=0,
                    end_index=34,
                    citation_type=CitationLabel.CLOSE_PARAPHRASE,
                    similarity_score=0.85,
                    confidence=0.88,
                )
                citation_stacks.append(
                    CitationStack(
                        span_text="machine learning",
                        span_start_index=0,
                        span_end_index=16,
                        primary_citation=citation,
                        supporting_citations=[],
                        stack_confidence=0.88,
                    )
                )

            if "neural network" in generated_text.lower():
                citation = Citation(
                    source_id="doc-ml-002",
                    source_doc_id="doc-ml-002",
                    quote_text="Neural networks are computational models",
                    start_index=0,
                    end_index=40,
                    citation_type=CitationLabel.DIRECT_QUOTE,
                    similarity_score=0.95,
                    confidence=0.92,
                )
                citation_stacks.append(
                    CitationStack(
                        span_text="neural networks",
                        span_start_index=0,
                        span_end_index=15,
                        primary_citation=citation,
                        supporting_citations=[],
                        stack_confidence=0.92,
                    )
                )

            metrics = PipelineMetrics(
                request_id=request_id or "test-request",
                total_duration_seconds=1.0,
                sentences_processed=1,
                citations_created=len(citation_stacks),
                stacks_created=len(citation_stacks),
            )

            return {
                "result": CitationPipelineResult(
                    citation_stacks=citation_stacks,
                    metrics=metrics,
                )
            }

        pipeline.run = Mock(side_effect=run_pipeline)
        return pipeline

    def test_citation_extractor_with_real_document_store(self, document_store, mock_citation_pipeline):
        """Test CitationExtractor with a real document store."""
        with patch("src.components.doc_assembler.citation_extractor.CitationPipeline") as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_citation_pipeline

            extractor = CitationExtractor(document_store=document_store)

            # Create test section
            section = Section(
                name="test_name",
                order=0,
                title="Machine Learning Overview",
                job_id="job-789",
                section_id="section-test-1",
                generated_content="Machine learning and neural networks are key technologies in AI.",
                source_ids=[["doc-ml-001", "doc-ml-002", "doc-ml-003"]],
                status=SectionStatus.COMPLETED,
            )

            # Extract citations
            result = extractor.extract_for_section(section=section, request_id="test-req-1")

            # Verify results
            assert result.citation_extraction_status == "completed"
            assert result.citations is not None
            assert len(result.citations) == 2  # Should have citations for both terms

            # Verify pipeline was called with correct documents
            mock_citation_pipeline.run.assert_called_once()
            call_args = mock_citation_pipeline.run.call_args
            assert len(call_args.kwargs["source_documents"]) == 3

    def test_end_to_end_citation_workflow(self, document_store, document_writer, mock_citation_pipeline):
        """Test complete workflow: extract citations, store section, retrieve citations."""
        with patch("src.components.doc_assembler.citation_extractor.CitationPipeline") as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_citation_pipeline

            # Create extractor
            extractor = CitationExtractor(document_store=document_store)

            # Create and process section
            section = Section(
                name="ai_technologies",
                order=1,
                title="AI Technologies",
                job_id="job-e2e",
                section_id="section-e2e-1",
                generated_content="Neural networks are a fundamental technology in machine learning.",
                source_ids=[["doc-ml-001", "doc-ml-002"]],
                status=SectionStatus.COMPLETED,
            )

            # Extract citations
            section_with_citations = extractor.extract_for_section(section=section)

            # Store section
            document_writer.update_document_section(section=section_with_citations)

            # Retrieve section and verify citations persisted
            retrieved_section = document_writer.get_document_section(
                job_id="job-e2e",
                section_id="section-e2e-1",
            )

            assert retrieved_section is not None
            assert retrieved_section.citation_extraction_status == "completed"
            assert retrieved_section.citations is not None
            assert len(retrieved_section.citations) > 0

            # Retrieve citations using citation-specific method
            citations = document_writer.get_section_citations(section_id="section-e2e-1")

            assert citations is not None
            assert len(citations) == len(section_with_citations.citations)

    def test_sequential_section_generator_integration(self, document_store, document_writer, mock_citation_pipeline):
        """Test citation extraction integration with SequentialSectionGenerator."""
        with patch("src.components.doc_assembler.citation_extractor.CitationPipeline") as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_citation_pipeline

            # Create section for generator
            section = Section(
                name="test_section",
                order=1,
                title="Test Section",
                job_id="job-seq-gen",
                section_id="section-seq-1",
                generated_content=None,  # Will be set during assembly
                source_ids=[],
                status=SectionStatus.PENDING,
            )

            # Mock the content generator
            mock_generator = MagicMock()
            mock_generator.run = MagicMock(
                return_value={"replies": ["Machine learning is transforming industries through neural networks."]}
            )

            # Create a document object (required when using document_writer)
            from src.components.doc_assembler.models import Document as AssembledDocument

            document = AssembledDocument(job_id="job-seq-gen", file_ids=["file1.pdf"])

            # Create SequentialSectionGenerator with citation extraction enabled
            generator = SequentialSectionGenerator(
                prompt_name="test_paragraph_assembly",
                section=section,
                document_store=document_store,
                method_parameters={
                    "doc_type": "chunk",
                    "paragraphs_count": 3,
                    "auto_title": False,
                    "top_k": 5,
                },
                document=document,
                document_writer=document_writer,
                extract_citations=True,  # Enable citation extraction
            )

            # Manually set source_ids as they would be set during actual assembly
            generator.section.source_ids = [["doc-ml-001", "doc-ml-002"]]
            generator.section.generated_content = "Machine learning is transforming industries through neural networks."
            generator.section.status = SectionStatus.COMPLETED

            # Manually trigger citation extraction (simulating what happens in run())
            if generator.extract_citations and generator.citation_extractor:
                try:
                    generator.section = generator.citation_extractor.extract_for_section(
                        section=generator.section,
                        request_id=f"{generator.section.job_id}-{generator.section.section_id}",
                    )
                except Exception:
                    generator.section.citation_extraction_status = "failed"

            # Verify citations were extracted
            assert generator.section.citation_extraction_status == "completed"
            assert generator.section.citations is not None
            assert len(generator.section.citations) > 0

    def test_multiple_sections_citation_retrieval(self, document_store, document_writer, mock_citation_pipeline):
        """Test retrieving citations for multiple sections in a document."""
        with patch("src.components.doc_assembler.citation_extractor.CitationPipeline") as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_citation_pipeline

            extractor = CitationExtractor(document_store=document_store)

            # Create and process multiple sections
            sections = [
                Section(
                    name="section_1",
                    order=1,
                    title="Section 1",
                    job_id="job-multi",
                    section_id="section-multi-1",
                    generated_content="Machine learning algorithms are powerful.",
                    source_ids=[["doc-ml-001"]],
                    status=SectionStatus.COMPLETED,
                ),
                Section(
                    name="section_2",
                    order=2,
                    title="Section 2",
                    job_id="job-multi",
                    section_id="section-multi-2",
                    generated_content="Neural networks process information in layers.",
                    source_ids=[["doc-ml-002"]],
                    status=SectionStatus.COMPLETED,
                ),
                Section(
                    name="section_3",
                    order=3,
                    title="Section 3",
                    job_id="job-multi",
                    section_id="section-multi-3",
                    generated_content="This section has no sources.",
                    source_ids=[],
                    status=SectionStatus.COMPLETED,
                ),
            ]

            # Extract citations for each section
            processed_sections = []
            for section in sections:
                processed = extractor.extract_for_section(section=section)
                document_writer.update_document_section(section=processed)
                processed_sections.append(processed)

            # Retrieve all citations for the document
            all_citations = document_writer.get_document_citations(job_id="job-multi")

            # Verify results
            assert len(all_citations) == 2  # Only sections with citations
            assert "section-multi-1" in all_citations
            assert "section-multi-2" in all_citations
            assert "section-multi-3" not in all_citations  # Skipped

            # Check extraction status
            status_map = document_writer.get_citation_extraction_status(job_id="job-multi")
            assert status_map["section-multi-1"] == "completed"
            assert status_map["section-multi-2"] == "completed"
            assert status_map["section-multi-3"] == "skipped"

    def test_citation_extraction_error_handling(self, document_store, document_writer, mock_citation_pipeline):
        """Test that citation extraction errors don't break the workflow."""
        with patch("src.components.doc_assembler.citation_extractor.CitationPipeline") as mock_pipeline_class:
            # Configure pipeline to fail
            mock_pipeline_class.return_value.run = MagicMock(side_effect=Exception("Pipeline failure"))

            extractor = CitationExtractor(document_store=document_store)

            section = Section(
                name="error_test",
                order=1,
                title="Error Test",
                job_id="job-error",
                section_id="section-error-1",
                generated_content="Test content that will fail.",
                source_ids=[["doc-ml-001"]],
                status=SectionStatus.COMPLETED,
            )

            # Extract citations (should handle error gracefully)
            result = extractor.extract_for_section(section=section)

            # Verify error was handled
            assert result.citation_extraction_status == "failed"
            assert result.citations is None

            # Store section even with failed citation extraction
            document_writer.update_document_section(section=result)

            # Verify section was stored
            retrieved = document_writer.get_document_section(
                job_id="job-error",
                section_id="section-error-1",
            )
            assert retrieved is not None
            assert retrieved.citation_extraction_status == "failed"


class TestConfigurationIntegration:
    """Test that configuration correctly controls citation extraction."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.citations = {
            "enable_in_documents": True,
            "fail_on_citation_error": False,
        }
        config.embedder = {
            "type": "api",
            "model": "tei",
            "url": "http://localhost:8080",
            "dimension": 768,
        }
        return config

    def test_citation_extraction_disabled_by_config(self, mock_config):
        """Test that citation extraction can be disabled via configuration."""
        mock_config.citations["enable_in_documents"] = False

        with patch("src.components.doc_assembler.section_generators.config.get_current_config") as mock_get_config:
            mock_get_config.return_value = mock_config

            mock_generator = MagicMock()
            mock_document_store = MagicMock()
            mock_document_writer = MagicMock()

            section = Section(
                name="test_config",
                order=1,
                title="Test",
                job_id="job-config",
                section_id="section-config-1",
                generated_content=None,
                source_ids=[],
                status=SectionStatus.PENDING,
            )

            # Create a document object (required when using document_writer)
            from src.components.doc_assembler.models import Document as AssembledDocument

            document = AssembledDocument(job_id="job-config", file_ids=["file1.pdf"])

            generator = SequentialSectionGenerator(
                prompt_name="test_prompt",
                section=section,
                document_store=mock_document_store,
                method_parameters={
                    "doc_type": "chunk",
                    "paragraphs_count": 3,
                    "auto_title": False,
                    "top_k": 5,
                },
                document=document,
                document_writer=mock_document_writer,
                extract_citations=None,  # Use config value
            )

            # Verify citation extraction is disabled
            assert generator.extract_citations is False
            assert generator.citation_extractor is None

    def test_citation_extraction_enabled_by_config(self, mock_config):
        """Test that citation extraction can be enabled via configuration."""
        mock_config.citations["enable_in_documents"] = True

        with patch("src.components.doc_assembler.section_generators.config.get_current_config") as mock_get_config:
            mock_get_config.return_value = mock_config

            mock_generator = MagicMock()
            mock_document_store = MagicMock()
            mock_document_writer = MagicMock()

            section = Section(
                name="test_config_2",
                order=1,
                title="Test",
                job_id="job-config",
                section_id="section-config-2",
                generated_content=None,
                source_ids=[],
                status=SectionStatus.PENDING,
            )

            # Create a document object (required when using document_writer)
            from src.components.doc_assembler.models import Document as AssembledDocument

            document = AssembledDocument(job_id="job-config", file_ids=["file1.pdf"])

            with patch("src.components.doc_assembler.section_generators.CitationExtractor"):
                generator = SequentialSectionGenerator(
                    prompt_name="test_prompt",
                    section=section,
                    document_store=mock_document_store,
                    method_parameters={
                        "doc_type": "chunk",
                        "paragraphs_count": 3,
                        "auto_title": False,
                        "top_k": 5,
                    },
                    document=document,
                    document_writer=mock_document_writer,
                    extract_citations=None,  # Use config value
                )

                # Verify citation extraction is enabled
                assert generator.extract_citations is True


class TestPerformance:
    """Test performance-related aspects of citation extraction."""

    def test_large_source_document_set(self):
        """Test citation extraction with a large number of source documents."""
        # Create document store with many documents
        store = InMemoryDocumentStore()
        documents = [
            Document(
                id=f"doc-{i}",
                content=f"Content for document {i} about various topics.",
                meta={"source": f"source_{i}.pdf", "page": i},
            )
            for i in range(50)
        ]
        store.write_documents(documents)

        with patch("src.components.doc_assembler.citation_extractor.CitationPipeline") as mock_pipeline_class:
            mock_pipeline = MagicMock()
            test_metrics = PipelineMetrics(
                request_id="test-perf",
                total_duration_seconds=1.0,
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
            mock_pipeline_class.return_value = mock_pipeline

            extractor = CitationExtractor(document_store=store)

            # Create section with many source IDs
            section = Section(
                name="large_source_set",
                order=1,
                title="Large Source Set",
                job_id="job-perf",
                section_id="section-perf-1",
                generated_content="Content referencing many sources.",
                source_ids=[[f"doc-{i}"] for i in range(50)],
                status=SectionStatus.COMPLETED,
            )

            # Extract citations
            result = extractor.extract_for_section(section=section)

            # Verify it completed successfully
            assert result.citation_extraction_status == "completed"

            # Verify all documents were retrieved
            call_args = mock_pipeline.run.call_args
            assert len(call_args.kwargs["source_documents"]) == 50
