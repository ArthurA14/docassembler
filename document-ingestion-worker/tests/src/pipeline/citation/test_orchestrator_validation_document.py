"""
Test suite for Validationdocument handling in the citation pipeline orchestrator.

Tests the proper handling of Pydantic Validationdocument objects and their
conversion to dict format for CitationPipelineResult serialization.
"""

import pytest

from src.components.citations.validator import ValidationDocument
from src.pipeline.citation.orchestrator import PipelineOrchestrator
from document_common.schemas.citations import (
    CitationPipelineResult,
    PipelineMetrics,
    SourceMetadata,
    SourceReferences,
)

# from src.utils.citation_executor import ExecutionMetrics


class TestValidationDocumentHandling:
    """Test Validationdocument type handling and serialization."""

    @pytest.fixture
    def orchestrator(self):
        """Create a PipelineOrchestrator instance."""
        return PipelineOrchestrator()

    @pytest.fixture
    def execution_metrics(self):
        """Create ExecutionMetrics instance."""
        return ExecutionMetrics()

    @pytest.fixture
    def validation_document(self):
        """Create a sample Validationdocument Pydantic object."""
        return ValidationDocument(
            errors=2,
            warnings=1,
            validated_count=10,
            invalid_count=2,
            validation_issues=[],
        )

    @pytest.fixture
    def source_references(self):
        """Create sample SourceReferences."""
        return SourceReferences(
            references={
                "source-1": SourceMetadata(
                    id="source-1",
                    file_id="file-1",
                    content="Sample content",
                    type="original",
                )
            }
        )

    def test_validation_document_is_pydantic_object(self, validation_document):
        """Verify Validationdocument is a Pydantic BaseModel."""
        assert hasattr(validation_document, "model_dump")
        assert hasattr(validation_document, "errors")
        assert hasattr(validation_document, "warnings")
        assert isinstance(validation_document.errors, int)
        assert isinstance(validation_document.warnings, int)

    def test_validation_document_model_dump_conversion(self, validation_document):
        """Test converting Validationdocument to dict via model_dump()."""
        document_dict = validation_document.model_dump()
        assert isinstance(document_dict, dict)
        assert document_dict["errors"] == 2
        assert document_dict["warnings"] == 1
        assert "validated_count" in document_dict
        assert "invalid_count" in document_dict

    def test_build_result_with_valid_validation_document(
        self,
        orchestrator,
        execution_metrics,
        validation_document,
        source_references,
    ):
        """Test _build_result with valid Validationdocument."""
        result_dict = orchestrator._build_result(
            request_id="test-123",
            execution_metrics=execution_metrics,
            valid_stacks=[],
            invalid_stacks=[],
            validation_document=validation_document.model_dump(),
            errors=[],
            source_references=source_references,
        )

        assert "result" in result_dict
        result = result_dict["result"]
        assert isinstance(result, CitationPipelineResult)
        assert result.validation_document is not None
        assert isinstance(result.validation_document, dict)
        assert result.validation_document["errors"] == 2
        assert result.validation_document["warnings"] == 1

    def test_build_result_with_none_validation_document(
        self,
        orchestrator,
        execution_metrics,
        source_references,
    ):
        """Test _build_result with None validation_document."""
        result_dict = orchestrator._build_result(
            request_id="test-123",
            execution_metrics=execution_metrics,
            valid_stacks=[],
            invalid_stacks=[],
            validation_document=None,
            errors=[],
            source_references=source_references,
        )

        assert "result" in result_dict
        result = result_dict["result"]
        assert result.validation_document is None

    def test_metrics_calculation_with_validation_document(
        self,
        orchestrator,
        execution_metrics,
        validation_document,
        source_references,
    ):
        """Test metrics are correctly calculated from Validationdocument."""
        result_dict = orchestrator._build_result(
            request_id="test-123",
            execution_metrics=execution_metrics,
            valid_stacks=[],
            invalid_stacks=[],
            validation_document=validation_document.model_dump(),
            errors=[],
            source_references=source_references,
        )

        result = result_dict["result"]
        metrics = result.metrics

        # Verify metrics were created from execution_metrics
        assert metrics.request_id == "test-123"
        assert isinstance(metrics.total_duration_seconds, float)
        assert isinstance(metrics.component_durations, dict)

    def test_error_metrics_with_validation_document(
        self,
        orchestrator,
        execution_metrics,
        validation_document,
        source_references,
    ):
        """Test error and warning counts from Validationdocument."""
        result_dict = orchestrator._build_result(
            request_id="test-123",
            execution_metrics=execution_metrics,
            valid_stacks=[],
            invalid_stacks=[],
            validation_document=validation_document.model_dump(),
            errors=[],
            source_references=source_references,
        )

        result = result_dict["result"]
        metrics = result.metrics

        # Verify validation errors and warnings are captured
        assert metrics.validation_errors == 2
        assert metrics.validation_warnings == 1

    def test_citation_pipeline_result_serialization(
        self,
        orchestrator,
        execution_metrics,
        validation_document,
        source_references,
    ):
        """Test CitationPipelineResult with validation_document can be serialized."""
        result_dict = orchestrator._build_result(
            request_id="test-123",
            execution_metrics=execution_metrics,
            valid_stacks=[],
            invalid_stacks=[],
            validation_document=validation_document.model_dump(),
            errors=[],
            source_references=source_references,
        )

        result = result_dict["result"]

        # Test Pydantic serialization
        serialized = result.model_dump()
        assert isinstance(serialized, dict)
        assert "validation_document" in serialized
        assert serialized["validation_document"] is not None
        assert isinstance(serialized["validation_document"], dict)

    def test_validation_document_dict_field_access(
        self,
        orchestrator,
        execution_metrics,
        validation_document,
        source_references,
    ):
        """Test that validation_document dict allows field access."""
        result_dict = orchestrator._build_result(
            request_id="test-123",
            execution_metrics=execution_metrics,
            valid_stacks=[],
            invalid_stacks=[],
            validation_document=validation_document.model_dump(),
            errors=[],
            source_references=source_references,
        )

        result = result_dict["result"]
        vr = result.validation_document

        # Verify dict-like access works
        assert vr["errors"] == 2
        assert vr["warnings"] == 1
        assert vr.get("errors") == 2
        assert vr.get("warnings") == 1
        assert vr.get("nonexistent", "default") == "default"

    def test_empty_validation_document(
        self,
        orchestrator,
        execution_metrics,
        source_references,
    ):
        """Test with empty/minimal Validationdocument."""
        empty_document = ValidationDocument(errors=0, warnings=0, validated_count=0, invalid_count=0, validation_issues=[])

        result_dict = orchestrator._build_result(
            request_id="test-123",
            execution_metrics=execution_metrics,
            valid_stacks=[],
            invalid_stacks=[],
            validation_document=empty_document.model_dump(),
            errors=[],
            source_references=source_references,
        )

        result = result_dict["result"]
        assert result.validation_document["errors"] == 0
        assert result.validation_document["warnings"] == 0

    def test_error_result_with_none_validation_document(
        self,
        orchestrator,
        execution_metrics,
    ):
        """Test _build_error_result handles None validation_document correctly."""
        result_dict = orchestrator._build_error_result(
            request_id="test-456",
            execution_metrics=execution_metrics,
        )

        assert "result" in result_dict
        result = result_dict["result"]
        assert result.validation_document is None
        assert result.metrics.validation_errors == 0
        assert result.metrics.validation_warnings == 0

    def test_multiple_validation_documents_in_sequence(
        self,
        orchestrator,
        execution_metrics,
        source_references,
    ):
        """Test handling multiple Validationdocument objects sequentially."""
        documents = [
            ValidationDocument(errors=1, warnings=0, validated_count=5, invalid_count=1, validation_issues=[]),
            ValidationDocument(errors=0, warnings=2, validated_count=10, invalid_count=0, validation_issues=[]),
            ValidationDocument(errors=3, warnings=1, validated_count=7, invalid_count=3, validation_issues=[]),
        ]

        results = []
        for document in documents:
            result_dict = orchestrator._build_result(
                request_id=f"test-{len(results)}",
                execution_metrics=execution_metrics,
                valid_stacks=[],
                invalid_stacks=[],
                validation_document=document.model_dump(),
                errors=[],
                source_references=source_references,
            )
            results.append(result_dict["result"])

        # Verify each result has correct validation_document data
        assert results[0].validation_document["errors"] == 1
        assert results[1].validation_document["warnings"] == 2
        assert results[2].validation_document["errors"] == 3

    def test_validation_document_preserves_all_fields(
        self,
        orchestrator,
        execution_metrics,
        source_references,
    ):
        """Test all Validationdocument fields are preserved through conversion."""
        document = ValidationDocument(
            errors=5,
            warnings=3,
            validated_count=100,
            invalid_count=5,
            validation_issues=[],
        )

        result_dict = orchestrator._build_result(
            request_id="test-789",
            execution_metrics=execution_metrics,
            valid_stacks=[],
            invalid_stacks=[],
            validation_document=document.model_dump(),
            errors=[],
            source_references=source_references,
        )

        result = result_dict["result"]
        vr = result.validation_document

        # All fields should be preserved
        assert vr["errors"] == 5
        assert vr["warnings"] == 3
        assert vr["validated_count"] == 100
        assert vr["invalid_count"] == 5
        assert "validation_issues" in vr


class TestValidationDocumentTypeConsistency:
    """Test type consistency across the validation pipeline."""

    def test_validation_document_field_types(self):
        """Verify Validationdocument field types are correct."""
        document = ValidationDocument(
            errors=2,
            warnings=1,
            validated_count=10,
            invalid_count=2,
            validation_issues=[],
        )

        # Fields should be integers, not lists
        assert isinstance(document.errors, int)
        assert isinstance(document.warnings, int)

    def test_citation_pipeline_result_validation_document_type_hint(self):
        """Verify CitationPipelineResult expects Optional[dict] for validation_document."""
        # Create a result with validation_document as dict
        result = CitationPipelineResult(
            citation_stacks=[],
            invalid_citations=[],
            source_references=SourceReferences(references={}),
            metrics=PipelineMetrics(
                request_id="test",
                total_duration_seconds=0.1,
                component_durations={},
            ),
            validation_document={"errors": 1, "warnings": 0},
            errors=[],
            warnings=[],
        )

        assert result.validation_document is not None
        assert isinstance(result.validation_document, dict)

    def test_validation_document_json_serializable(self):
        """Test Validationdocument converts to JSON-serializable dict."""
        import json

        document = ValidationDocument(
            errors=2,
            warnings=1,
            validated_count=10,
            invalid_count=2,
            validation_issues=[],
        )

        document_dict = document.model_dump()
        json_str = json.dumps(document_dict)
        loaded = json.loads(json_str)

        assert loaded["errors"] == 2
        assert loaded["warnings"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
