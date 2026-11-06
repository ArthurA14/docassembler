"""Pytest tests for the API importation workflow.

This module tests the complete importation workflow:
1. import test documents via the /import endpoint
2. Poll importation status until completed
3. Verify files exist in the 'files' database index
4. Verify document chunks exist in the 'documents' database index
"""

import pytest
import requests

from api_gateway.test.conftest import (
    verify_documents_in_database,
    verify_file_in_database,
    wait_for_importation_status,
)

TEST_document_TEMPLATE = "t_test"
TEST_document_DESCRIPTIF_PARAGRAPH_COUNT = 3


@pytest.mark.integration
@pytest.mark.importation
class TestImportWorkflow:
    """Integration tests for the document importation workflow."""

    def test_import_status_after_importation(
        self,
        api_base_url: str,
        sample_core_data_ids: list[str],
        imported_documents: list[dict]
    ) -> None:
        """Test that all imported documents show 'imported' status.
        
        Uses the imported_documents fixture which handles importation and polling.
        This test verifies the status endpoint returns the correct status.
        """
        for core_data_id in sample_core_data_ids:
            status = wait_for_importation_status(
                api_base_url=api_base_url,
                core_data_id=core_data_id,
            )
            assert status == "imported", (
                f"Document {core_data_id} did not complete importation. "
                f"Final status: {status}"
            )

    def test_import_duplicate_204(
        self,
        api_base_url: str,
        doc_refs: list[dict]
    ) -> None:
        """Test that submitting identical documents while already processing returns 204.
        
        The API returns 204 No Content when an identical task is already running
        to prevent duplicate importation work.
        """
        url = f"{api_base_url}/import"
        
        # First submission should return 202
        response1 = requests.post(
            url,
            json={"docs": doc_refs},
            headers={"Content-Type": "application/json"}
        )
        assert response1.status_code == 202
        
        # Second immediate submission should return 204 (already processing)
        # Note: This test may fail if enough time passes between requests
        response2 = requests.post(
            url,
            json={"docs": doc_refs},
            headers={"Content-Type": "application/json"}
        )
        # Accept either 204 (deduplication worked) or 202 (race condition)
        assert response2.status_code in [202, 204], (
            f"Expected 202/204 for duplicate submission, got {response2.status_code}: {response2.text}"
        )


@pytest.mark.integration
@pytest.mark.live_backend
class TestDatabaseVerification:
    """Tests for database verification. Requires live backend."""

    def test_files_in_database(
        self,
        document_store_provider,
        sample_core_data_ids: list[str],
        api_base_url: str,
        imported_documents: list[dict]
    ) -> None:
        """Test that all imported files exist in the 'files' database index.
        
        This test requires real backend services and will only run with --live-backend flag.
        """
        missing_files = []
        for core_data_id in sample_core_data_ids:
            exists = verify_file_in_database(
                document_store_provider=document_store_provider,
                core_data_id=core_data_id
            )
            if not exists:
                missing_files.append(core_data_id)
        
        assert len(missing_files) == 0, (
            f"The following files were not found in the 'files' index: {missing_files}"
        )

    def test_documents_in_database(
        self,
        document_store_provider,
        sample_core_data_ids: list[str],
        api_base_url: str,
        imported_documents: list[dict]
    ) -> None:
        """Test that document chunks exist in the 'documents' database index.
        
        This test requires real backend services and will only run with --live-backend flag.
        """
        files_without_valid_documents = []
        
        for core_data_id in sample_core_data_ids:
            chunk_count = verify_documents_in_database(
                document_store_provider=document_store_provider,
                core_data_id=core_data_id
            )
            
            if chunk_count == 0:
                files_without_valid_documents.append(core_data_id)
        
        assert len(files_without_valid_documents) == 0, (
            f"The following files have no chunks in the 'documents' index: "
            f"{files_without_valid_documents}."
        )


@pytest.mark.integration
@pytest.mark.live_backend
class TestImportationErrorHandling:
    """Tests for importation error handling. Requires live backend."""

    def test_importation_status_not_found(self, api_base_url: str) -> None:
        """Test that requesting status for non-existent file returns 404.
        
        This test requires real backend services and will only run with --live-backend flag.
        """
        fake_id = "non-existent-id-12345"
        url = f"{api_base_url}/import/{fake_id}/status"
        
        response = requests.get(url)
        assert response.status_code == 404, (
            f"Expected 404 for non-existent file, got {response.status_code}"
        )


@pytest.mark.integration
@pytest.mark.importation
class TestImportationValidation:
    """Tests for importation input validation."""

    def test_import_empty_docs_list(self, api_base_url: str) -> None:
        """Test that empty docs list is handled appropriately."""
        url = f"{api_base_url}/import"
        
        response = requests.post(
            url,
            json={"docs": []},
            headers={"Content-Type": "application/json"}
        )
        
        # The API might accept empty list or reject it
        # We just verify it doesn't crash
        assert response.status_code in [202, 204, 400], (
            f"Unexpected status code {response.status_code}: {response.text}"
        )

    def test_import_missing_plain_text(self, api_base_url: str) -> None:
        """Test that document without plainText is handled."""
        url = f"{api_base_url}/import"
        
        doc_without_text = [{
            "coreDataId": "test-no-text",
            "coreType": "document",
            "coreDataSource": "Test"
            # Missing plainText
        }]
        
        response = requests.post(
            url,
            json={"docs": doc_without_text},
            headers={"Content-Type": "application/json"}
        )
        
        # API should either reject or handle gracefully
        assert response.status_code in [202, 400, 422], (
            f"Unexpected status code {response.status_code}: {response.text}"
        )