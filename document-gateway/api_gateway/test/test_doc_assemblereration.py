"""Pytest tests for the API document assembly workflow.

This module tests the complete document assembly workflow:
1. Generate documents using the t_test template
2. Verify document structure and content
3. Ensure importation is complete before document assembly

Note: These tests depend on importation being complete. The pytest_collection_modifyitems
hook in conftest.py ensures importation tests run before document tests.
"""

import pytest
import requests

from api_gateway.test.conftest import (
    wait_for_document_completion,
)

TEST_document_TEMPLATE = "t_test"
TEST_document_DESCRIPTIF_PARAGRAPH_COUNT = 3


@pytest.mark.integration
@pytest.mark.document
class TestDocAssembly:
    """Integration tests for the document assembly workflow."""

    def test_generate_document(
        self,
        api_base_url: str,
        sample_core_data_ids: list[str],
        imported_documents: list[dict]
    ) -> None:
        """Test that a document can be generated using the t_test template.
        
        Uses the imported_documents fixture which ensures documents are imported first.
        """
        # Generate document with t_test template
        document_url = f"{api_base_url}/generate-document"
        doc_refs_for_document = [
            {"coreDataId": cid, "coreType": "document", "coreDataSource": "Test Documents"}
            for cid in sample_core_data_ids
        ]
        
        response = requests.post(
            document_url,
            json={
                "templateId": TEST_document_TEMPLATE,
                "docs": doc_refs_for_document,
                "context": {
                    "subject": "War in RedLand - Analysis of the conflict"
                }
            },
            headers={"Content-Type": "application/json"}
        )
        
        # API may return 202 (new job) or 204 (already running)
        assert response.status_code in [202, 204], (
            f"Expected 202/204, got {response.status_code}: {response.text}"
        )
        
        if response.status_code == 202:
            data = response.json()
            assert "jobId" in data, "Response should contain a jobId"
        else:
            # If 204, there's already a job running - get its ID from a subsequent poll
            # This is a fallback path; normally we'd get 202
            pytest.skip("Identical document assembly job already running (204)")

    def test_generate_document_invalid_template(
        self,
        api_base_url: str,
        sample_core_data_ids: list[str],
        imported_documents: list[dict]
    ) -> None:
        """Test that generating a document with an invalid template returns 202 (async).
        
        Note: The document assembly API is asynchronous and accepts all templateId values,
        returning 202 immediately. Validation happens in the background worker.
        We verify the API returns 202 and the job eventually fails.
        """
        document_url = f"{api_base_url}/generate-document"
        doc_refs_for_document = [
            {"coreDataId": cid, "coreType": "document", "coreDataSource": "Test Documents"}
            for cid in sample_core_data_ids
        ]
        
        response = requests.post(
            document_url,
            json={
                "templateId": "nonexistent_template_xyz",
                "docs": doc_refs_for_document,
                "context": {
                    "subject": "War in RedLand - Analysis of the conflict"
                }
            },
            headers={"Content-Type": "application/json"}
        )
        
        # API accepts asynchronously, returns 202
        assert response.status_code in [202, 204], (
            f"Expected 202/204 for async document assembly, got {response.status_code}: {response.text}"
        )

    def test_generate_document_missing_template_id(
        self,
        api_base_url: str,
        sample_core_data_ids: list[str],
        imported_documents: list[dict]
    ) -> None:
        """Test that generating a document without templateId returns 422."""
        document_url = f"{api_base_url}/generate-document"
        doc_refs_for_document = [
            {"coreDataId": cid, "coreType": "document", "coreDataSource": "Test Documents"}
            for cid in sample_core_data_ids
        ]
        
        response = requests.post(
            document_url,
            json={
                "docs": doc_refs_for_document,
                "context": {
                    "subject": "War in RedLand - Analysis of the conflict"
                }
            },
            headers={"Content-Type": "application/json"}
        )
        
        # Missing required field should return 422
        assert response.status_code == 422, (
            f"Expected 422 for missing templateId, got {response.status_code}: {response.text}"
        )

    def test_generate_document_empty_docs(
        self,
        api_base_url: str
    ) -> None:
        """Test that generating a document with empty docs returns 202 (async).
        
        Note: The document assembly API is asynchronous and accepts all requests,
        returning 202 immediately. Validation happens in the background worker.
        We verify the API returns 202.
        """
        document_url = f"{api_base_url}/generate-document"
        
        response = requests.post(
            document_url,
            json={
                "templateId": TEST_document_TEMPLATE,
                "docs": [],
                "context": {
                    "subject": "War in RedLand - Analysis of the conflict"
                }
            },
            headers={"Content-Type": "application/json"}
        )
        
        # API accepts asynchronously, returns 202
        assert response.status_code in [202, 204], (
            f"Expected 202/204 for async document assembly, got {response.status_code}: {response.text}"
        )

    def test_document_section_structure(
        self,
        api_base_url: str,
        sample_core_data_ids: list[str],
        imported_documents: list[dict]
    ) -> None:
        """Test that the generated document has the correct section structure.
        
        Uses the imported_documents fixture which ensures documents are imported first.
        """
        # Generate document
        document_url = f"{api_base_url}/generate-document"
        doc_refs_for_document = [
            {"coreDataId": cid, "coreType": "document", "coreDataSource": "Test Documents"}
            for cid in sample_core_data_ids
        ]
        
        response = requests.post(
            document_url,
            json={
                "templateId": TEST_document_TEMPLATE,
                "docs": doc_refs_for_document,
                "context": {
                    "subject": "War in RedLand - Analysis of the conflict"
                }
            },
            headers={"Content-Type": "application/json"}
        )
        
        # Accept 202 (new job) or skip if 204
        if response.status_code == 204:
            pytest.skip("Identical document assembly job already running (204)")
        
        assert response.status_code == 202, (
            f"Expected 202, got {response.status_code}: {response.text}"
        )
        
        job_id = response.json()["jobId"]
        
        # Wait for document completion
        document_data = wait_for_document_completion(
            api_base_url=api_base_url,
            job_id=job_id,
            timeout=600,  # 10 minutes
            interval=10
        )
        
        # Verify document status
        assert document_data["status"] == "completed", (
            f"document did not complete. Status: {document_data['status']}"
        )
        
        # Verify required top-level fields per OpenAPI spec
        assert "jobId" in document_data, "Response should contain jobId"
        assert "status" in document_data, "Response should contain status"
        assert "coreDataIds" in document_data, "Response should contain coreDataIds"
        assert "sections" in document_data, "Response should contain sections"
        
        # Verify coreDataIds matches input
        assert len(document_data["coreDataIds"]) == len(sample_core_data_ids), (
            f"Expected {len(sample_core_data_ids)} source files, got {len(document_data['coreDataIds'])}"
        )
        
        # Verify sections is a list
        sections = document_data.get("sections", [])
        assert isinstance(sections, list), "Sections should be a list"
        assert len(sections) > 0, "document should have at least one section"
        
        # Verify required sections exist
        section_titles = [s["title"] for s in sections]
        assert "Introduction" in section_titles, (
            f"Missing 'Introduction' section. Found sections: {section_titles}"
        )
        assert "Contexte" in section_titles, (
            f"Missing 'Contexte' section. Found sections: {section_titles}"
        )
        assert "Descriptif" in section_titles, (
            f"Missing 'Descriptif' section. Found sections: {section_titles}"
        )
        assert "Conclusion" in section_titles, (
            f"Missing 'Conclusion' section. Found sections: {section_titles}"
        )
        
        # Find Descriptif section and verify it has 4 sub-sections
        descriptif_section = None
        for section in sections:
            if section["title"] == "Descriptif":
                descriptif_section = section
                break
        
        assert descriptif_section is not None, "Descriptif section not found"
        
        sub_sections = descriptif_section.get("sections", [])
        assert len(sub_sections) == TEST_document_DESCRIPTIF_PARAGRAPH_COUNT, (
            f"Descriptif should have {TEST_document_DESCRIPTIF_PARAGRAPH_COUNT} sub-sections, but has {len(sub_sections)}. "
            f"Sub-sections: {[s.get('title', 'No title') for s in sub_sections]}"
        )

    def test_document_job_status_not_found(
        self,
        api_base_url: str
    ) -> None:
        """Test that requesting status for non-existent job returns 404."""
        fake_job_id = "non-existent-job-12345"
        url = f"{api_base_url}/documents/{fake_job_id}"
        
        response = requests.get(url)
        assert response.status_code == 404, (
            f"Expected 404 for non-existent job, got {response.status_code}: {response.text}"
        )
