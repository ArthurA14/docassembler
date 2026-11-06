"""Pytest tests for system endpoints.

This module tests the system endpoints:
- GET /files - List available files in database
- POST /flush - Flush system data
- GET /health - Health check

Usage:
    pytest api/test/test_system_endpoints.py              # Run tests
    pytest api/test/test_system_endpoints.py --live-backend  # Run with backend verification
"""

import pytest
import requests

from api_gateway.test.conftest import logger


@pytest.mark.integration
class TestHealthEndpoint:
    """Integration tests for the health check endpoint."""

    def test_health_check(self, api_base_url: str) -> None:
        """Test that GET /health returns 200."""
        url = f"{api_base_url}/health"

        response = requests.get(url)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        assert isinstance(response.json(), dict), "Health response should be a dict"


@pytest.mark.integration
@pytest.mark.live_backend
class TestFilesEndpoint:
    """Integration tests for the files endpoint. Requires live backend."""

    def test_get_available_files(self, api_base_url: str) -> None:
        """Test that GET /files returns a list of files.
        
        This test requires real backend services and will only run with --live-backend flag.
        """
        url = f"{api_base_url}/files"

        response = requests.get(url)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        
        data = response.json()
        assert "files" in data, "Response should contain 'files' key"
        assert isinstance(data["files"], list), "files should be a list"

    def test_get_available_files_has_documents(
        self,
        api_base_url: str,
        imported_documents: list[dict],
        sample_core_data_ids: list[str]
    ) -> None:
        """Test that imported documents appear in GET /files.
        
        This test requires real backend services and will only run with --live-backend flag.
        """
        url = f"{api_base_url}/files"

        response = requests.get(url)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        
        data = response.json()
        files = data.get("files", [])
        
        # Check that at least some of our imported documents are present
        imported_ids = set(sample_core_data_ids)
        found_ids = set()
        
        for file_info in files:
            file_id = file_info.get("id", "")
            file_meta_core_data_id = file_info.get("meta", {}).get("core_data_id", "")
            
            if file_id in imported_ids:
                found_ids.add(file_id)
            elif file_meta_core_data_id in imported_ids:
                found_ids.add(file_meta_core_data_id)
        
        logger.info(f"Found {len(found_ids)}/{len(imported_ids)} imported files in database")
        
        # At least one file should be found
        assert len(found_ids) > 0, (
            "None of the imported documents were found in GET /files"
        )
