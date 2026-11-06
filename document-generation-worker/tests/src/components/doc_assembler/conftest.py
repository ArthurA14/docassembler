"""
Pytest configuration and fixtures for document assembly tests.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_citation_config():
    """
    Auto-use fixture that mocks citation configuration for all tests.

    This prevents tests from failing due to missing citation config in the
    test environment.
    """
    with patch("src.components.doc_assembler.citation_extractor.config") as mock_config:
        # Create a mock config with citations section
        mock_current_config = MagicMock()
        mock_current_config.citations = {}
        mock_config.get_current_config.return_value = mock_current_config
        yield mock_config
