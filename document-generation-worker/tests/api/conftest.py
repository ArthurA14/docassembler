"""Pytest configuration and fixtures for API tests."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest


def pytest_configure(config):
    """Configure pytest before running tests."""
    # Set environment variables before any imports
    os.environ.setdefault("HF_HOME", "/tmp/hf_home")
    os.environ.setdefault("ELASTICSEARCH_HOST", "localhost")
    os.environ.setdefault("ELASTICSEARCH_PORT", "9200")

    # Mock the config module before imports
    sys.modules["config.config"] = MagicMock()


@pytest.fixture(autouse=True)
def mock_config_module():
    """Mock configuration to prevent loading errors during tests."""
    with patch("config.config"):
        yield
