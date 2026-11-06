"""Pytest configuration and fixtures for API end-to-end tests.

This module provides:
- Command line option --live-backend for running tests with real backend services
- Fixtures for API testing and database verification
- Helper functions for test execution and logging
- Shared test data fixtures with proper setup/teardown

Usage:
    pytest api/test/test_import.py                    # Run tests only (polling only)
    pytest api/test/test_import.py --live-backend     # Run all tests including database verification
    pytest api/test/ --live-backend                   # Run entire test suite with real backend
"""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import requests
from document_common.components.base.document_store import DocumentStoreProvider

# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_addoption(parser):
    """Add command line options for pytest."""
    parser.addoption(
        "--live-backend",
        action="store_true",
        default=False,
        help="Enable tests that require real backend services (database, etc.)"
    )
    parser.addoption(
        "--poll-timeout",
        action="store",
        default="300",
        help="Maximum polling timeout in seconds (default: 300)"
    )
    parser.addoption(
        "--poll-interval",
        action="store",
        default="5",
        help="Polling interval in seconds (default: 5)"
    )


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "live_backend: mark test to run only with --live-backend flag"
    )
    config.addinivalue_line(
        "markers", "importation: marks tests as importation tests (run first)"
    )
    config.addinivalue_line(
        "markers", "document: marks tests as document assembly tests (require importation)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to run importation tests before document tests."""
    # If --live-backend is NOT specified, skip tests marked with live_backend
    if not config.getoption("--live-backend"):
        skip_live_backend = pytest.mark.skip(reason="need --live-backend option to run")
        for item in items:
            if "live_backend" in item.keywords:
                item.add_marker(skip_live_backend)

    # Ensure importation tests run before document tests
    importation_order = {}
    document_order = {}
    other_order = {}

    for i, item in enumerate(items):
        if "importation" in item.keywords:
            importation_order[item] = i
        elif "document" in item.keywords:
            document_order[item] = i
        else:
            other_order[item] = i

    # Reorder: other tests first, then importation, then document
    reordered = (
        list(other_order.keys()) +
        list(importation_order.keys()) +
        list(document_order.keys())
    )
    if reordered:
        items[:] = reordered

# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:5000")
TEST_DOCUMENTS_DIR = Path("api_gateway/test/test_documents")

# Configure test logger
TEST_LOGGER_NAME = "test.e2e"
logger = logging.getLogger(TEST_LOGGER_NAME)


# ============================================================================
# Pytest Hooks for Test Execution Logging
# ============================================================================

def pytest_runtest_setup(item: pytest.Item) -> None:
    """Log before each test starts."""
    logger.info(f"{'='*60}")
    logger.info(f"STARTING TEST: {item.name}")
    logger.info(f"{'='*60}")


def pytest_runtest_teardown(item: pytest.Item) -> None:
    """Log after each test completes."""
    logger.info(f"{'='*60}")
    logger.info(f"COMPLETED TEST: {item.name}")
    logger.info(f"{'='*60}\n")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def api_base_url() -> str:
    """Return the base URL for the API."""
    logger.info(f"API Base URL: {API_BASE_URL}")
    return API_BASE_URL


@pytest.fixture(scope="module")
def poll_timeout(request) -> int:
    """Return the polling timeout from command line options."""
    return int(request.config.getoption("--poll-timeout"))


@pytest.fixture(scope="module")
def poll_interval(request) -> int:
    """Return the polling interval from command line options."""
    return int(request.config.getoption("--poll-interval"))


@pytest.fixture(scope="module")
def test_documents_dir() -> Path:
    """Return the path to test documents directory."""
    logger.info(f"Test documents directory: {TEST_DOCUMENTS_DIR}")
    return TEST_DOCUMENTS_DIR


@pytest.fixture(scope="module")
def document_store_provider() -> DocumentStoreProvider:
    """Initialize and return a DocumentStoreProvider for database verification."""
    logger.info("Initializing DocumentStoreProvider")
    provider = DocumentStoreProvider()
    logger.info("DocumentStoreProvider initialized successfully")
    return provider


@pytest.fixture(scope="module")
def sample_core_data_ids() -> List[str]:
    """Generate unique UUIDs for test files."""
    ids = [str(uuid.uuid4()) for _ in range(5)]
    logger.info(f"Generated {len(ids)} sample coreDataIds:")
    for i, doc_id in enumerate(ids):
        logger.info(f"  [{i+1}] {doc_id}")
    return ids


@pytest.fixture(scope="module")
def test_documents(test_documents_dir: Path) -> Dict[str, str]:
    """Read all test documents and return their contents."""
    logger.info(f"Reading test documents from {test_documents_dir}")
    documents = {}
    for txt_file in sorted(test_documents_dir.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
            documents[txt_file.name] = content
            logger.info(f"  Loaded: {txt_file.name} ({len(content)} chars)")
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


@pytest.fixture(scope="module")
def doc_refs(sample_core_data_ids: List[str], test_documents: Dict[str, str]) -> List[Dict]:
    """Create DocRef objects from test documents with unique coreDataIds."""
    logger.info("Creating DocRef objects from test documents")
    doc_refs_list = []
    for (filename, content), core_data_id in zip(test_documents.items(), sample_core_data_ids):
        doc_ref = {
            "coreDataId": core_data_id,
            "coreType": "document",
            "coreDataSource": "Test Documents",
            "plainText": content
        }
        doc_refs_list.append(doc_ref)
        logger.debug(f"  Created DocRef: {filename} -> {core_data_id}")
    logger.info(f"Created {len(doc_refs_list)} DocRef objects")
    return doc_refs_list


@pytest.fixture(scope="module")
def imported_documents(
    api_base_url: str,
    doc_refs: List[Dict],
    sample_core_data_ids: List[str],
    poll_timeout: int,
    poll_interval: int
) -> List[Dict]:
    """import test documents and wait for completion. Available to all tests in the module."""
    logger.info("=== importation FIXTURE: importing documents ===")
    url = f"{api_base_url}/import"

    # Step 1: Submit documents for importation
    response = requests.post(
        url,
        json={"docs": doc_refs},
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code in [202, 204], (
        f"Expected 202/204, got {response.status_code}: {response.text}"
    )

    if response.status_code == 204:
        logger.info("Documents already being processed (204 No Content)")
        yield doc_refs
        return

    # Step 2: Wait for each document to complete importation
    for core_data_id in sample_core_data_ids:
        status = wait_for_importation_status(
            api_base_url=api_base_url,
            core_data_id=core_data_id,
            expected_status="imported",
            timeout=poll_timeout,
            interval=poll_interval
        )
        assert status == "imported", (
            f"Document {core_data_id} did not complete importation. Final status: {status}"
        )

    logger.info("=== importation FIXTURE: All documents imported successfully ===")
    yield doc_refs


# ============================================================================
# Helper Functions with Logging
# ============================================================================

def wait_for_importation_status(
    api_base_url: str,
    core_data_id: str,
    expected_status: str = "imported",
    timeout: int = 300,
    interval: int = 5
) -> str:
    """
    Poll the importation status endpoint until expected status is reached or timeout.

    Args:
        api_base_url: Base URL of the API
        core_data_id: ID of the document to check
        expected_status: Status to wait for (default: "imported")
        timeout: Maximum time to wait in seconds
        interval: Time between polls in seconds

    Returns:
        Final status string

    Raises:
        TimeoutError: If timeout is reached before expected status
        RuntimeError: If importation fails
    """
    logger.info(f"Waiting for importation status for {core_data_id} (expected: {expected_status})")
    start_time = time.time()
    attempts = 0
    url = f"{api_base_url}/import/{core_data_id}/status"

    while time.time() - start_time < timeout:
        attempts += 1
        logger.debug(f"  Polling attempt {attempts}: GET {url}")
        response = requests.get(url)
        logger.debug(f"  Response status: {response.status_code}")

        if response.status_code == 200:
            status = response.json().get("status")
            logger.debug(f"  Current status: {status}")
            if status == expected_status:
                elapsed = time.time() - start_time
                logger.info(f"  importation completed for {core_data_id} after {attempts} attempts in {elapsed:.1f}s")
                return status
            elif status == "failed":
                elapsed = time.time() - start_time
                raise RuntimeError(f"importation failed for {core_data_id} after {attempts} attempts ({elapsed:.1f}s)")
        elif response.status_code == 404:
            logger.debug("  Document not yet indexed, continuing to wait...")
        else:
            raise RuntimeError(f"Unexpected status code {response.status_code}: {response.text}")

        time.sleep(interval)

    elapsed = time.time() - start_time
    raise TimeoutError(f"Timeout waiting for {core_data_id} to reach status '{expected_status}' after {elapsed:.1f}s ({attempts} attempts)")


def verify_file_in_database(
    document_store_provider: DocumentStoreProvider,
    core_data_id: str
) -> bool:
    """
    Verify that a file exists in the 'files' index with the given core_data_id.

    Args:
        document_store_provider: Provider for document stores
        core_data_id: ID to search for

    Returns:
        True if file exists, False otherwise
    """
    logger.info(f"Verifying file exists in 'files' index with core_data_id={core_data_id}")
    store = document_store_provider.get_store("files")
    filters = {
        'operator': 'AND',
        'conditions': [
            {'field': 'meta.core_data_id', 'operator': '==', 'value': core_data_id}
        ]
    }
    docs = store.filter_documents(filters=filters)
    logger.debug(f"  Query returned {len(docs)} document(s)")
    if len(docs) > 0:
        logger.info(f"  File found: {core_data_id}")
    else:
        logger.warning(f"  File NOT found: {core_data_id}")
    return len(docs) > 0


def verify_documents_in_database(
    document_store_provider: DocumentStoreProvider,
    core_data_id: str
) -> int:
    """
    Verify that document chunks exist in the 'documents' index with the given core_data_id.

    Checks for original, refined_summary, and entity_extraction types.

    Args:
        document_store_provider: Provider for document stores
        core_data_id: core_data_id to search for

    Returns:
        Number of valid document chunks found (0 if none found)
    """
    logger.info(f"Verifying documents in 'documents' index with core_data_id={core_data_id}")
    store = document_store_provider.get_store("documents")

    # Check original documents
    originals = store.filter_documents(filters={
        'operator': 'AND',
        'conditions': [
            {'field': 'meta.type', 'operator': '==', 'value': 'original'},
            {'field': 'meta.core_data_id', 'operator': '==', 'value': core_data_id}
        ]
    })
    logger.debug(f"  Found {len(originals)} original chunks")

    # Check refined summaries
    refined_summaries = store.filter_documents(filters={
        'operator': 'AND',
        'conditions': [
            {'field': 'meta.type', 'operator': '==', 'value': 'refined_summary'},
            {'field': 'meta.core_data_id', 'operator': '==', 'value': core_data_id}
        ]
    })
    logger.debug(f"  Found {len(refined_summaries)} refined summary chunks")

    # Check entity extractions
    entity_extractions = store.filter_documents(filters={
        'operator': 'AND',
        'conditions': [
            {'field': 'meta.type', 'operator': '==', 'value': 'entity_extraction'},
            {'field': 'meta.core_data_id', 'operator': '==', 'value': core_data_id}
        ]
    })
    logger.debug(f"  Found {len(entity_extractions)} entity extraction chunks")

    total = len(originals) + len(refined_summaries) + len(entity_extractions)
    if total > 0:
        logger.info(f"  Total chunks found for {core_data_id}: {total}")
    else:
        logger.warning(f"  No chunks found for {core_data_id}")

    return total


def wait_for_document_completion(
    api_base_url: str,
    job_id: str,
    timeout: int = 600,
    interval: int = 10
) -> dict:
    """
    Poll the document endpoint until the document is completed or timeout.

    Args:
        api_base_url: Base URL of the API
        job_id: ID of the document job
        timeout: Maximum time to wait in seconds
        interval: Time between polls in seconds

    Returns:
        The final document response JSON

    Raises:
        TimeoutError: If timeout is reached before completion
        RuntimeError: If the document fails
    """
    logger.info(f"Waiting for document completion for job_id={job_id}")
    start_time = time.time()
    attempts = 0
    url = f"{api_base_url}/documents/{job_id}"

    while time.time() - start_time < timeout:
        attempts += 1
        logger.debug(f"  Polling attempt {attempts}: GET {url}")
        response = requests.get(url)
        logger.debug(f"  Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            status = data.get("status")
            logger.debug(f"  Current status: {status}")
            if status == "completed":
                elapsed = time.time() - start_time
                logger.info(f"  document assembly completed for job {job_id} after {attempts} attempts in {elapsed:.1f}s")
                return data
            elif status == "failed":
                elapsed = time.time() - start_time
                raise RuntimeError(f"document assembly failed for job {job_id} after {attempts} attempts ({elapsed:.1f}s)")
        elif response.status_code == 404:
            logger.debug("  Job not yet available, continuing to wait...")
        else:
            raise RuntimeError(f"Unexpected status code {response.status_code}: {response.text}")

        time.sleep(interval)

    elapsed = time.time() - start_time
    raise TimeoutError(f"Timeout waiting for document job {job_id} to complete after {elapsed:.1f}s ({attempts} attempts)")


# ============================================================================
# Template/Prompt Helper Functions with Logging
# ============================================================================

def get_all_templates(api_base_url: str) -> requests.Response:
    """
    Get all templates from the API.

    Args:
        api_base_url: Base URL of the API

    Returns:
        requests.Response object with status_code and JSON content
    """
    url = f"{api_base_url}/templates"
    logger.info(f"Getting all templates: GET {url}")
    response = requests.get(url)
    logger.debug(f"  Response status: {response.status_code}")
    if response.status_code == 200:
        templates = response.json().get("templates", [])
        logger.info(f"  Found {len(templates)} template(s)")
        for t in templates:
            logger.debug(f"    - {t.get('templateId')}")
    return response


def get_all_prompts(api_base_url: str) -> requests.Response:
    """
    Get all prompts from the API.

    Args:
        api_base_url: Base URL of the API

    Returns:
        requests.Response object with status_code and JSON content
    """
    url = f"{api_base_url}/prompts"
    logger.info(f"Getting all prompts: GET {url}")
    response = requests.get(url)
    logger.debug(f"  Response status: {response.status_code}")
    if response.status_code == 200:
        prompts = response.json().get("prompts", [])
        logger.info(f"  Found {len(prompts)} prompt(s)")
        for p in prompts:
            logger.debug(f"    - {p.get('promptId')}")
    return response


def get_prompt(api_base_url: str, prompt_name: Optional[str]) -> requests.Response:
    """
    Get a specific prompt from the API.

    Args:
        api_base_url: Base URL of the API
        prompt_name: Name of the prompt to retrieve

    Returns:
        requests.Response object with status_code and JSON content
    """
    url = f"{api_base_url}/prompts/{prompt_name}"
    logger.info(f"Getting prompt '{prompt_name}': GET {url}")
    response = requests.get(url)
    logger.debug(f"  Response status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        logger.info(f"  Prompt '{data.get('promptId')}' found (version {data.get('version')})")
    return response


def create_template(api_base_url: str, template_name: str, content: str, description: str = None) -> requests.Response:
    """
    Create or update a template.

    Args:
        api_base_url: Base URL of the API
        template_name: Name of the template
        content: YAML content of the template
        description: Optional description

    Returns:
        requests.Response object with status_code and JSON content
    """
    url = f"{api_base_url}/templates/{template_name}"
    payload = {"content": content}
    if description:
        payload["description"] = description

    logger.info(f"Creating template '{template_name}': PUT {url}")
    logger.debug(f"  Payload: content={content[:100]}...")
    response = requests.put(url, json=payload, headers={"Content-Type": "application/json"})
    logger.debug(f"  Response status: {response.status_code}")
    if response.status_code == 200:
        logger.info(f"  Template '{template_name}' created successfully")
    return response


def create_prompt(api_base_url: str, prompt_name: str, content: str) -> requests.Response:
    """
    Create or update a prompt.

    Args:
        api_base_url: Base URL of the API
        prompt_name: Name of the prompt
        content: YAML content of the prompt

    Returns:
        requests.Response object with status_code and JSON content
    """
    url = f"{api_base_url}/prompts/{prompt_name}"
    logger.info(f"Creating prompt '{prompt_name}': PUT {url}")
    logger.debug(f"  Payload: content={content[:100]}...")
    response = requests.put(url, json={"content": content}, headers={"Content-Type": "application/json"})
    logger.debug(f"  Response status: {response.status_code}")
    if response.status_code == 200:
        logger.info(f"  Prompt '{prompt_name}' created successfully")
    return response


def template_exists_in_list(templates_response: dict, template_name: str) -> bool:
    """
    Check if a template exists in the templates list response.

    Args:
        templates_response: Response from GET /templates
        template_name: Name of the template to find

    Returns:
        True if template exists, False otherwise
    """
    templates = templates_response.get("templates", [])
    return any(t.get("templateId") == template_name for t in templates)


def prompt_exists_in_list(prompts_response: dict, prompt_name: str) -> bool:
    """
    Check if a prompt exists in the prompts list response.

    Args:
        prompts_response: Response from GET /prompts
        prompt_name: Name of the prompt to find

    Returns:
        True if prompt exists, False otherwise
    """
    prompts = prompts_response.get("prompts", [])
    return any(p.get("promptId") == prompt_name for p in prompts)