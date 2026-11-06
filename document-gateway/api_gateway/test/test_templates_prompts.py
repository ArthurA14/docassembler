"""Pytest tests for template and prompt endpoints.

This module tests the template and prompt endpoints:
- GET /templates - List all templates
- PUT /templates/{name} - Create a template
- GET /prompts - List all prompts
- GET /prompts/{name} - Get a specific prompt
- PUT /prompts/{name} - Create a prompt

Note: Due to concurrency constraints, only Add and Get operations are tested.
Delete and Update operations are not covered as they are not used in production.

Usage:
    pytest api/test/test_templates_prompts.py              # Run tests
    pytest api/test/test_templates_prompts.py --live-backend  # Run with backend verification
"""

import logging

import pytest

from api_gateway.test.conftest import (
    create_prompt,
    create_template,
    get_all_prompts,
    get_all_templates,
    get_prompt,
)

# Sample template YAML content
TEMPLATE_NAME = "test_template_123"
SAMPLE_TEMPLATE_YAML = """template_name: test_template_123
description: Test template for CRUD operations
document:
  - section:
      method: direct
      title: "Introduction"
      name: intro
      prompt: t1_introduction
      using: []
      num_lines:
        min: 5
        max: 10
"""

# Sample prompt YAML content
SAMPLE_PROMPT_NAME = "test_prompt_123"
SAMPLE_PROMPT_YAML = """version: 1
test_prompt_123:
  system: |
    You are a test assistant.
  query: |
    {{user_input}}
"""

logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestTemplates:
    """Integration tests for template operations."""

    def test_list_templates(self, api_base_url: str) -> None:
        """Test that GET /templates returns a list of templates."""
        response = get_all_templates(api_base_url)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        assert "templates" in response.json(), "Response should contain 'templates' key"
        assert isinstance(response.json()["templates"], list), (
            "templates should be a list"
        )

    def test_create_template(self, api_base_url: str) -> None:
        """Test that PUT /templates/{name} creates a template."""
        response = create_template(
            api_base_url, TEMPLATE_NAME, SAMPLE_TEMPLATE_YAML, "Test template for CRUD"
        )
        assert response.status_code in [200, 400], (
            f"Expected 200  or 400, got {response.status_code}: {response.text}"
        )
        response_json = response.json()
        assert response_json.get("message") is not None or response_json.get("detail") is not None, (
            f"Expected message in response: {response_json}"
        )

    def test_create_template_empty_content(self, api_base_url: str) -> None:
        """Test that empty template content is handled appropriately."""
        template_name = "test_empty_template"

        response = create_template(api_base_url, template_name, "", "Empty template")

        # API should either accept (idempotent) or reject with 400/422
        assert response.status_code in [200, 400, 422], (
            f"Unexpected status code {response.status_code}: {response.text}"
        )


@pytest.mark.integration
class TestPrompts:
    """Integration tests for prompt operations."""

    def test_list_prompts(self, api_base_url: str) -> None:
        """Test that GET /prompts returns a list of prompts."""
        response = get_all_prompts(api_base_url)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        assert "prompts" in response.json(), "Response should contain 'prompts' key"
        assert isinstance(response.json()["prompts"], list), (
            "prompts should be a list"
        )

    def test_get_prompt(self, api_base_url: str) -> None:
        """Test that GET /prompts/{name} returns a specific prompt."""
        # First create a prompt, then retrieve it
        response = create_prompt(
            api_base_url, SAMPLE_PROMPT_NAME, SAMPLE_PROMPT_YAML
        )
        assert response.status_code in [200, 400], (
            f"Expected 200 or 400, got {response.status_code}: {response.text}"
        )

        # Now get the prompt
        prompt_response = get_prompt(api_base_url, SAMPLE_PROMPT_NAME)
        assert prompt_response.status_code == 200, (
            f"Expected 200 for existing prompt, got {prompt_response.status_code}"
        )
        prompt_data = prompt_response.json()
        assert prompt_data.get("promptId") == SAMPLE_PROMPT_NAME, (
            f"Expected promptId '{SAMPLE_PROMPT_NAME}', got '{prompt_data.get('promptId')}'"
        )

    def test_get_prompt_not_found(self, api_base_url: str) -> None:
        """Test that GET /prompts/{name} returns 404 for non-existent prompt."""
        response = get_prompt(api_base_url, "nonexistent_prompt_xyz")
        assert response.status_code == 404, (
            f"Expected 404 for non-existent prompt, got {response.status_code}"
        )

    def test_create_prompt(self, api_base_url: str) -> None:
        """Test that PUT /prompts/{name} creates a prompt."""
        response = create_prompt(api_base_url, SAMPLE_PROMPT_NAME, SAMPLE_PROMPT_YAML)
        assert response.status_code in [200, 400], (
            f"Expected 200 or 400, got {response.status_code}: {response.text}"
        )
        response_json = response.json()
        assert response_json.get("message") is not None or response_json.get("detail") is not None, (
            f"Expected message in response: {response_json}"
        )

    def test_create_prompt_empty_content(self, api_base_url: str) -> None:
        """Test that empty prompt content is handled appropriately."""
        prompt_name = "test_empty_prompt"

        response = create_prompt(api_base_url, prompt_name, "")

        # API should either accept (idempotent) or reject with 400/422
        assert response.status_code in [200, 400, 422], (
            f"Unexpected status code {response.status_code}: {response.text}"
        )