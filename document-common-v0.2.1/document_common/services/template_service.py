"""
Service layer for managing templates WITHOUT versioning.

This service provides high-level operations for template management,
using a TemplateStorage implementation to handle persistence.

The service is backend-agnostic - it works with any TemplateStorage
implementation (Elasticsearch, Qdrant, InMemory, etc.)
"""

from pathlib import Path
from typing import List, Optional

import yaml

from document_common.schemas.document_template import DocumentTemplate
from document_common.storage.base import TemplateStorage
from document_common.storage.factory import TemplateStorageFactory
from document_common.utils.logger import Logger


class TemplateService:
    """
    Service for managing templates without versioning.

    This service provides business logic for:
    - Loading templates from YAML files into storage backend
    - Creating or updating templates (upsert)
    - Deleting templates
    - Retrieving templates
    - Listing all templates

    Example:
        template_service = TemplateService(template_storage)
        template_service.initialize_from_yaml_files(Path("config/templates/document/"))
        template_service.upsert_template("my_template", "content...")
    """

    def __init__(self, storage: TemplateStorage | None = None):
        """
        Initialize the service with a storage backend.

        Args:
            storage: TemplateStorage implementation (backend-agnostic)
        """
        self.storage = storage if storage else TemplateStorageFactory.create()
        self.logger = Logger().get_logger("TemplateService")

    def initialize_from_yaml_files(self, root_folder: Path) -> int:
        """
        Load all YAML files from templates/ directory into the storage backend.

        This method is idempotent:
        - If template doesn't exist: create it
        - If template exists and content changed: update it
        - If template exists and content identical: skip

        Args:
            root_folder: Path to templates/ directory (e.g., config/templates/document/)

        Returns:
            int: Number of templates created/updated
        """
        count = 0

        self.logger.info(f"Initializing templates from {root_folder}")

        for yaml_file in root_folder.rglob("*.yaml"):
            try:
                with open(yaml_file, "r") as f:
                    yaml_content = f.read()
                    data = yaml.safe_load(yaml_content)

                if not data:
                    self.logger.warning(f"Empty YAML file: {yaml_file}")
                    continue

                # Extract template_name from the YAML data
                template_name = data.get("template_name")

                if not template_name:
                    self.logger.warning(f"No 'template_name' field in {yaml_file}, skipping")
                    continue

                # Check if exists and content changed
                if self.storage.template_exists(template_name):
                    existing = self.storage.get_template(template_name)

                    if existing and existing.content != yaml_content:
                        # Content changed → update
                        description = data.get("description", f"Auto-updated from {yaml_file.name}")
                        self.storage.upsert_template(
                            template_name=template_name,
                            content=yaml_content,
                            description=description
                        )
                        self.logger.info(f"Updated template '{template_name}' from {yaml_file}")
                        count += 1
                    else:
                        self.logger.debug(f"Template '{template_name}' unchanged, skipping")
                else:
                    # New template
                    description = data.get("description", f"Loaded from {yaml_file.name}")
                    self.storage.upsert_template(
                        template_name=template_name,
                        content=yaml_content,
                        description=description
                    )
                    self.logger.info(f"Created new template '{template_name}' from {yaml_file}")
                    count += 1

            except yaml.YAMLError as e:
                self.logger.error(f"YAML parsing error in {yaml_file}: {e}")
            except Exception as e:
                self.logger.error(f"Error loading {yaml_file}: {e}")

        self.logger.info(f"Initialized {count} templates from YAML files")
        return count
    
    def upsert_template_yaml(
        self,
        content: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Create or update a template from a yaml content.

        Args:
            template_name: Unique identifier for the template
            content: YAML content as string
            description: Human-readable description
            tags: List of tags for categorization
        """
        data = yaml.safe_load(content)

        # Validate template structure
        document_conf = DocumentTemplate.model_validate(data)
        template_name = document_conf.template_name

        final_description = description or data.get("description", "Added via API")

        self.storage.upsert_template(
            template_name=template_name,
            content=content,
            description=final_description,
            tags=tags
        )
        self.logger.info(f"Upserted template '{template_name}'")

    def upsert_template(
        self,
        template_name: str,
        content: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        duplicate_rejection: bool = False
    ) -> None:
        """
        Create or update a template.

        Args:
            template_name: Unique identifier for the template
            content: YAML content as string
            description: Human-readable description
            tags: List of tags for categorization
        """
        self.storage.upsert_template(
            template_name,
            content,
            description=description,
            tags=tags,
            duplicate_rejection=duplicate_rejection
        )
        self.logger.info(f"Upserted template '{template_name}'")

    def get_template(self, template_name: str) -> str:
        """
        Get YAML content of a template.

        Args:
            template_name: Name of the template

        Returns:
            str: YAML content

        Raises:
            ValueError: If template not found
        """
        doc = self.storage.get_template(template_name)
        if not doc or not doc.content:
            raise ValueError(f"Template '{template_name}' not found")
        return doc.content

    def delete_template(self, template_name: str) -> None:
        """
        Delete a template.

        Args:
            template_name: Name of the template to delete

        Raises:
            ValueError: If template doesn't exist
        """
        self.storage.delete_template(template_name)
        self.logger.info(f"Deleted template '{template_name}'")

    def list_all_templates(self):
        """
        List all templates.

        Returns:
            List of dicts with template metadata
        """
        return self.storage.list_all_templates()

    def template_exists(self, template_name: str) -> bool:
        """
        Check if template exists.

        Args:
            template_name: Name to check

        Returns:
            True if exists, False otherwise
        """
        return self.storage.template_exists(template_name)
