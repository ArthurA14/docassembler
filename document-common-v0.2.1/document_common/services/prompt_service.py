"""
Service layer for managing prompts with simplified versioning.

This service provides high-level operations for prompt management,
using a PromptStorage implementation to handle persistence.

The service is backend-agnostic - it works with any PromptStorage
implementation (Elasticsearch, Qdrant, InMemory, etc.)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from document_common.storage.base import PromptStorage
from document_common.storage.factory import PromptStorageFactory
from document_common.utils.logger import Logger


class PromptService:
    """
    Service for managing prompts with simplified versioning.

    This service provides business logic for:
    - Loading prompts from YAML files into storage backend
    - Creating/updating prompts
    - Retrieving prompts
    - Listing all prompts
    - Deleting prompts

    Version management:
    - Version is stored in the YAML content (version: 1)
    - Type is extracted from folder structure (e.g., "t1", "citations")
    - Mode is extracted from filename (_reasoning = "think", otherwise "no_think")
    - Name is extracted from YAML keys (each top-level key except "version" is a prompt)

    Example:
        prompt_service = PromptService(prompt_storage)
        prompt_service.initialize_from_yaml_files(Path("prompts/"))
        prompt_service.upsert_prompt("my_prompt", "content...")
    """

    def __init__(self, storage: Optional[PromptStorage] = None):
        """
        Initialize the service with a storage backend.

        Args:
            storage: PromptStorage implementation (backend-agnostic)
        """
        self.storage = storage if storage else PromptStorageFactory.create()
        self.logger = Logger().get_logger("PromptService")

    def _extract_metadata_from_path(self, yaml_file: Path, root_folder: Path) -> Dict[str, Any]:
        """
        Extract type and mode from file path.

        Args:
            yaml_file: Path to the YAML file
            root_folder: Root prompts folder

        Returns:
            Dict with 'type' and 'mode'
        """
        # Extract type from first parent folder
        relative_path = yaml_file.relative_to(root_folder)
        type_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else "default"

        # Determine mode from filename
        file_stem = yaml_file.stem
        mode = "think" if "_reasoning" in file_stem else "no_think"

        return {
            "type": type_folder,
            "mode": mode,
        }

    def _extract_version_from_yaml(self, yaml_content: str) -> Optional[int]:
        """
        Extract version from YAML content.

        Args:
            yaml_content: YAML string content

        Returns:
            Version number or None if not found
        """
        try:
            data = yaml.safe_load(yaml_content)
            return data.get("version")
        except Exception:
            return None

    def initialize_from_yaml_files(self, root_folder: Path) -> int:
        """
        Load all YAML files from prompts/ directory into the storage backend.

        Version-aware logic:
        - If prompt doesn't exist in DB: create it
        - If prompt exists in DB with same or higher version: skip it
        - If in-memory YAML has higher version: add new version to DB
        - If no version field in YAML: default to version 1

        A single YAML file can contain multiple prompts. Each top-level key
        (except 'version') is treated as a separate prompt name.

        Args:
            root_folder: Path to prompts/ directory

        Returns:
            int: Number of prompts created/updated
        """
        count = 0

        self.logger.info(f"Initializing prompts from {root_folder}")

        for yaml_file in root_folder.rglob("*.yaml"):
            try:
                with open(yaml_file, "r") as f:
                    yaml_content = f.read()

                data = yaml.safe_load(yaml_content)
                if not isinstance(data, dict):
                    self.logger.warning(
                        f"Skipping {yaml_file}: expected a YAML mapping at top level"
                    )
                    continue

                # Extract version from top level, default to 1 if missing
                version = data.get("version")
                if version is None:
                    self.logger.warning(
                        f"No version field found in {yaml_file}, will set it to 1"
                    )
                    version = 1

                # Extract metadata from path (same for all prompts in this file)
                metadata = self._extract_metadata_from_path(yaml_file, root_folder)
                prompt_type = metadata["type"]
                prompt_mode = metadata["mode"]

                for prompt_name, prompt_data in data.items():
                    if prompt_name == "version":
                        continue

                    # Re-serialize single prompt with version so consumers can parse it
                    single_prompt_doc = {"version": version, prompt_name: prompt_data}
                    single_prompt_yaml = yaml.dump(
                        single_prompt_doc,
                        default_flow_style=False,
                        sort_keys=False,
                        allow_unicode=True,
                    )

                    # Check if exists in DB
                    existing = self.storage.get_prompt(prompt_name)

                    if existing:
                        existing_version = existing.meta.get("version", 0)

                        # Skip if DB has same or higher version
                        if existing_version >= version:
                            self.logger.debug(
                                f"Prompt '{prompt_name}' already exists in DB (v{existing_version}) "
                                f"with same/higher version than YAML (v{version}), skipping"
                            )
                            continue

                        # In-memory YAML has higher version → update DB
                        self.logger.info(
                            f"Prompt '{prompt_name}' YAML (v{version}) is newer than DB (v{existing_version}), updating"
                        )
                        self.storage.upsert_prompt(
                            prompt_name=prompt_name,
                            content=single_prompt_yaml,
                            prompt_type=prompt_type,
                            mode=prompt_mode,
                            version=version,
                        )
                        count += 1
                    else:
                        # New prompt - doesn't exist in DB
                        self.storage.upsert_prompt(
                            prompt_name=prompt_name,
                            content=single_prompt_yaml,
                            prompt_type=prompt_type,
                            mode=prompt_mode,
                            version=version,
                        )
                        self.logger.info(f"Created new prompt '{prompt_name}' from {yaml_file}")
                        count += 1

            except yaml.YAMLError as e:
                error_msg = str(e).replace("[", r"\[").replace("]", r"\]")
                self.logger.error(f"YAML parsing error in {yaml_file}: {error_msg}")
            except Exception as e:
                error_msg = str(e).replace("[", r"\[").replace("]", r"\]")
                self.logger.error(f"Error loading {yaml_file}: {error_msg}")

        self.logger.info(f"Initialized {count} prompts from YAML files")
        return count

    def upsert_prompt(
        self,
        prompt_name: str,
        content: str,
        prompt_type: Optional[str] = None,
        mode: Optional[str] = None,
        duplicate_rejection: bool = False
    ) -> str:
        """
        Create or update a prompt.

        The YAML content should include a version field.

        Args:
            prompt_name: Unique identifier for the prompt (snake-case)
            content: YAML content as string (must include version field)
            prompt_type: Type of the prompt (e.g., "t1", "citations")
            mode: Mode of the prompt ("think" or "no_think")

        Returns:
            str: The prompt name that was upserted

        """
        # Extract version from content, default to 1 if missing
        version = self._extract_version_from_yaml(content)
        if version is None:
            self.logger.warning("No version field found in YAML content, will set it to 1")
            version = 1

        self.storage.upsert_prompt(
            prompt_name=prompt_name,
            content=content,
            prompt_type=prompt_type,
            mode=mode,
            version=version,
            duplicate_rejection=duplicate_rejection
        )

        self.logger.info(f"Upserted prompt '{prompt_name}' (version {version})")
        return prompt_name

    def get_prompt(self, prompt_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a prompt by name.

        Args:
            prompt_name: Name of the prompt

        Returns:
            Dict with prompt data or None if not found
        """
        doc = self.storage.get_prompt(prompt_name)
        if not doc:
            return None

        return {
            "name": doc.meta["prompt_name"],
            "type": doc.meta.get("type"),
            "mode": doc.meta.get("mode"),
            "version": doc.meta.get("version"),
            "content": doc.content,
            "created_at": doc.meta.get("created_at"),
            "updated_at": doc.meta.get("updated_at"),
        }

    def get_all_prompts(self) -> Dict[str, str]:
        """
        Get all prompts as a dict of {name: content}.

        Returns:
            Dict mapping prompt names to their YAML content
        """
        prompts = self.storage.list_all_prompts()
        return {p["name"]: self.storage.get_prompt(p["name"]).content for p in prompts}

    def list_all_prompts(self) -> List[Dict[str, Any]]:
        """
        List all prompts with metadata.

        Returns:
            List of dicts with prompt metadata
        """
        return self.storage.list_all_prompts()

    def delete_prompt(self, prompt_name: str) -> bool:
        """
        Delete a prompt by name.

        Args:
            prompt_name: Name of the prompt to delete

        Returns:
            True if deleted, False if not found
        """
        return self.storage.delete_prompt(prompt_name)

    def prompt_exists(self, prompt_name: str) -> bool:
        """
        Check if prompt exists.

        Args:
            prompt_name: Name to check

        Returns:
            True if exists, False otherwise
        """
        return self.storage.get_prompt(prompt_name) is not None

    def get_latest_content(self, prompt_name: str) -> str:
        """
        Get YAML content of a prompt.

        Args:
            prompt_name: Name of the prompt

        Returns:
            str: YAML content

        Raises:
            ValueError: If prompt not found
        """
        prompt_data = self.get_prompt(prompt_name)
        if not prompt_data:
            raise ValueError(f"Prompt '{prompt_name}' not found")
        return prompt_data["content"]
