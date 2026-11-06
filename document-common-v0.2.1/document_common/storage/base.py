"""
Base storage classes for prompts and templates WITHOUT versioning.

This module provides base classes with CONCRETE implementations that work
with any Haystack DocumentStore backend (Elasticsearch, Qdrant, InMemory, etc.).

The key insight: All backends use the same Haystack interface, so the logic
is identical. Subclasses just inherit - no need to override unless custom behavior needed.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from haystack.dataclasses import Document
from haystack.document_stores.types import DocumentStore
from haystack.document_stores.types.policy import DuplicatePolicy

from document_common.storage.models import TemplateDocument
from document_common.utils.logger import Logger


class PromptStorage:
    """
    Base class for prompt storage WITHOUT versioning.

    This is a CONCRETE implementation that works with any Haystack DocumentStore.
    Subclasses (ElasticsearchPromptStorage, QdrantPromptStorage, etc.) just inherit
    everything - no need to override anything unless they need custom behavior.

    Key features:
    - Simple upsert: create or replace by prompt_name
    - Document ID = prompt_name (no UUID, no versions)
    - Works with any backend that supports Haystack's DocumentStore interface

    Example:
        storage = PromptStorage(document_store)
        storage.upsert_prompt("my_prompt", "content...", category="t1")
    """

    def __init__(self, document_store: DocumentStore):
        self.ds = document_store
        self.logger = Logger().get_logger(self.__class__.__name__)

    def upsert_prompt(
        self,
        prompt_name: str,
        content: str,
        prompt_type: Optional[str] = None,
        mode: Optional[str] = None,
        version: Optional[int] = None,
        duplicate_rejection: bool = False
    ) -> Document:
        """
        Create or replace a prompt with version support.

        Uses prompt_name + version as document ID to preserve version history.
        Each version is stored as a separate document.

        Args:
            prompt_name: Unique identifier for the prompt
            content: YAML content as string
            prompt_type: Type of the prompt (e.g., "t1", "citations", "preprocessing")
            mode: Mode of the prompt ("think" or "no_think")
            version: Version number of the prompt

        Returns:
            Document: The created/updated prompt document
        """
        timestamp = self._get_timestamp()

        # Check if latest version exists to preserve type/mode if not provided
        existing = self.get_prompt(prompt_name)

        if existing and duplicate_rejection:
            raise ValueError(f"A prompt already exists with the name : {prompt_name}.")

        # Preserve type and mode from existing if not provided
        if existing:
            if prompt_type is None:
                prompt_type = existing.meta.get("type")
            if mode is None:
                mode = existing.meta.get("mode")

        # Use version in document ID to keep version history
        # Format: "prompt_name_v{version}"
        doc_id = f"{prompt_name}_v{version}" if version else prompt_name

        doc = Document(
            id=doc_id,  # Unique ID per version
            content=content,
            meta={
                "prompt_name": prompt_name,
                "type": prompt_type,
                "mode": mode,
                "version": version,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
        )

        # Use SKIP policy to avoid overwriting existing versions
        self.ds.write_documents([doc], policy=DuplicatePolicy.SKIP)
        self.logger.info(f"Created prompt '{prompt_name}' version {version} at {timestamp}")

        return doc

    def get_prompt(self, prompt_name: str) -> Optional[Document]:
        """
        Get the latest version of a prompt by name.

        Args:
            prompt_name: Name of the prompt

        Returns:
            Document with highest version if found, None otherwise
        """
        # Get all versions of this prompt
        results = self.ds.filter_documents(
            filters={"field": "meta.prompt_name", "operator": "==", "value": prompt_name}
        )

        if not results:
            return None

        # Return the one with highest version
        return max(results, key=lambda d: d.meta.get("version", 0))

    def get_all_versions(self, prompt_name: str) -> List[Document]:
        """
        Get all versions of a prompt.

        Args:
            prompt_name: Name of the prompt

        Returns:
            List of Documents sorted by version (newest first)
        """
        results = self.ds.filter_documents(
            filters={"field": "meta.prompt_name", "operator": "==", "value": prompt_name}
        )
        return sorted(results, key=lambda d: d.meta.get("version", 0), reverse=True)

    def delete_prompt(self, prompt_name: str) -> bool:
        """
        Delete ALL versions of a prompt by name.

        Args:
            prompt_name: Name of the prompt to delete

        Returns:
            True if deleted, False if not found
        """
        if not self.prompt_exists(prompt_name):
            return False

        # Get all versions
        versions = self.get_all_versions(prompt_name)
        doc_ids = [doc.id for doc in versions]

        # Delete all versions
        self.ds.delete_documents(doc_ids)
        self.logger.info(f"Deleted all versions of prompt '{prompt_name}' ({len(doc_ids)} versions)")
        return True

    def list_all_prompts(self) -> List[Dict[str, Any]]:
        """
        List all prompts (latest version only for each prompt name).

        Returns:
            List of dicts with prompt metadata (name, type, mode, version, etc.)
        """
        # Get all documents
        docs = self.ds.filter_documents()

        # Group by prompt_name and keep only latest version
        prompts_dict: dict[str, Document] = {}
        for doc in docs:
            name = doc.meta["prompt_name"]
            version = doc.meta.get("version", 0)

            if name not in prompts_dict or version > prompts_dict[name].meta.get("version", 0):
                prompts_dict[name] = doc

        return [
            {
                "prompt": doc.content,
                "name": doc.meta["prompt_name"],
                "type": doc.meta.get("type"),
                "mode": doc.meta.get("mode"),
                "version": doc.meta.get("version"),
                "created_at": doc.meta.get("created_at"),
                "updated_at": doc.meta.get("updated_at"),
            }
            for doc in prompts_dict.values()
        ]

    def prompt_exists(self, prompt_name: str) -> bool:
        """
        Check if a prompt exists.

        Args:
            prompt_name: Name to check

        Returns:
            True if exists, False otherwise
        """
        return self.get_prompt(prompt_name) is not None

    # Compatibility methods for old versioning API
    def create_prompt(
        self,
        prompt_name: str,
        content: str,
        category: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs  # Ignore old params like created_by, source
    ) -> Document:
        """Compatibility wrapper for old create_prompt - delegates to upsert_prompt"""
        # Map old params to new params (category -> type)
        return self.upsert_prompt(prompt_name, content, prompt_type=category)

    def add_version(
        self,
        prompt_name: str,
        content: str,
        change_message: Optional[str] = None,
        **kwargs  # Ignore old params
    ) -> Document:
        """Compatibility wrapper for old add_version - delegates to upsert_prompt"""
        # Extract metadata from existing prompt if available
        existing = self.get_prompt(prompt_name)
        prompt_type = existing.meta.get("type") if existing else None
        mode = existing.meta.get("mode") if existing else None
        return self.upsert_prompt(prompt_name, content, prompt_type=prompt_type, mode=mode)

    def get_latest(self, prompt_name: str) -> Optional[Document]:
        """Compatibility wrapper for old get_latest - delegates to get_prompt"""
        return self.get_prompt(prompt_name)

    def get_latest_content(self, prompt_name: str) -> str:
        """Compatibility wrapper for old get_latest_content"""
        doc = self.get_prompt(prompt_name)
        if not doc or not doc.content:
            raise ValueError(f"Prompt '{prompt_name}' not found")
        return doc.content

    def get_version_history(self, prompt_name: str) -> List[Document]:
        """Compatibility wrapper - returns single version (no history)"""
        doc = self.get_prompt(prompt_name)
        return [doc] if doc else []

    # Utility methods
    def _get_timestamp(self) -> str:
        """Get current UTC timestamp in ISO format"""
        return datetime.now(timezone.utc).isoformat()
    

class TemplateStorage:
    """
    Base class for template storage WITHOUT versioning.

    This is a CONCRETE implementation that works with any Haystack DocumentStore.
    Identical pattern to PromptStorage but for templates.

    Key features:
    - Simple upsert: create or replace by template_name
    - Document ID = template_name (no UUID, no versions)
    - Works with any backend that supports Haystack's DocumentStore interface

    Example:
        storage = TemplateStorage(document_store)
        storage.upsert_template("t1", "content...")
        content = storage.get_template("t1").content
    """

    def __init__(self, document_store: DocumentStore):
        self.ds = document_store
        self.logger = Logger().get_logger(self.__class__.__name__)

    def upsert_template(
        self,
        template_name: str,
        content: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        duplicate_rejection: bool = False
    ) -> Document:
        """
        Create or replace a template (idempotent).

        Uses template_name as document ID and OVERWRITE policy.
        If template exists, it's completely replaced.

        Args:
            template_name: Unique identifier (used as document ID)
            content: YAML content as string
            description: Human-readable description
            tags: List of tags for categorization

        Returns:
            Document: The created/updated template document
        """
        timestamp = self._get_timestamp()

        # Check if exists to preserve created_at
        existing = self.get_template(template_name)
        if existing and duplicate_rejection:
            raise ValueError(f"A template already exists with the name : {template_name}.")

        created_at = existing.meta.get("created_at") if existing else timestamp

        doc = Document(
            id=template_name,  # ID = name (no UUID!)
            content=content,
            meta={
                "template_name": template_name,
                "description": description,
                "tags": tags or [],
                "created_at": created_at,
                "updated_at": timestamp,
            }
        )

        self.ds.write_documents([doc], policy=DuplicatePolicy.OVERWRITE)
        action = "Updated" if existing else "Created"
        self.logger.info(f"{action} template '{template_name}' at {timestamp}")

        return doc

    def get_template(self, template_name: str) -> Optional[Document]:
        """
        Get a template by name.

        Args:
            template_name: Name of the template

        Returns:
            Document if found, None otherwise
        """
        # Direct retrieval by ID
        results = self.ds.filter_documents(
            filters={"field": "id", "operator": "==", "value": template_name}
        )
        return results[0] if results else None

    def delete_template(self, template_name: str) -> None:
        """
        Delete a template by name.

        Args:
            template_name: Name of the template to delete

        Raises:
            ValueError: If template doesn't exist
        """
        if not self.template_exists(template_name):
            raise ValueError(f"Template '{template_name}' not found")

        self.ds.delete_documents([template_name])
        self.logger.info(f"Deleted template '{template_name}'")

    def list_all_templates(self) -> List[TemplateDocument]:
        """
        List all templates.

        Returns:
            List of dicts with template metadata (name, description, etc.)
        """
        # Get all documents
        docs = self.ds.filter_documents()

        return [
            TemplateDocument(
                name = doc.meta["template_name"],
                description = doc.meta.get("description"),
                tags = doc.meta.get("tags", []),
                created_at = doc.meta.get("created_at"),
                updated_at = doc.meta.get("updated_at"),
                content = doc.content,
            )
            for doc in docs
        ]

    def template_exists(self, template_name: str) -> bool:
        """
        Check if a template exists.

        Args:
            template_name: Name to check

        Returns:
            True if exists, False otherwise
        """
        return self.get_template(template_name) is not None

    # Utility methods
    def _get_timestamp(self) -> str:
        """Get current UTC timestamp in ISO format"""
        return datetime.now(timezone.utc).isoformat()
