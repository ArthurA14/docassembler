"""
Qdrant implementation of storage interfaces for prompts and templates.

This module provides Qdrant-specific wrappers that inherit all functionality
from the base storage classes. No custom logic needed - just type hints for clarity.
"""

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from document_common.storage.base import PromptStorage, TemplateStorage


class QdrantPromptStorage(PromptStorage):
    """
    Qdrant implementation of PromptStorage.

    Inherits all functionality from base PromptStorage class.
    This class exists primarily for type hinting and clarity.

    Example:
        storage = QdrantPromptStorage(prompts_document_store)
        storage.upsert_prompt("my_prompt", "content...", category="t1")
        prompt = storage.get_prompt("my_prompt")
    """

    def __init__(self, document_store: QdrantDocumentStore):
        super().__init__(document_store)
        self.ds: QdrantDocumentStore = document_store


class QdrantTemplateStorage(TemplateStorage):
    """
    Qdrant implementation of TemplateStorage.

    Inherits all functionality from base TemplateStorage class.
    This class exists primarily for type hinting and clarity.

    Example:
        storage = QdrantTemplateStorage(templates_document_store)
        storage.upsert_template("t1", "content...")
        template = storage.get_template("t1")
    """

    def __init__(self, document_store: QdrantDocumentStore):
        super().__init__(document_store)
        self.ds: QdrantDocumentStore = document_store
