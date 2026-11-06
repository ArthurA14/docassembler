"""
In-memory implementation of storage interfaces for prompts and templates.

This module provides in-memory wrappers that inherit all functionality
from the base storage classes. No custom logic needed - just type hints for clarity.
Useful for testing and development. No persistence - data is lost on restart.
"""

from haystack.document_stores.in_memory.document_store import InMemoryDocumentStore

from document_common.storage.base import PromptStorage, TemplateStorage


class InMemoryPromptStorage(PromptStorage):
    """
    In-memory implementation of PromptStorage.

    Inherits all functionality from base PromptStorage class.
    This class exists primarily for type hinting and clarity.
    No persistence - data lost on restart.

    Example:
        storage = InMemoryPromptStorage(inmemory_document_store)
        storage.upsert_prompt("test_prompt", "content...")
        prompt = storage.get_prompt("test_prompt")
    """

    def __init__(self, document_store: InMemoryDocumentStore):
        super().__init__(document_store)
        self.ds: InMemoryDocumentStore = document_store


class InMemoryTemplateStorage(TemplateStorage):
    """
    In-memory implementation of TemplateStorage.

    Inherits all functionality from base TemplateStorage class.
    This class exists primarily for type hinting and clarity.
    No persistence - data lost on restart.

    Example:
        storage = InMemoryTemplateStorage(inmemory_document_store)
        storage.upsert_template("test_template", "content...")
        template = storage.get_template("test_template")
    """

    def __init__(self, document_store: InMemoryDocumentStore):
        super().__init__(document_store)
        self.ds: InMemoryDocumentStore = document_store
