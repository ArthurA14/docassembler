"""
Elasticsearch implementation of storage interfaces for prompts and templates.

This module provides Elasticsearch-specific wrappers that inherit all functionality
from the base storage classes. No custom logic needed - just type hints for clarity.
"""

from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

from document_common.storage.base import PromptStorage, TemplateStorage


class ElasticsearchPromptStorage(PromptStorage):
    """
    Elasticsearch implementation of PromptStorage.

    Inherits all functionality from base PromptStorage class.
    This class exists primarily for type hinting and clarity.

    Example:
        storage = ElasticsearchPromptStorage(prompts_document_store)
        storage.upsert_prompt("my_prompt", "content...", category="t1")
        prompt = storage.get_prompt("my_prompt")
    """

    def __init__(self, document_store: ElasticsearchDocumentStore):
        super().__init__(document_store)
        self.ds: ElasticsearchDocumentStore = document_store


class ElasticsearchTemplateStorage(TemplateStorage):
    """
    Elasticsearch implementation of TemplateStorage.

    Inherits all functionality from base TemplateStorage class.
    This class exists primarily for type hinting and clarity.

    Example:
        storage = ElasticsearchTemplateStorage(templates_document_store)
        storage.upsert_template("t1", "content...")
        template = storage.get_template("t1")
    """

    def __init__(self, document_store: ElasticsearchDocumentStore):
        super().__init__(document_store)
        self.ds: ElasticsearchDocumentStore = document_store
