"""
Factory pattern for creating appropriate storage implementations based on DocumentStore type.

This module provides factory classes that automatically instantiate the correct
storage implementation (Elasticsearch, Qdrant, InMemory) based on the type of
DocumentStore provided.
"""

from typing import Optional

from haystack.document_stores.in_memory.document_store import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from document_common.components.base.document_store import DocumentStoreProvider
from document_common.storage.base import PromptStorage, TemplateStorage
from document_common.storage.elasticsearch_storage import (
    ElasticsearchPromptStorage,
    ElasticsearchTemplateStorage,
)
from document_common.storage.inmemory_storage import InMemoryPromptStorage, InMemoryTemplateStorage
from document_common.storage.qdrant_storage import QdrantPromptStorage, QdrantTemplateStorage


class PromptStorageFactory:
    """
    Factory to create the appropriate PromptStorage implementation
    based on the type of DocumentStore provided.

    Supported backends:
    - ElasticsearchDocumentStore → ElasticsearchPromptStorage
    - QdrantDocumentStore → QdrantPromptStorage
    - InMemoryDocumentStore → InMemoryPromptStorage

    Example:
        prompts_ds = store_manager.get_store("prompts")
        prompt_storage = PromptStorageFactory.create(prompts_ds)
        prompt_storage.create_prompt("my_prompt", "content...")
    """

    @staticmethod
    def create(document_store: Optional[DocumentStore] = None) -> PromptStorage:
        """
        Create appropriate PromptStorage based on DocumentStore type.

        Args:
            document_store: Haystack DocumentStore instance

        Returns:
            PromptStorage implementation

        Raises:
            NotImplementedError: If document_store type is not supported
        """

        document_store = document_store if document_store else DocumentStoreProvider().get_store("prompts")

        if isinstance(document_store, ElasticsearchDocumentStore):
            return ElasticsearchPromptStorage(document_store)

        elif isinstance(document_store, InMemoryDocumentStore):
            return InMemoryPromptStorage(document_store)

        elif isinstance(document_store, QdrantDocumentStore):
            return QdrantPromptStorage(document_store)

        else:
            raise NotImplementedError(
                f"PromptStorage not implemented for {type(document_store).__name__}. "
                f"Supported: ElasticsearchDocumentStore, QdrantDocumentStore, InMemoryDocumentStore"
            )


class TemplateStorageFactory:
    """
    Factory to create the appropriate TemplateStorage implementation
    based on the type of DocumentStore provided.

    Supported backends:
    - ElasticsearchDocumentStore → ElasticsearchTemplateStorage
    - QdrantDocumentStore → QdrantTemplateStorage
    - InMemoryDocumentStore → InMemoryTemplateStorage

    Example:
        templates_ds = store_manager.get_store("templates")
        template_storage = TemplateStorageFactory.create(templates_ds)
        template_storage.create_template("my_template", "content...")
    """

    @staticmethod
    def create(document_store: DocumentStore | None = None) -> TemplateStorage:
        """
        Create appropriate TemplateStorage based on DocumentStore type.

        Args:
            document_store: Haystack DocumentStore instance

        Returns:
            TemplateStorage implementation

        Raises:
            NotImplementedError: If document_store type is not supported
        """
        document_store = document_store if document_store else DocumentStoreProvider().get_store("templates")
        if isinstance(document_store, ElasticsearchDocumentStore):
            return ElasticsearchTemplateStorage(document_store)

        elif isinstance(document_store, InMemoryDocumentStore):
            return InMemoryTemplateStorage(document_store)

        elif isinstance(document_store, QdrantDocumentStore):
            return QdrantTemplateStorage(document_store)

        else:
            raise NotImplementedError(
                f"TemplateStorage not implemented for {type(document_store).__name__}. "
                f"Supported: ElasticsearchDocumentStore, QdrantDocumentStore, InMemoryDocumentStore"
            )
