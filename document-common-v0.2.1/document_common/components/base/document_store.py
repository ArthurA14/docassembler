from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch
from haystack import component
from haystack.components.retrievers.in_memory.embedding_retriever import (
    InMemoryEmbeddingRetriever,
)
from haystack.dataclasses.document import Document
from haystack.document_stores.in_memory.document_store import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore
from haystack.document_stores.types.policy import DuplicatePolicy
from haystack_integrations.components.retrievers.elasticsearch import (
    ElasticsearchEmbeddingRetriever,
)
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from config import config
from document_common.utils.config.manager import ConfigManager
from document_common.utils.logger import Logger


@component
class UniversalEmbeddingRetriever:
    """
    A unified embedding retriever interface that routes retrieval to the correct
    backend-specific retriever based on the type of `DocumentStore` provided.

    This abstraction allows seamless support for multiple document store implementations
    (e.g., Elasticsearch, InMemory, documentDocumentStore), each with their own
    retriever logic and optimization.

    Use this when you want embedding-based retrieval but do not want to manually manage
    which retriever to call depending on the document store backend.

    Supported stores:
    - `ElasticsearchDocumentStore` → uses `ElasticsearchEmbeddingRetriever`
    - `InMemoryDocumentStore` → uses `InMemoryEmbeddingRetriever`
    - `documentDocumentStore` → delegates back to `.retrieve()`

    The limit is Haystack compatible stores so in the future we could easily implement : `Azure`, `MongoDB`, `Neo4j`, `OpenSearch`, `Qdrant`, ...
    """

    def __init__(self):
        self.logger = Logger().get_logger("retriever.embedding")

    @component.output_types(documents=List[Document])
    def run(
        self,
        document_store: DocumentStore,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        dense_embedding: Optional[List[float]] = None,
    ) -> dict[str, List[Document]]:
        if not dense_embedding:  # if no dense embedding is given we return all the documents ("LIMIT")
            self.logger.debug(f"No dense embedding. Filtering documents with filter:\n{filters}")
            docs = document_store.filter_documents(filters=filters)

            if top_k != -1:
                docs = docs[:top_k]

            return {"documents": docs}

        results = {}
        try:
            if isinstance(document_store, ElasticsearchDocumentStore):
                self.logger.debug(f"Filters on ESDS: {filters}")
                retriever = ElasticsearchEmbeddingRetriever(document_store=document_store, filters=filters, top_k=top_k)
                results = retriever.run(query_embedding=dense_embedding)

            elif isinstance(document_store, QdrantDocumentStore):
                self.logger.debug(f"Filters on QdrantDS: {filters}")
                retriever = QdrantEmbeddingRetriever(document_store=document_store, filters=filters, top_k=top_k)
                results = retriever.run(query_embedding=dense_embedding)

            elif isinstance(document_store, InMemoryDocumentStore):
                self.logger.debug(f"Filters on InMemoryDS: {filters}")
                retriever = InMemoryEmbeddingRetriever(document_store=document_store, filters=filters, top_k=top_k)
                results = retriever.run(query_embedding=dense_embedding)

            elif isinstance(document_store, DocumentDocumentStore):
                self.logger.debug(
                    f"Filters on documentDS: {filters} \
                    Forwarding filtering to {document_store.document_store.to_dict().get('type')}"
                )
                results = document_store.retrieve(filters=filters, top_k=top_k, dense_embedding=dense_embedding)
            else:
                raise NotImplementedError(
                    "The document store set with DocumentStoreProvider is not implemented yet for retrieving."
                )

            num_docs = len(results['documents'])
            self.logger.debug(f"Retrieved {num_docs} results")
            return results

        except Exception:
            self.logger.error("Error while Retrieving", exc_info=True)
            raise


class ESDStoreManager:
    """Simplified manager for ElasticsearchDocumentStore with create-if-not-exists behavior"""

    def __init__(self, config: ConfigManager):
        self.logger = Logger().get_logger("document_store.es_store_manager")
        self.config = config.get_current_config().database
        self.document_stores: Dict[str, ElasticsearchDocumentStore] = {}

        # Initialize Elasticsearch client for index operations
        self.es_client = Elasticsearch(hosts=[self.config.url] if isinstance(self.config.url, str) else self.config.url)

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test Elasticsearch connection"""
        try:
            health = self.es_client.cluster.health()
            self.logger.info(f"Connected to Elasticsearch. Status: {health['status']}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Elasticsearch at {self.config.url}: {e}")
            raise

    def index_exists(self, index_name: str) -> bool:
        """Check if an Elasticsearch index exists"""
        try:
            return self.es_client.indices.exists(index=index_name)
        except Exception as e:
            self.logger.error(f"Error checking if index '{index_name}' exists: {e}")
            return False

    def create_index_if_not_exists(self, index_name: str, store_config: Dict[str, Any]):
        """Create index if it doesn't exist"""
        if not self.index_exists(index_name):
            body = {}
            if "settings" in store_config:
                body["settings"] = store_config["settings"]
            if "mappings" in store_config:
                body["mappings"] = store_config["mappings"]

            try:
                self.es_client.indices.create(index=index_name, body=body)
                self.logger.info(f"Created index '{index_name}' with custom mappings")
            except Exception as e:
                self.logger.error(f"Failed to create index '{index_name}': {e}")
                raise
        else:
            self.logger.info(f"Index '{index_name}' already exists")

    def create_document_store(self, store_key: str, require_config_match: bool = True) -> ElasticsearchDocumentStore:
        """Create ElasticsearchDocumentStore with automatic index creation"""
        if store_key in self.document_stores:
            return self.document_stores[store_key]

        # Get store configuration
        store_config = self.config.indexes.get(store_key)
        if not store_config:
            if require_config_match:
                raise ValueError(f"Document store configuration not found: {store_key}")
            self.logger.info(
                f"No configuration found for '{store_key}'. Creating Elasticsearch index using the key as name."
            )
            store_config = {}

        index_name = store_config.get("name", store_key)

        # Create index if it doesn't exist
        # self.create_index_if_not_exists(index_name, store_config)

        # Create Haystack document store
        document_store = ElasticsearchDocumentStore(
            hosts=[self.config.url] if isinstance(self.config.url, str) else self.config.url,
            index=index_name,
            # custom_mapping=store_config.get('mappings'),
        )

        self.document_stores[store_key] = document_store
        self.logger.info(f"Created document store '{store_key}' with index '{index_name}'")

        return document_store

    def get_store(self, store_key: str, require_config_match: bool = True) -> ElasticsearchDocumentStore:
        """Get document store - creates if not cached"""
        store = self.document_stores.get(store_key)
        if store:
            return store
        else:
            # Try to create it if not found
            try:
                return self.create_document_store(store_key, require_config_match=require_config_match)
            except ValueError:
                raise ValueError(f"No document store called {store_key} is configured.")

    def initialize_stores(self):
        """Initialize all configured document stores"""
        self.logger.info(f"Initializing {len(self.config.indexes)} document stores...")

        for store_key in self.config.indexes:
            self.create_document_store(store_key)
            self.logger.info(f"Initialized document store: {store_key}")


class QdrantStoreManager:
    """Manager for QdrantDocumentStore with create-if-not-exists behavior"""

    def __init__(self, config: ConfigManager):
        self.logger = Logger().get_logger("document_store.qdrant_store_manager")
        self.config = config.get_current_config().database
        self.document_stores: Dict[str, QdrantDocumentStore] = {}

        # Test connection if using remote Qdrant
        self._test_connection()

    def _test_connection(self):
        """Test Qdrant connection"""
        try:
            # Create a test connection to verify accessibility
            if self.config.url and self.config.url != ":memory:":
                test_store = QdrantDocumentStore(
                    url=self.config.url, index="__test__", embedding_dim=128, recreate_index=False
                )
                self.logger.info(f"Connected to Qdrant at {self.config.url}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant at {self.config.url}: {e}")
            raise

    def create_document_store(self, store_key: str, require_config_match: bool = True) -> QdrantDocumentStore:
        """Create QdrantDocumentStore with automatic collection creation"""
        if store_key in self.document_stores:
            return self.document_stores[store_key]

        # Get store configuration
        store_config = self.config.indexes.get(store_key)
        if not store_config:
            if require_config_match:
                raise ValueError(f"Document store configuration not found: {store_key}")
            self.logger.info(
                f"No configuration found for '{store_key}'. Creating Qdrant collection using the key as name."
            )
            store_config = {}

        index_name = store_config.get("name", store_key)
        embedding_dim = store_config.get("embedding_dim", 768)  # Default to 768

        # Build optional configurations
        kwargs = {
            "url": self.config.url,
            "index": index_name,
            "embedding_dim": embedding_dim,
            "recreate_index": store_config.get("recreate_index", False),
        }

        # Add HNSW config if specified
        if "hnsw_config" in store_config:
            kwargs["hnsw_config"] = store_config["hnsw_config"]

        # Create Haystack document store
        document_store = QdrantDocumentStore(**kwargs)

        self.document_stores[store_key] = document_store
        self.logger.info(f"Created Qdrant document store '{store_key}' with collection '{index_name}'")

        return document_store

    def get_store(self, store_key: str, require_config_match: bool = True) -> QdrantDocumentStore:
        """Get document store - creates if not cached"""
        store = self.document_stores.get(store_key)
        if store:
            return store
        else:
            try:
                return self.create_document_store(store_key, require_config_match=require_config_match)
            except ValueError:
                raise ValueError(f"No document store called {store_key} is configured.")

    def initialize_stores(self):
        """Initialize all configured document stores"""
        self.logger.info(f"Initializing {len(self.config.indexes)} Qdrant document stores...")

        for store_key in self.config.indexes:
            self.create_document_store(store_key)
            self.logger.info(f"Initialized Qdrant document store: {store_key}")


class DocumentDocumentStore(DocumentStore):
    """
    A wrapper around an existing `DocumentStore` that automatically applies contextual filters
    (such as file IDs, project name, or version) to all document retrieval operations.

    This is designed for use in **document assembly workflows**, where users specify
    which subset of documents (e.g. files 1, 2, and 3) should be considered for a document.

    Once initialized, all retrieval, filtering, and search operations will transparently apply
    the configured filters without requiring the calling code to manage them.

    This ensures consistent scoping of data across the system while keeping the main application
    logic clean and unaware of filtering specifics.

    Args:
        document_store (DocumentStore): The underlying document store to wrap.
        file_ids (Optional[List[str]]): List of file IDs to filter documents by.
        project_name (Optional[str]): Project name to filter by (matches `meta.project_name`).
        version (Optional[str]): Version string to filter by (matches `meta.version`).

    Example:
        store = documentDocumentStore(
            document_store=MyDocumentStore(),
            file_ids=["file1", "file2"],
            project_name="MyProject",
            version="v1.0"
        )
        docs = store.filter_documents()  # Automatically applies filters
    """

    def __init__(
        self,
        document_store: DocumentStore,
        file_ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        project_name: Optional[str] = None,
        version: Optional[str] = None,
    ):
        self.logger = Logger().get_logger("document_store.document")
        self.document_store = document_store
        self.file_ids = file_ids
        conditions = []
        if file_ids:
            file_filter = {
                'operator': 'OR', 'conditions': [
                    {'field': 'meta.core_data_id', 'operator': '==', 'value': id} for id in file_ids
                ]
            }
            conditions.append(file_filter)
        if doc_ids:
            doc_filter = {
                'operator': 'OR', 'conditions': [
                    {'field': 'id', 'operator': '==', 'value': id} for id in doc_ids
                ]
            }
            conditions.append(doc_filter)

        # Not implemented yet: TODO ?
        # if project_name:
        #     conditions.append({"field": "project_name", "operator": "==", "value": project_name})
        # if version:
        #     conditions.append({"field": "version", "operator": "==", "value": version})

        self.filters = {"operator": "AND", "conditions": conditions} if conditions else None

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        if self.filters and filters:
            docs = self.document_store.filter_documents(
                filters={"operator": "AND", "conditions": [self.filters, filters]}
            )
        elif self.filters:
            docs = self.document_store.filter_documents(filters=self.filters)
        else:
            docs = self.document_store.filter_documents(filters=filters)

        return docs

    def write_documents(
        self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:  # TODO add version and project name to docs before write
        return self.document_store.write_documents(documents=documents, policy=policy)

    def delete_documents(self, document_ids: List[str]) -> None:
        self.document_store.delete_documents(document_ids=document_ids)

    def count_documents(self) -> int:
        return self.document_store.count_documents()

    def get_store(self):
        return self.document_store

    def update_store(self, document_store: DocumentStore):
        self.document_store = document_store

    def to_dict(self) -> Dict[str, Any]:
        return self.document_store.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentStore":
        return cls.from_dict(data=data)

    def retrieve(
        self,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        dense_embedding: Optional[List[float]] = None,
    ) -> dict[str, List[Document]]:
        if self.filters and filters:
            # Extract conditions from both filters
            all_conditions = []

            # Handle self.filters - if it's an AND filter with conditions, extract them; otherwise, wrap as single condition
            if (
                isinstance(self.filters, dict)
                and self.filters.get("operator") == "AND"
                and "conditions" in self.filters
            ):
                all_conditions.extend(self.filters["conditions"])
            else:
                all_conditions.append(self.filters)

            # Handle passed filters - if it's an AND filter with conditions, extract them; otherwise, wrap as single condition
            if isinstance(filters, dict) and filters.get("operator") == "AND" and "conditions" in filters:
                all_conditions.extend(filters["conditions"])
            else:
                all_conditions.append(filters)

            # Combine all conditions under a single AND operator
            built_filters = {"operator": "AND", "conditions": all_conditions}
        elif self.filters:
            built_filters = self.filters
        else:
            built_filters = filters

        return UniversalEmbeddingRetriever().run(
            self.document_store, filters=built_filters, top_k=top_k, dense_embedding=dense_embedding
        )


class DocumentStoreProvider:
    """
    Centralized provider that exposes document stores based on the configured backend.

    Args:
        config_manager: Optional configuration manager override.
        require_config_match: When True, requests for unknown store keys raise an error instead
            of creating a new in-memory store using the key as the collection name.
    """

    def __init__(self, config_manager: Optional[ConfigManager] = None, require_config_match: bool = True):
        self.logger = Logger().get_logger("document_store.provider")
        self.config_manager = config_manager or config
        self.db_config = self.config_manager.get_current_config().database
        self.backend = getattr(self.db_config, "database", None)
        self.require_config_match = require_config_match
        self._store_manager: Optional[ESDStoreManager | QdrantStoreManager] = None
        self._inmemory_stores: Dict[str, InMemoryDocumentStore] = {}

        self._initialize_backend()

    def _initialize_backend(self):
        backend = (self.backend or "").lower()

        if backend == "elastic":
            self._store_manager = ESDStoreManager(self.config_manager)
        elif backend == "qdrant":
            self._store_manager = QdrantStoreManager(self.config_manager)
        elif backend == "inmemory":
            self.logger.info("Using in-memory document stores.")
        else:
            message = (
                "No document store backend defined. "
                "Choices are: \n- qdrant (url required)\n- elastic (url required)\n - inmemory"
            )
            self.logger.error(message)
            raise ValueError(message)

    def get_store(self, store_key: str, require_config_match: Optional[bool] = None) -> DocumentStore:
        require_config_match = (
            self.require_config_match if require_config_match is None else require_config_match
        )

        if self._store_manager:
            return self._store_manager.get_store(store_key, require_config_match=require_config_match)

        if self.backend == "inmemory":
            return self._get_inmemory_store(store_key, require_config_match=require_config_match)

        raise ValueError(f"No document store backend initialized for '{self.backend}'.")

    def _get_inmemory_store(self, store_key: str, require_config_match: bool = True) -> InMemoryDocumentStore:
        if store_key in self._inmemory_stores:
            return self._inmemory_stores[store_key]

        index_name = self._resolve_index_name(store_key)
        if index_name is None:
            if self.require_config_match:
                raise ValueError(f"Document store configuration not found: {store_key}")
            self.logger.info(
                f"No configuration found for '{store_key}'. Creating in-memory store using the key as index name."
            )
            index_name = store_key

        store = InMemoryDocumentStore(index=index_name)
        self._inmemory_stores[store_key] = store
        self.logger.info(f"Created in-memory document store '{store_key}' with index '{index_name}'")
        return store

    def _resolve_index_name(self, store_key: str) -> Optional[str]:
        indexes = getattr(self.db_config, "indexes", None)
        store_config = None

        if indexes:
            if isinstance(indexes, dict):
                store_config = indexes.get(store_key)
            if store_config and isinstance(store_config, dict):
                name = store_config.get("name")
                if name:
                    return name

        return None
