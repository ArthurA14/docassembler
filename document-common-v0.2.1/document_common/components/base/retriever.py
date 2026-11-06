"""
This module defines Haystack pipeline components for document retrieval, filtering, and duplication checks
within a semantic document processing workflow using a vector-based DocumentStore.

Components included:

1. **DocIdsRetriever**:
    Retrieves documents from the document store based on a list of document IDs.
    - Input: List of document IDs (`List[str]`)
    - Output: List of `Document` objects

2. **Retriever**:
    Retrieves the top-k most similar documents (based on dense embedding similarity), optionally filtered by document type.
    - Input: Dense embedding vector (`List[float]`)
    - Output: List of matched `Document` objects

3. **DuplicateChecker**:
    Checks whether documents with a given `doc_id` already have associated sub-documents (e.g. summaries) of a specific type.
    Filters out those that already have such children.
    - Input: List of document IDs (`List[str]`)
    - Output: Filtered list of document IDs (`List[str]`)

4. **ParentFinder**:
    Given a list of child documents, retrieves their corresponding parent documents by accessing the `meta.parent_id` field.
    - Input: List of `Document` objects
    - Output: List of parent `Document` objects

All components use a `DocumentStore` backend (such as Qdrant) and are compatible with Haystack's pipeline system.

Dependencies:
- Haystack
- A custom logger from `src.utils.logger`
- A custom `UniversalEmbeddingRetriever` from `.document_store`
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from haystack import Document, component
from haystack.document_stores.types import DocumentStore

from document_common.components.base.embedder import TextEmbedder
from document_common.components.base.reranker import DocumentReranker
from document_common.schemas import FileStatus
from document_common.utils.logger import Logger

from .document_store import (
    UniversalEmbeddingRetriever,
)


@component
class DocIdsRetriever:
    """
    A Haystack component that retrieves documents from the document store by their IDs.

    Given a list of document IDs (`doc_ids`), this component fetches the corresponding `Document` objects
    using the document store's filtering mechanism.

    Args:
        document_store (DocumentStore): The document store to query

    Input:
        doc_ids (List[str]): List of document IDs to retrieve.

    Output:
        documents (List[Document]): List of retrieved `Document` objects.
    """

    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store

    @component.output_types(documents=List[Document])
    def run(self, doc_ids: List[str]) -> dict[str, list[Document]]:
        if doc_ids == []:
            return {"documents": []}

        filters = [{"field": "id", "operator": "==", "value": id} for id in doc_ids]
        docs = self.document_store.filter_documents(filters={"operator": "OR", "conditions": filters})
        return {"documents": docs}


@component
class Retriever:
    """
    A Haystack component that retrieves documents similar to a given dense embedding,
    returning the top-K most relevant documents based on vector similarity.

    Args:
        document_store: The document store to search in.
        doc_type: If set, restricts retrieval to documents with this meta.type.
        top_k: Number of top results to return. If set to -1, retrieves all documents.
        with_rerank: Whether to apply reranking after initial retrieval.

    Input:
        query: Dense vector (list[float]), raw query string, or None for unfiltered retrieval.
        top_k: Overrides the instance-level `top_k` if provided.
        doc_type: Overrides the instance-level `doc_type` if provided.

    Output:
        documents: List of top-matching `Document` objects.
    """

    def __init__(
        self,
        document_store: DocumentStore,
        doc_type: Optional[str] = None,
        top_k: int = 10,
        with_rerank: bool = True,
    ):
        """
        Initialize the Retriever component.

        Args:
            document_store: The document store to search in.
            doc_type: Default document type filter.
            top_k: Default number of results to return. -1 retrieves all matching documents.
            with_rerank: Whether to apply reranking for string queries.

        Note:
            If top_k is set to -1, all documents matching the filter will be retrieved.
            Reranking is automatically disabled when top_k is -1 to avoid performance issues.
        """
        self.logger = Logger().get_logger("retriever.dense")
        self.document_store = document_store
        self.doc_type = doc_type
        self.top_k = top_k
        self.with_rerank = with_rerank
        self.reranker = DocumentReranker() if with_rerank else None

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: Optional[list[float] | str] = None,
        top_k: Optional[int] = None,
        doc_type: Optional[str] = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents based on the query.

        Args:
            query: Either a dense embedding (list[float]), a raw query string, or None.
            top_k: Number of results to return. Overrides instance default if provided.
            doc_type: Document type filter. Overrides instance default if provided.

        Returns:
            Dictionary with 'documents' key containing the retrieved Document objects.

        Raises:
            ValueError: If query is provided but is neither a list nor a string.
        """
        # Resolve parameters with proper precedence (instance defaults overridden by method args)
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_doc_type = doc_type if doc_type is not None else self.doc_type

        # Build filters
        filters: dict[str, Any] = {}
        if effective_doc_type:
            filters = {"field": "meta.type", "operator": "==", "value": effective_doc_type}

        # Process query to get embedding
        embedding: Optional[list[float]] = None
        is_string_query = isinstance(query, str)

        if is_string_query:
            try:
                embedder = TextEmbedder()
                result = embedder.run(text=query, mode="query")
                embedding = result.get("embedding")
                if embedding is None:
                    self.logger.error("TextEmbedder returned no embedding for query")
                    return {"documents": []}
            except Exception as e:
                self.logger.error(f"Failed to embed query: {e}")
                return {"documents": []}
        elif isinstance(query, list):
            embedding = query

        # Determine if reranking should be applied
        should_rerank = self._should_rerank(
            with_rerank=self.with_rerank,
            top_k=effective_top_k,
            is_string_query=is_string_query,
        )

        # Log retrieval parameters
        self.logger.info(
            f"Retrieving documents with top_k={effective_top_k}, " +
            f"doc_type={effective_doc_type}, rerank={should_rerank}"
        )

        # Calculate effective top_k for retrieval (increase if reranking will reduce results)
        retrieval_top_k = effective_top_k
        if should_rerank and effective_top_k != -1:
            retrieval_top_k = effective_top_k * 3
            self.logger.debug(f"Increasing retrieval top_k to {retrieval_top_k} for reranking")

        # Perform initial retrieval
        try:
            results = UniversalEmbeddingRetriever().run(
                document_store=self.document_store,
                filters=filters,
                top_k=retrieval_top_k,
                dense_embedding=embedding,
            )
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return {"documents": []}

        # Apply reranking if enabled
        if should_rerank and is_string_query and self.reranker is not None:
            try:
                rerank_results = self.reranker.run(
                    query=query,  # type: ignore
                    documents=results["documents"],
                    top_k_output=effective_top_k,
                )
                results = rerank_results
            except Exception as e:
                self.logger.warning(f"Reranking failed: {e}. Returning initial retrieval results.")
                # Results remain as initial retrieval

        # Validate result count (only when top_k is not -1)
        if effective_top_k != -1:
            retrieved_count = len(results["documents"])
            if retrieved_count != effective_top_k:
                self.logger.warning(
                    f"Retrieval returned {retrieved_count} documents, expected {effective_top_k}"
                )

        return results

    def _should_rerank(
        self, with_rerank: bool, top_k: int, is_string_query: bool
    ) -> bool:
        """
        Determine whether reranking should be applied.

        Args:
            with_rerank: Whether reranking is enabled on the instance.
            top_k: The number of results to retrieve.
            is_string_query: Whether the query is a string.

        Returns:
            True if reranking should be applied, False otherwise.
        """
        if not with_rerank:
            return False
        if top_k == -1:
            self.logger.warning(
                "Retriever is set to retrieve all documents (-1), reranking is disabled."
            )
            return False
        if not is_string_query:
            self.logger.debug(
                "Reranking skipped: query is not a string (embedding provided directly)"
            )
            return False
        return True


@component
class DuplicateChecker:
    """
    A Haystack component that filters out document IDs that already have child documents of a given type.

    For each input document ID, this component checks whether a sub-document (child) of a specified type
    already exists (e.g., summaries, analyses, etc.). If such a child exists, the ID is excluded from output.

    Args:
        document_store (DocumentStore): The document store to query.
        doc_type (str): The type of sub-document to check for (must match `meta.type` field).

    Input:
        doc_ids (List[str]): List of parent document IDs to check for duplicates.

    Output:
        doc_ids (List[str]): Filtered list of document IDs that do not yet have a sub-document of the given type.
    """

    def __init__(self, document_store: DocumentStore, doc_types: List[str]):
        self.logger = Logger().get_logger("DuplicateChecker")
        self.document_store = document_store

        self.doc_types = doc_types

    @component.output_types(doc_ids=List[str])
    def run(self, doc_ids: List[str]):
        filters = [
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.parent_id", "operator": "==", "value": doc_id},
                    {
                        "operator": "OR",
                        "conditions": [
                            {"field": "meta.type", "operator": "==", "value": dt} for dt in self.doc_types
                        ]
                    }
                ],
            }
            for doc_id in doc_ids
        ]
        existing_docs = self.document_store.filter_documents(filters={"operator": "OR", "conditions": filters})
        existing_ids = set(doc.meta["parent_id"] for doc in existing_docs)
        kept_doc_ids = [doc_id for doc_id in doc_ids if doc_id not in existing_ids]
        self.logger.debug(f"kept_doc_ids : {kept_doc_ids}")
        self.logger.debug(f"existing_ids : {existing_ids}")

        self.logger.debug(f"Number of documents kept/Number of original documents : {len(kept_doc_ids)}/{len(doc_ids)}")
        return {"doc_ids": kept_doc_ids}


# Recupère les parents d'un document
@component
class ParentFinder:
    """
    A Haystack component that retrieves the parent documents for a list of child documents.

    This component uses the `meta.parent_id` field of each input `Document` to find and return their
    corresponding parent documents from the document store.

    Args:
        document_store (DocumentStore): The document store to query.

    Input:
        documents (List[Document]): List of child `Document` objects, each containing a `meta.parent_id`.

    Output:
        documents (List[Document]): List of parent `Document` objects.
    """
    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        conditions = [
            {"field": "id", "operator": "==", "value": doc.meta.get("parent_id")} 
            for doc in documents
            if doc.meta.get("parent_id")
        ]
        
        parent_docs : List[Document] = []
        if conditions:
            parent_docs = self.document_store.filter_documents(
                filters={
                    "operator": "OR", 
                    "conditions": conditions
                }
            )

        return {"documents": parent_docs}

class ChildFinder:
    def __init__(self, document_store: DocumentStore, doc_type: str):
        self.document_store = document_store
        self.doc_type = doc_type

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        conditions = [
            {"field": "meta.parent_id", "operator": "==", "value": doc.id} 
            for doc in documents
        ]

        filters = {
            "operator": "AND",
            "conditions": [
                {
                    "operator": "OR",
                    "conditions": conditions
                },
                {
                    "field": "meta.doc_type",
                    "operator": "==",
                    "value": self.doc_type,
                },
            ],
        }

        # Query the document store
        child_docs = self.document_store.filter_documents(filters=filters)

        return {"documents": child_docs}
    
    # def run(self, documents: List[Document]):
    #     parent_ids = [doc.id for doc in documents]
    #     all_candidates = self.document_store.filter_documents(
    #         filters={"field": "meta.doc_type", "operator": "==", "value": self.doc_type}
    #     )

    #     child_docs = [
    #         d for d in all_candidates
    #         if any(pid in d.meta.get("parent_id", []) for pid in parent_ids)
    #     ]

    #     return {"documents": child_docs}

@component
class DocumentVersionFilter:
    """
    A Haystack component that filters documents to keep only one version
    per content ID based on a specified selection strategy.
    """

    def __init__(
        self,
        content_id_field: str = "content_id",
        version_field: str = "created_at",
        selection_strategy: str = "latest",
        custom_selector: Optional[Callable[[List[Document]], Document]] = None,
    ):
        """
        Initialize the DocumentVersionFilter.

        Args:
            content_id_field: Field name that contains the content identifier/hash
            version_field: Field name used for version comparison (e.g., datetime field)
            selection_strategy: Strategy for selecting which version to keep:
                - "latest": Keep the document with the highest version_field value
                - "earliest": Keep the document with the lowest version_field value
                - "custom": Use custom_selector function
            custom_selector: Custom function that takes a list of documents and returns one
        """
        self.logger = Logger().get_logger("retriever.version_filter")
        self.content_id_field = content_id_field
        self.version_field = version_field
        self.selection_strategy = selection_strategy
        self.custom_selector = custom_selector

        if selection_strategy == "custom" and custom_selector is None:
            raise ValueError("custom_selector must be provided when selection_strategy is 'custom'")

    @component.output_types(documents=List[Document])
    def run(self, document_store: DocumentStore, filters: Optional[Dict[str, Any]] = None) -> Dict[str, List[Document]]:
        """
        Filter documents to keep only one version per content ID.

        Args:
            documents: List of documents to filter
            filters: Optional additional filters on base documents
        Returns:
            Dictionary with 'documents' key containing filtered documents
        """
        if not document_store:
            self.logger.error("No document store selected.")
            return {"documents": []}

        condition = {
            "field": "meta.status",
            "operator": "==",
            "value": FileStatus.importED,
        }

        if filters:
            condition = {
                "operator": "AND",
                "conditions": [condition, filters],
            }

        self.logger.debug(f"Getting files with filters:\n\t{condition}")

        documents = document_store.filter_documents(filters=condition)

        # Group documents by content_id
        content_groups = self._group_by_content_id(documents)

        # Select one document per group
        selected_documents: list[Document] = []
        for _, doc_list in content_groups.items():
            selected_doc = self._select_document_from_group(doc_list)
            selected_documents.append(selected_doc)

        self.logger.info(
            f"Filtered {len(documents)} documents down to {len(selected_documents)} unique content versions"
        )

        return {"documents": selected_documents}

    def _group_by_content_id(self, documents: List[Document]) -> dict[str, list[Document]]:
        """Group documents by their content ID."""
        groups: dict[str, list[Document]] = {}

        for doc in documents:
            content_id = doc.meta.get(self.content_id_field)
            if content_id is None:
                self.logger.warning(f"Document {doc.id} missing {self.content_id_field} field, skipping")
                continue

            if content_id not in groups:
                groups[content_id] = []
            groups[content_id].append(doc)

        return groups

    def _select_document_from_group(self, documents: List[Document]) -> Document:
        """Select one document from a group based on the selection strategy."""
        if len(documents) == 1:
            return documents[0]

        if self.selection_strategy == "custom" and self.custom_selector is not None:
            return self.custom_selector(documents)

        elif self.selection_strategy == "latest":
            return self._select_by_version(documents, reverse=True)

        elif self.selection_strategy == "earliest":
            return self._select_by_version(documents, reverse=False)

        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    def _select_by_version(self, documents: List[Document], reverse: bool) -> Document:
        """Select document based on version field value."""
        try:
            sorted_docs = sorted(documents, key=lambda doc: self._get_version_value(doc), reverse=reverse)
            return sorted_docs[0]
        except Exception as e:
            self.logger.warning(f"Error sorting documents by {self.version_field}: {e}. Returning first document.")
            return documents[0]

    def _get_version_value(self, document: Document):
        """Extract version value from document for comparison."""
        version_value = document.meta.get(self.version_field)

        if version_value is None:
            raise ValueError(f"Document {document.id} missing {self.version_field} field")

        # Handle string datetime conversion if needed
        if isinstance(version_value, str):
            try:
                return datetime.fromisoformat(version_value.replace("Z", "+00:00"))
            except ValueError:
                # If not a datetime string, return as-is for comparison
                return version_value

        return version_value
