from unittest.mock import MagicMock, patch

import pytest

from document_common.components.base.document_store import (
    DocumentStoreProvider,
    DocumentDocumentStore,
    UniversalEmbeddingRetriever,
)
from haystack.document_stores.in_memory.document_store import InMemoryDocumentStore
from haystack import Document


class TestUniversalEmbeddingRetriever:
    def test_no_dense_embedding_returns_filtered(self, mock_document_store):
        mock_doc = MagicMock(spec=Document)
        mock_document_store.filter_documents.return_value = [mock_doc, mock_doc]

        retriever = UniversalEmbeddingRetriever()
        result = retriever.run(document_store=mock_document_store, top_k=10, filters={"field": "id", "operator": "==", "value": "test"})

        assert len(result["documents"]) == 2

    def test_no_dense_embedding_with_top_k(self, mock_document_store):
        mock_docs = [MagicMock(spec=Document) for _ in range(5)]
        mock_document_store.filter_documents.return_value = mock_docs

        retriever = UniversalEmbeddingRetriever()
        result = retriever.run(document_store=mock_document_store, top_k=3)

        assert len(result["documents"]) == 3

    def test_no_dense_embedding_all_documents(self, mock_document_store):
        mock_docs = [MagicMock(spec=Document) for _ in range(5)]
        mock_document_store.filter_documents.return_value = mock_docs

        retriever = UniversalEmbeddingRetriever()
        result = retriever.run(document_store=mock_document_store, top_k=-1)

        assert len(result["documents"]) == 5

    def test_with_inmemory_document_store(self):
        ds = InMemoryDocumentStore()
        ds.write_documents([Document(content="test", meta={"id": "1"}, embedding=[0.1, 0.2])])

        retriever = UniversalEmbeddingRetriever()
        result = retriever.run(
            document_store=ds,
            top_k=10,
            dense_embedding=[0.1, 0.2],
        )
        assert isinstance(result["documents"], list)

    def test_unsupported_document_store_raises(self):
        unsupported = MagicMock()
        retriever = UniversalEmbeddingRetriever()
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            retriever.run(document_store=unsupported, top_k=10, dense_embedding=[0.1, 0.2])


class TestDocumentDocumentStore:
    def test_init_with_file_ids(self):
        inner_ds = MagicMock(spec=InMemoryDocumentStore)
        store = DocumentDocumentStore(document_store=inner_ds, file_ids=["file1", "file2"])
        assert store.file_ids == ["file1", "file2"]
        assert store.filters is not None
        assert "operator" in store.filters
        assert store.filters["operator"] == "AND"

    def test_init_with_doc_ids(self):
        inner_ds = MagicMock(spec=InMemoryDocumentStore)
        store = DocumentDocumentStore(document_store=inner_ds, doc_ids=["doc1"])
        assert store.file_ids is None
        assert store.filters is not None

    def test_init_no_filters(self):
        inner_ds = MagicMock(spec=InMemoryDocumentStore)
        store = DocumentDocumentStore(document_store=inner_ds)
        assert store.file_ids is None
        assert store.filters is None

    def test_filter_documents_with_both_filters(self):
        inner_ds = MagicMock(spec=InMemoryDocumentStore)
        inner_ds.filter_documents.return_value = []
        store = DocumentDocumentStore(document_store=inner_ds, file_ids=["f1"])
        store.filter_documents(filters={"field": "type", "operator": "==", "value": "doc"})
        inner_ds.filter_documents.assert_called_once()
        call_filter = inner_ds.filter_documents.call_args.kwargs["filters"]
        assert call_filter["operator"] == "AND"

    def test_filter_documents_with_only_store_filters(self):
        inner_ds = MagicMock(spec=InMemoryDocumentStore)
        inner_ds.filter_documents.return_value = []
        store = DocumentDocumentStore(document_store=inner_ds, file_ids=["f1"])
        store.filter_documents()
        inner_ds.filter_documents.assert_called_once()

    def test_filter_documents_no_store_filters(self):
        inner_ds = MagicMock(spec=InMemoryDocumentStore)
        inner_ds.filter_documents.return_value = []
        store = DocumentDocumentStore(document_store=inner_ds)
        store.filter_documents(filters={"field": "x", "operator": "==", "value": "y"})
        inner_ds.filter_documents.assert_called_once()
        call_filter = inner_ds.filter_documents.call_args.kwargs["filters"]
        assert call_filter["field"] == "x"

    def test_write_documents_delegates(self):
        inner_ds = MagicMock(spec=InMemoryDocumentStore)
        inner_ds.write_documents.return_value = 1
        store = DocumentDocumentStore(document_store=inner_ds)
        doc = Document(content="test")
        store.write_documents([doc])
        inner_ds.write_documents.assert_called_once()

    def test_delete_documents_delegates(self):
        inner_ds = MagicMock(spec=InMemoryDocumentStore)
        store = DocumentDocumentStore(document_store=inner_ds)
        store.delete_documents(["id1"])
        inner_ds.delete_documents.assert_called_once_with(document_ids=["id1"])

    def test_count_documents_delegates(self):
        inner_ds = MagicMock(spec=InMemoryDocumentStore)
        inner_ds.count_documents.return_value = 5
        store = DocumentDocumentStore(document_store=inner_ds)
        assert store.count_documents() == 5

    def test_to_dict_delegates(self):
        inner_ds = MagicMock(spec=InMemoryDocumentStore)
        inner_ds.to_dict.return_value = {"type": "in_memory"}
        store = DocumentDocumentStore(document_store=inner_ds)
        assert store.to_dict() == {"type": "in_memory"}

    def test_get_store(self):
        inner_ds = MagicMock(spec=InMemoryDocumentStore)
        store = DocumentDocumentStore(document_store=inner_ds)
        assert store.get_store() is inner_ds

    def test_update_store(self):
        inner_ds = MagicMock(spec=InMemoryDocumentStore)
        new_ds = MagicMock(spec=InMemoryDocumentStore)
        store = DocumentDocumentStore(document_store=inner_ds)
        store.update_store(new_ds)
        assert store.document_store is new_ds

    def test_retrieve_with_filters(self):
        inner_ds = MagicMock(spec=InMemoryDocumentStore)
        inner_ds.filter_documents.return_value = []
        store = DocumentDocumentStore(document_store=inner_ds, file_ids=["f1"])
        store.retrieve(filters={"field": "type", "operator": "==", "value": "doc"}, dense_embedding=[0.1, 0.2])


class TestDocumentStoreProvider:
    def test_inmemory_backend(self, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.database = MagicMock(
            database="inmemory",
            indexes={},
        )
        monkeypatch.setattr("document_common.components.base.document_store.config", mock_config)

        provider = DocumentStoreProvider(require_config_match=False)
        store = provider.get_store("test_store")
        assert isinstance(store, InMemoryDocumentStore)

    def test_inmemory_caches_stores(self, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.database = MagicMock(
            database="inmemory",
            indexes={},
        )
        monkeypatch.setattr("document_common.components.base.document_store.config", mock_config)

        provider = DocumentStoreProvider(require_config_match=False)
        store1 = provider.get_store("cached_store")
        store2 = provider.get_store("cached_store")
        assert store1 is store2

    def test_inmemory_require_config_match_no_config(self, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.database = MagicMock(
            database="inmemory",
            indexes={},
        )
        monkeypatch.setattr("document_common.components.base.document_store.config", mock_config)

        provider = DocumentStoreProvider(require_config_match=True)
        with pytest.raises(ValueError, match="configuration not found"):
            provider.get_store("unknown_store")

    def test_unknown_backend_raises(self, mock_config, monkeypatch):
        mock_config.get_current_config.return_value.database = MagicMock(
            database="unknown_backend",
        )
        monkeypatch.setattr("document_common.components.base.document_store.config", mock_config)

        with pytest.raises(ValueError, match="No document store backend"):
            DocumentStoreProvider(require_config_match=False)
