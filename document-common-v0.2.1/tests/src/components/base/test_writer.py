from unittest.mock import MagicMock, patch

import pytest

from document_common.components.base.writer import Writer
from haystack.document_stores.types import DuplicatePolicy
from haystack import Document


class TestWriter:
    def test_default_policy(self):
        mock_ds = MagicMock()
        writer = Writer(document_store=mock_ds)
        assert writer.document_writer is not None

    def test_custom_policy(self):
        mock_ds = MagicMock()
        writer = Writer(document_store=mock_ds, policy=DuplicatePolicy.OVERWRITE)
        assert writer.document_writer is not None

    def test_run_writes_documents(self):
        mock_ds = MagicMock()
        mock_ds.to_dict.return_value = {"type": "in_memory"}

        writer = Writer(document_store=mock_ds)

        docs = [
            Document(content="doc1", meta={"id": "1"}),
            Document(content="doc2", meta={"id": "2"}),
        ]
        result = writer.run(documents=docs, batch_size=10)

        assert "doc_ids" in result
        assert len(result["doc_ids"]) == 2
        assert result["documents"] == docs

    def test_run_batched_writes(self):
        mock_ds = MagicMock()
        mock_ds.to_dict.return_value = {"type": "in_memory"}

        writer = Writer(document_store=mock_ds)

        docs = [
            Document(content="doc1", meta={"id": "1"}),
            Document(content="doc2", meta={"id": "2"}),
            Document(content="doc3", meta={"id": "3"}),
        ]
        result = writer.run(documents=docs, batch_size=2)

        assert len(result["doc_ids"]) == 3

    def test_run_empty_documents(self):
        mock_ds = MagicMock()
        writer = Writer(document_store=mock_ds)
        result = writer.run(documents=[], batch_size=10)
        assert result["doc_ids"] == []
        assert result["documents"] == []
