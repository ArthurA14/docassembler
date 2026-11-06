from typing import List, Optional

from haystack import Document, component
from haystack.components.writers.document_writer import DocumentWriter
from haystack.document_stores.types import DocumentStore, DuplicatePolicy


@component
class Writer:
    """
    A Haystack component that writes documents to the document store.

    This is a wrapper around Haystack's `DocumentWriter`, designed to provide future flexibility
    for adding checks, validations, and policy enforcement during the document writing phase.

    Documents are written in batches to the store, and their IDs are returned as output.
    The batch size can be configured via the `batch_size` parameter in the `run` method.
    If a duplicate document is encountered, the behavior is controlled by the specified `DuplicatePolicy`.

    Args:
        document_store (DocumentStore): The document store where documents will be written.
        policy (Optional[DuplicatePolicy]): Strategy for handling duplicate documents. Defaults to `DuplicatePolicy.SKIP`.

    Haystack Inputs:
        - documents (List[Document]): A list of documents to write to the store.
        - batch_size (int, optional): Number of documents to write per batch. Defaults to 25.

    Haystack Outputs:
        - doc_ids (List[str]): List of IDs of successfully written documents.
        - documents (List[Document]): The list of documents that were written.
    """

    def __init__(self, document_store: DocumentStore, policy: Optional[DuplicatePolicy] = None):
        if not policy:
            policy = DuplicatePolicy.SKIP

        self.document_writer = DocumentWriter(document_store=document_store, policy=policy)

    @component.output_types(doc_ids=List[str], documents=List[Document])
    def run(self, documents: List[Document], batch_size: int = 25) -> dict[str, list[str|Document]]:
        # for doc in documents:
        #     if "type" not in doc.meta:
        #         print("Warning : The type of the doc wasn't specified, set to default : 'original'")
        #         doc.meta["type"] = "original"
        #     # if "project_name" in doc.meta:
        #     #     if doc.meta["project_name"] != config.get_current_config().database.project_name:
        #     #         print("Warning : A document already had a project_name set different from the current project_name, by default we will replace it by the name of the current project")
        #     # if "version" in doc.meta:
        #     #     if doc.meta["version"] != config.get_current_config().database.version:
        #     #         print("Warning : A document already had a version set different from the current version, by default we will replace it by the current version")

        #     doc.meta["project_name"] = config.get_current_config().database.project_name
        #     doc.meta["version"] = config.get_current_config().database.version
        all_doc_ids: List[str] = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self.document_writer.run(documents=batch)
            all_doc_ids.extend(doc.id for doc in batch)

        return {"doc_ids": all_doc_ids, "documents": documents}
