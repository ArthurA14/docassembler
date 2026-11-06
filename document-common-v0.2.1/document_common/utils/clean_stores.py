
from haystack.document_stores.errors import MissingDocumentError
from haystack.document_stores.types import DocumentStore

from document_common.components.base.document_store import DocumentStoreProvider
from document_common.utils.logger import Logger

logger = Logger().get_logger("utils.clean_stores")

def empty_document_store(document_store: DocumentStore):
    docs = document_store.filter_documents()
    if docs:
        try :
            document_store.delete_documents(document_ids=[doc.id for doc in docs])
        except MissingDocumentError as e:
            logger.warning(f"While trying to empty the document store {document_store.to_dict()}, a MissingDocumentError was raised : {e}")

def flush_all():
    """
    Flush all document stores managed by the system.

    Notes:
        - Deletes all documents, sections, files, and generated document chunks.
        - Resets the system to a clean state.
        - Keeps the templates and prompts
    """
    docstoreprovider = DocumentStoreProvider()
    for store in ["sections", "documents", "files", "documents"]:
        document_store = docstoreprovider.get_store(store)
        empty_document_store(document_store)
