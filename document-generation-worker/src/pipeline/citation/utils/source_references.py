"""
Source references utility for building source lookup data.

Provides utility functions for extracting source references from citations.
"""

from typing import Dict, List

from document_common.schemas.citations import Citation, SourceMetadata, SourceReferences
from document_common.utils.logger import Logger


def build_source_references(
    citations: List[Citation], source_documents: List[SourceMetadata]
) -> SourceReferences:
    """
    Build SourceReferences from citations and available source documents.

    Creates a SourceReferences object containing all unique source documents
    that are referenced by the citations. This enables API consumers to look up
    source documents by source_id without making separate requests.

    Args:
        citations: List of Citation objects (may be empty)
        source_documents: All available source documents from database

    Returns:
        SourceReferences containing only the sources referenced in citations
    """
    logger = Logger().get_logger("source_references")

    # Collect all unique source_ids from citations
    cited_source_ids = set()
    for citation in citations:
        cited_source_ids.add(citation.chunk_id)

    # Build references dict with only cited sources
    references_dict: Dict[str, SourceMetadata] = {}
    source_by_id = {doc.chunk_id: doc for doc in source_documents}

    for source_id in cited_source_ids:
        if source_id in source_by_id:
            references_dict[source_id] = source_by_id[source_id]
        else:
            logger.warning(f"Citation references source_id '{source_id}' but source not found in documents")

    return SourceReferences(references=references_dict)
