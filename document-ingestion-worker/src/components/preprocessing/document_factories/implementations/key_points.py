from typing import List

from src.components.preprocessing.document_factories.base import DocumentFactory
from src.components.preprocessing.document_factories.models import DocumentTemplate
from src.components.preprocessing.document_factories.registry import register_factory


class KeyPoints(DocumentTemplate):
    """Template for key points extraction."""
    key_points: List[str]

    def to_doc_text(self):
        return self.key_points


@register_factory("key_points", doc_types=["key_points"])
def make_key_points_factory():
    """Create a KeyPoints document factory.
    
    Returns:
        DocumentFactory: Factory for extracting key points from documents
    """
    return DocumentFactory(
        doc_type="key_points",
        prompt_name="extract_key_points",
        base_model=KeyPoints,
        fixed_inputs={},
    )
