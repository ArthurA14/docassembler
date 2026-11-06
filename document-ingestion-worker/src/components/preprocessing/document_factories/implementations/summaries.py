from src.components.preprocessing.document_factories.base import DocumentFactory
from src.components.preprocessing.document_factories.models import DocumentTemplate
from src.components.preprocessing.document_factories.registry import register_factory


class Summary(DocumentTemplate):
    """Template for summary extraction."""
    summary: str

    def to_doc_text(self):
        return self.summary

@register_factory("summaries")
def make_summaries_factory():
    """Create a Summary document factory.
    
    Returns:
        DocumentFactory: Factory for extracting summaries from documents
    """
    return DocumentFactory(
        doc_type="summaries",
        prompt_name="extract_summary",
        base_model=Summary,
        fixed_inputs={},
    )
