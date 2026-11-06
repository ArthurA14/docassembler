from typing import List, cast

from haystack import Document, component

from src.components.preprocessing.document_factories.base import DocumentFactory
from src.components.preprocessing.document_factories.models import DocumentTemplate
from src.components.preprocessing.document_factories.registry import register_factory


class Summary(DocumentTemplate):
    """Template for summary extraction."""
    summary: str

    def to_doc_text(self):
        return self.summary

@component  
class RefinedSummaryFactory(DocumentFactory):
    """Factory for creating refined summaries by aggregating individual summaries."""

    def __init__(
        self,
        doc_type: str,
        prompt_name: str,
        fixed_inputs: dict[str, str] = {},
    ):
        DocumentFactory.__init__(self, doc_type, prompt_name, Summary, fixed_inputs)
        self.summary_factory = DocumentFactory(
            doc_type="summaries",
            prompt_name="extract_summary",
            base_model=Summary,
            fixed_inputs={},
        )

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict[str, List[Document]]:
        """Create refined summaries by first extracting summaries, then aggregating them.
        
        Args:
            documents: List of documents to process
            
        Returns:
            dict with "documents" key containing individual summaries and refined summary
        """
        if not documents:
            return {"documents": []}
            
        # First, extract summaries from all documents
        summaries_docs = self.summary_factory.run(documents=documents)["documents"]

        if not summaries_docs:
            return {"documents": []}

        reference_doc = documents[0]

        # Then create a refined summary from all the summaries
        llm_answer = self.generator.run(inputs={"documents": summaries_docs})
        refined_summary: Summary = cast(Summary, llm_answer["structured_reply"])

        # Create document for the refined summary
        refined_doc = self._create_child_doc(
            refined_summary.to_doc_text(), 
            reference_doc, 
            self.doc_type, 
            parenting=False
        )

        # Return all summaries including the refined one
        summaries_docs.append(refined_doc)
        return {"documents": summaries_docs}


@register_factory("refined_summary", doc_types=["summaries", "refined_summary"])
def make_refined_summaries_factory():
    """Create a Refined Summary document factory.
    
    Returns:
        RefinedSummaryFactory: Factory for creating refined summaries from multiple documents
    """
    return RefinedSummaryFactory(
        doc_type="refined_summary",
        prompt_name="aggregate_summaries",
    )