from abc import ABC, abstractmethod
from typing import List, Union

from pydantic import BaseModel


class DocumentTemplate(ABC, BaseModel):
    """
    Abstract base class for structured representations of generated document content.

    In many pipelines, large documents are split into smaller "chunks" using an LLM
    or another generator. However, the structure of generated objects can vary widely
    (e.g., queries, key-value pairs, summaries). To ensure consistent behavior,
    every document template must define a `to_doc_text` method, which converts
    the structured object into one or more text strings suitable for creating
    new `Document` instances.

    Subclasses define the schema of the generated object and specify how it
    should be transformed into textual content for downstream use.
    """

    @abstractmethod
    def to_doc_text(self) -> Union[str, List[str]]:
        """
        Convert the structured object into one or more strings
        that will form the content of new documents (chunks).

        Returns:
            Union[str, List[str]]: The text content for one or multiple
            child documents derived from this template.
        """
        pass


# ---------- Example : --------------
class Hqueries(DocumentTemplate):
    """
    Example template for extracting queries from a document.

    The LLM is expected to populate the `queries` field with a list of strings.
    Each extracted query will then become the content of a new child document.
    """

    queries: List[str]  # The llm will fill this field

    def to_doc_text(
        self,
    ):  # This function describes the way new documents are created, here each query extracted will be a new 'child' of the original text
        return self.queries