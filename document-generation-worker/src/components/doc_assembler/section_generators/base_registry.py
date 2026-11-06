from typing import Any, Dict, List, Optional, TypedDict, Union

from haystack import component
from haystack.document_stores.types import DocumentStore
from document_common.components.writers.document_writer import DocumentWriter
from document_common.schemas import Document, Section, SectionStatus


class DocumentRequirements(TypedDict):
    """TypedDict for document requirements"""
    critical: list[str]
    optional: list[str]


@component
class SectionGenerator:
    """
    Base class for generating document sections using different strategies.

    This class implements a registry pattern to allow dynamic instantiation of
    subclasses based on a string code. Subclasses must implement the `run()` method
    to generate section content.

    Registry Usage:
        - Subclasses can register themselves with a unique code using the `@SectionGenerator.register(code)` decorator.
        - Use `SectionGenerator.from_code(code, **kwargs)` to create instances dynamically.

    Attributes:
        section (Section): Section object that this generator will populate.
        document (Optional[document]): The document object associated with this section, required if `document_writer` is provided.
        document_writer (Optional[documentWriter]): Used to update the document with new section content.
        context (Optional[Dict[str, Any]]): Optional context inputs to pass to the assembly method.
        method_parameters (Optional[Dict[str, Any]]): Optional parameters specific to the assembly method.

    Methods:
        run(sections): Subclasses must implement this to generate content for the section.
        update_document_section(): If a `document_writer` and `document` are provided, updates the section in the document.
    """

    _registry = {}
    _critical_documents = {}  # Maps generator codes to lists of required document types
    _optional_documents = {}  # Maps generator codes to lists of optional document types

    @classmethod
    def register(cls, code: str, critical_documents: Optional[List[str]] = None, optional_documents: Optional[List[str]] = None):
        def decorator(subclass: type["SectionGenerator"]) -> type["SectionGenerator"]:
            cls._registry[code] = subclass
            cls._critical_documents[code] = critical_documents or []
            cls._optional_documents[code] = optional_documents or []
            return subclass

        return decorator

    @classmethod
    def from_code(cls, code: str, **kwargs: Any) -> "SectionGenerator":
        if code not in cls._registry:
            raise ValueError(f"Unknown SectionGenerator code: '{code}'")
        return cls._registry[code](**kwargs)

    @classmethod
    def get_document_requirements(cls, code: str) -> DocumentRequirements:
        """
        Get the document requirements for a registered section generator.

        Args:
            code: The code of the registered section generator

        Returns:
            A DocumentRequirements TypedDict with 'critical' and 'optional' keys containing lists of document types

        Raises:
            ValueError: If the code is not registered
        """
        if code not in cls._registry:
            raise ValueError(f"Unknown SectionGenerator code: '{code}'")

        return {
            'critical': cls._critical_documents.get(code, []),
            'optional': cls._optional_documents.get(code, [])
        }

    def __init__(
        self,
        prompt_name: str,
        section: Section,
        document_store: DocumentStore,
        document: Optional[Document] = None,
        document_writer: Optional[DocumentWriter] = None,
        context: Optional[Dict[str, Any]] = None,
        method_parameters: Optional[Dict[str, Any]] = None,
    ):
        if document_writer and not document:
            raise ValueError("When setting a document_writer, document must be provided.")
        self.section = section
        self.document = document
        self.document_writer = document_writer
        self.context = context
        self.method_parameters = method_parameters
        self.document_store = document_store

    def run(
        self, sections: Union[List[Section], Section] = []
    ) -> Union[
        dict[str, List[Section]], dict[str, Section]
    ]:  # TODO: c'est moche, ça renvoie une Liste d'une seul section mais c'est ce qui marche avec ListJoiner
        raise NotImplementedError("Subclasses must implement the run() method.")

    def update_document_section(self):
        if self.document_writer and self.document:
            self.document_writer.update_document_section(section=self.section)

    def update_all_sections_status(self, status: SectionStatus, section: Section) -> Section:
        section.status = status
        if section.sections:
            section.sections = [self.update_all_sections_status(status=status, section=s) for s in section.sections]
        
        return section
