import copy
from typing import Callable, List, Union

from haystack import Document, component
from document_common.components.base import StructuredAssembly
from document_common.components.base.batched_operations import batch_function_calls
from document_common.utils.logger import Logger

from src.components.preprocessing.document_factories.models import (
    DocumentTemplate,
)


@component
class DocumentFactory:
    """
    Factory component to generate new documents (chunks) from existing ones using an LLM.

    This class takes a list of input `Document` objects, applies a structured LLM prompt to each,
    and produces new `Document` objects based on the model's structured reply.

    Attributes:
        doc_type (str): Type assigned to the newly generated documents.
        generator (Structuredassembly): Internal generator that handles the LLM prompt execution.
    """

    def __init__(
        self,
        doc_type: str,
        prompt_name: str,
        base_model: type[DocumentTemplate],
        fixed_inputs: dict[str, str] = {},
    ):
        self.doc_type = doc_type
        self.generator = StructuredAssembly(prompt_name=prompt_name, base_model=base_model, fixed_inputs=fixed_inputs)
        self.logger = Logger().get_logger(f"{self.__class__.__name__}")

    def _process_document(self, doc: Document) -> List[str]:
        """Process a single document using the LLM generator and return generated text content."""
        llm_answer = self.generator.run(inputs={"document": doc})
        object_generated: DocumentTemplate = llm_answer["structured_reply"]  # type: ignore

        to_docs = object_generated.to_doc_text()
        if isinstance(to_docs, List):
            return to_docs
        else:
            return [to_docs]

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict[str, List[Document]]:
        """
        Generates new documents from input documents.

        For each input document:
            1. Runs the LLM.
            2. Extracts the structured reply.
            3. Converts the structured reply into one or more new Document objects.
            4. Assigns parent Document metadata

        Args:
            documents (List[Document]): Documents to process.

        Returns:
            dict: {"documents": List[Document]} - the newly created documents.
        """
        # Process all documents using batch processing
        all_generated_content = batch_function_calls( # TOASK : ça revient ordonné ?
            self._process_document,
            [[doc] for doc in documents],
            [None for _ in documents],
            batch_size=16,
        )

        # Create child documents from generated content and handle potential exceptions
        new_docs: list[Document] = []
        for idx, (content_result, doc) in enumerate(zip(all_generated_content, documents)):
            if isinstance(content_result, Exception):
                self.logger.error(f"Failed to process document {idx}: {content_result}")
                raise content_result

            # content_result is a list of generated text content
            for content_text in content_result:
                new_docs.append(self._create_child_doc(content_text, doc, self.doc_type))

        return {"documents": new_docs}

    def _create_child_doc(self, content: str, parent_doc: Document, doc_type: str, parenting: bool = True):
        """Helper method to create a new Document derived from a parent.

        Preserves source metadata (core_data_id, core_type, core_data_source) from parent
        to enable traceability back to the original file across all document assemblies.
        """
        new_doc = Document(content=content, meta=copy.deepcopy(parent_doc.meta))
        new_doc.meta["type"] = doc_type
        if parenting:
            new_doc.meta["parent_id"] = parent_doc.id

        return new_doc


@component
class ManualDocumentFactory(DocumentFactory):
    """
    Generates new documents from existing ones using a manual processing function instead of an LLM.

    This is useful when you want to create child documents using alternative methods,
    such as entity extraction with SpaCy or custom rules.
    """

    def __init__(self, doc_type: str, process_func: Callable[[str], Union[str, List[str]]]):
        """
        Initializes the ManualDocumentFactory.

        Args:
            doc_type: Type to assign to generated documents.
            process_func: Function to generate new content from input document text.
        """
        self.doc_type = doc_type
        self.process_func = process_func

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict[str, list[Document]]:
        new_docs: list[Document] = []
        for doc in documents:
            to_docs = self.process_func(doc.content or "")
            if isinstance(to_docs, List):
                temp_docs = []

                for doc_text in to_docs:
                    temp_docs.append(self._create_child_doc(doc_text, doc, self.doc_type))

                new_docs += temp_docs

            else:
                new_docs.append(self._create_child_doc(to_docs, doc, self.doc_type))

        return {"documents": new_docs}
