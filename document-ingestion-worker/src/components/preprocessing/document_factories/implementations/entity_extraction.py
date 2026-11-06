from typing import List

from haystack import Document, component
from document_common.components.base.batched_operations import batch_function_calls
from document_common.components.base.deduplicators import EntityDeduplicator
from document_common.schemas.entities import Entity
from document_common.utils.logger import Logger

from src.components.preprocessing.document_factories.base import (
    DocumentFactory,
)
from src.components.preprocessing.document_factories.models import DocumentTemplate
from src.components.preprocessing.document_factories.registry import register_factory


class EntityExtraction(DocumentTemplate):
    entities: List[Entity]

    def to_doc_text(self):
        return [entity.model_dump_json() for entity in self.entities]

@component
class EntitiesExtractionDocumentFactory(DocumentFactory):
    """Specialized factory for entity extraction with deduplication."""

    def __init__(
        self,
        doc_type: str,
        prompt_name: str,
        fixed_inputs: dict[str, str] = {},
    ):
        DocumentFactory.__init__(self, doc_type, prompt_name, EntityExtraction, fixed_inputs)
        self.base_model = EntityExtraction
        self.logger = Logger().get_logger(self.__class__.__name__)

    def _extract_entities_from_doc(self, doc: Document) -> List[Entity]:
        """Extract entities from a single document using the LLM generator."""
        llm_answer = self.generator.run(inputs={"document": doc})
        object_generated: EntityExtraction = llm_answer["structured_reply"]  # type: ignore
        return object_generated.entities

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict[str, List[Document]]:
        """Extract entities from multiple documents with deduplication.
        
        Args:
            documents: List of documents to process
            
        Returns:
            dict with "documents" key containing extracted entities as documents
        """
        new_docs: List[Document] = []
        if not documents:
            return {"documents": new_docs}
            
        reference_doc = documents[0]

        # Extract entities from all documents using batch processing
        all_entities = batch_function_calls(
            self._extract_entities_from_doc,
            [[doc] for doc in documents],
            [None for _ in documents],
            batch_size=64,
        )

        # Flatten all entity lists and handle potential exceptions
        entities: list[Entity] = []
        for entity in all_entities:
            if isinstance(entity, Exception):
                self.logger.warning(f"Failed to extract entities from document: {entity}")
                raise entity
            entities.extend(entity)

        # Deduplicate the entities at document level
        refined_entities = EntityDeduplicator().deduplicate_with_combined_scoring(entities)

        # And return them as documents
        for entity in refined_entities:
            doc = self._create_child_doc(entity.model_dump_json(), reference_doc, self.doc_type, parenting=False)
            new_docs.append(doc)

        return {"documents": new_docs}


@register_factory("entity_extraction")
def make_entity_extraction_factory():
    """Create an Entity Extraction document factory.
    
    Returns:
        EntitiesExtractionDocumentFactory: Factory for extracting entities from documents
    """
    return EntitiesExtractionDocumentFactory(
        doc_type="entity_extraction",
        prompt_name="entity_extraction",
        fixed_inputs={},
    )

