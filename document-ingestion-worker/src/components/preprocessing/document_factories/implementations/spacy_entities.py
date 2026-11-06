from typing import List, Union

from src.components.preprocessing.document_factories.base import ManualDocumentFactory
from src.components.preprocessing.document_factories.registry import register_factory


@register_factory("spacy_entities")
def make_spacy_entities_factory():
    """Create a spaCy entity extraction document factory.
    
    Returns:
        ManualDocumentFactory: Factory for extracting named entities using spaCy
    """
    import spacy
    
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError("spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
    
    def extract_entities(text: str) -> Union[str, List[str]]:
        """Extract named entities from text using spaCy.
        
        Args:
            text: Input text to process
            
        Returns:
            List[str]: List of extracted entity names
        """
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities
    
    return ManualDocumentFactory(
        doc_type="spacy_entities",
        process_func=extract_entities
    )
