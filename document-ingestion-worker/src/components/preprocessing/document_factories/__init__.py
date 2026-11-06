# src/components/preprocessing/document_factories/__init__.py

# Import base classes and registry system
from .base import DocumentFactory, ManualDocumentFactory

# Import implementations
from .implementations import (
    make_entity_extraction_factory,
    make_key_points_factory,
    make_refined_summaries_factory,
    make_spacy_entities_factory,
    make_summaries_factory,
)
from .registry import (
    FactoryRegistry,
    create_document_factory,
    get_document_types_for_factory,
    register_factory,
)

__all__ = [
    # Base classes
    "DocumentFactory", 
    "ManualDocumentFactory", 
    
    # Factory functions
    "make_key_points_factory",
    "make_summaries_factory",
    "make_spacy_entities_factory", 
    "make_entity_extraction_factory",
    "make_refined_summaries_factory",
    
    # Registry system
    "register_factory", 
    "create_document_factory", 
    "FactoryRegistry",
    "get_document_types_for_factory"
]
