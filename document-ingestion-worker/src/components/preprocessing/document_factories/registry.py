from typing import Any, Callable, Dict, List, Optional

from haystack import Document, component

from src.components.preprocessing.document_factories.base import DocumentFactory

# Registry mapping factory names to factory functions
FactoryRegistry: Dict[str, Callable[..., DocumentFactory]] = {}

# Registry mapping factory names to document types
DocumentTypeRegistry: Dict[str, List[str]] = {}

def register_factory(
    name: str,
    doc_types: Optional[List[str]] = None
) -> Callable[[Callable[..., DocumentFactory]], Callable[..., DocumentFactory]]:
    """Register a document factory with optional document types.

    Args:
        name: Name of the factory to register
        doc_types: Optional list of document types this factory produces.
                  If None, uses the factory name as the single document type.

    Returns:
        Decorator function for factory registration

    Raises:
        ValueError: If document types are duplicated across factories
    """
    def decorator(fn: Callable[..., DocumentFactory]) -> Callable[..., DocumentFactory]:
        # Determine document types - default to factory name if not specified
        resolved_doc_types = doc_types if doc_types is not None else [name]

        # Register the factory
        FactoryRegistry[name] = fn

        DocumentTypeRegistry[name] = resolved_doc_types

        return fn

    return decorator

def create_document_factory(name: str, *args: Any, **kwargs: Any) -> DocumentFactory:
    """Create a factory instance by name.

    Args:
        name: Name of the factory to create
        *args: Positional arguments to pass to the factory function
        **kwargs: Keyword arguments to pass to the factory function

    Returns:
        DocumentFactory: An instance of the requested factory

    Raises:
        ValueError: If no factory is registered with the given name
    """
    if name not in FactoryRegistry:
        raise ValueError(f"No factory registered for '{name}'")
    factory_fn = FactoryRegistry[name]
    instance = factory_fn(*args, **kwargs)
    return DocumentFactoryValidator(factory=instance, factory_name=name) # We wrap the factory in a custom object that will check if the documents generated match the declared types.

def get_factory_for_document_type(doc_type: str) -> List[str]:
    """Get the factory name that produces a specific document type.

    Args:
        doc_type: The document type to look up

    Returns:
        The name of the factory that produces this document type

    Raises:
        ValueError: If no factory is registered for this document type
    """
    factories: List[str] = []
    for fact, doc_types in DocumentTypeRegistry.items():
        if doc_type in doc_types:
            factories.append(fact)
    return factories

def get_document_types_for_factory(factory_name: str) -> List[str]:
    """Get all document types produced by a specific factory.

    Args:
        factory_name: The name of the factory to look up

    Returns:
        List of document types produced by this factory

    Raises:
        ValueError: If no factory is registered with this name
    """
    if factory_name not in FactoryRegistry:
        raise ValueError(f"No factory registered with name '{factory_name}'")
    if factory_name not in DocumentTypeRegistry:
        raise ValueError(f"No document types with name '{factory_name}', which should never happen.")

    # Find all document types that map to this factory
    return DocumentTypeRegistry[factory_name]


@component
class DocumentFactoryValidator(DocumentFactory):
    def __init__(self, factory: DocumentFactory, factory_name: str):
        self.factory_name = factory_name
        self.factory = factory
        self.doc_types = get_document_types_for_factory(factory_name)
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        new_docs = self.factory.run(documents=documents)

        for doc in new_docs["documents"]:
            doc_meta = doc.meta['type']
            if doc_meta not in self.doc_types:
                raise ValueError(f"The factory {self.factory_name} generated a doc with doc_type {doc_meta} which was not declared by the factory at registration. Declared types : {self.doc_types}")
        
        return new_docs