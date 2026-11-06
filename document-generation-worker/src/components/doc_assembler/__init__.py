"""document assembly components and utilities."""

from src.components.doc_assembler.citation_generators import (
    CitationExtractor,
    ExtractionStrategy,
    ExtractionStrategyFactory,
    MultiParagraphStrategy,
    SingleParagraphStrategy,
)

__all__ = [
    # Extraction
    "CitationExtractor",
    # Strategies
    "ExtractionStrategy",
    "ExtractionStrategyFactory",
    "SingleParagraphStrategy",
    "MultiParagraphStrategy",
]
