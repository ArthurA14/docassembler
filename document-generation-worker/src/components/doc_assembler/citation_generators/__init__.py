# Main public API
from .citation_extractor import CitationExtractor

# Strategy components
from .extraction_strategies import (
    ExtractionStrategy,
    ExtractionStrategyFactory,
    MultiParagraphStrategy,
    SingleParagraphStrategy,
    TimeoutException,
)

__all__ = [
    "CitationExtractor",
    "ExtractionStrategy",
    "SingleParagraphStrategy",
    "MultiParagraphStrategy",
    "ExtractionStrategyFactory",
    "TimeoutException",
]
