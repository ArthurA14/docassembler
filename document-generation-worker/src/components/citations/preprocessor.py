import re
import unicodedata
from typing import Dict, List

from haystack import component
from pydantic import BaseModel
from document_common.utils.logger import Logger

from config import config


class Sentence(BaseModel):
    """Sentence with character-level offsets"""

    text: str
    normalized_text: str
    start_index: int
    end_index: int
    length: int


@component
class TextPreprocessor:
    """
    Haystack component for text preprocessing with regex-based sentence segmentation.

    Splits text into sentences using simple regex pattern matching on sentence boundaries.
    No chunking or tokenization - just straightforward sentence splitting.
    """

    def __init__(self, min_sentence_length: int = 20):
        self.logger = Logger().get_logger("text_preprocessor")
        citation_config = config.get_current_config().citations
        self.min_sentence_length = min_sentence_length or citation_config.preprocessing.get("min_sentence_length", 20)
        self.logger.debug(f"TextPreprocessor initialized (min_sentence_length: {self.min_sentence_length})")

    @component.output_types(sentences=List[Sentence])
    def run(self, text: str) -> Dict[str, List[Sentence]]:
        """
        Segment text into sentences with character offsets using regex.

        Args:
            text: Generated text to split into sentences

        Returns:
            Dict with 'sentences' key containing list of Sentence objects
        """
        # Step 1: Regex-based sentence segmentation
        sentence_texts = self._split_sentences(text)

        # Step 2: Compute character offsets and create Sentence objects
        sentences = []
        current_pos = 0

        for sent_text in sentence_texts:
            # Find sentence in original text
            start_idx = text.find(sent_text, current_pos)
            if start_idx == -1:
                self.logger.debug(f"Skipped sentence: could not find in original text at position {current_pos}")
                continue  # Skip if not found (shouldn't happen)

            end_idx = start_idx + len(sent_text)

            # Step 3: Unicode normalization
            normalized = unicodedata.normalize("NFKD", sent_text)

            # Step 4: Length check
            if len(normalized.strip()) < self.min_sentence_length:
                self.logger.debug(
                    f"Skipped short sentence (length {len(normalized.strip())} < {self.min_sentence_length}): "
                    f"{normalized.strip()[:50]}..."
                )
                current_pos = end_idx
                continue

            sentences.append(
                Sentence(
                    text=sent_text,
                    normalized_text=normalized,
                    start_index=start_idx,
                    end_index=end_idx,
                    length=len(sent_text),
                )
            )

            current_pos = end_idx

        self.logger.info(f"Preprocessed {len(sentences)} valid sentences from {len(sentence_texts)} total segments")
        return {"sentences": sentences}

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex pattern matching.

        Uses lookbehind for sentence-ending punctuation (. ! ?) followed by whitespace.
        This is a simple but effective approach for most text.

        Args:
            text: Text to split

        Returns:
            List of sentence strings
        """
        # Sentence boundary detection: split after .!? followed by whitespace
        pattern = r"(?<=[.!?])\s+"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
