import importlib.resources as pkg_resources
from pathlib import Path
from typing import Any, Dict, List

from chonkie import BaseChunker, RecursiveChunker, SemanticChunker, SentenceChunker
from haystack import Document, component
from pydantic import ValidationError
from tokenizers import Tokenizer

from config import config
from document_common.components.base.embedder import EmbedderFactory, TEIChonkieEmbeddings
from document_common.schemas import Chunk, File
from document_common.utils.logger import Logger


class ChunkerFactory:
    """
    Factory that creates the configured chunker (semantic, sentence, or recursive).

    Reads chunking configuration from config and instantiates the appropriate
    chonkie chunker with the specified parameters.

    Usage:
        factory = ChunkerFactory()
        chunker = factory.get_chunker()
        chunks = chunker("Some long text")

    Expected config keys in `chunking`:
        - type: "semantic", "sentence", or "recursive"
        - embedding_model (semantic only)
        - chunking type-specific params
    """

    def __init__(self):
        # P2 : Allow to choose between in memory chunking or from a chunking api. For now only in-memory.
        chunking: Dict[str, Any] = dict(config.get_current_config().chunking)
        chunker_type = chunking.pop("type", None)
        recipe_path = Path("config/configs/chunking/recipe")

        if chunker_type not in {"semantic", "sentence", "recursive"}:
            raise ValueError(
                f"In your chunker configuration, the type is set to {chunker_type} which is not allowed."
            )

        params: Dict[str, Any] = dict(chunking.get(chunker_type, {}))

        if chunker_type == "semantic":
            embedding_client = TEIChonkieEmbeddings(EmbedderFactory().get_embedder())
            self.chunker = SemanticChunker(embedding_model=embedding_client, **params)

        if chunker_type == "sentence":
            self.chunker = SentenceChunker(**params)

        elif chunker_type == "recursive":
            # we support three ways to configure recursive chunker:
            # 1. recipe name (+ optional lang) via from_recipe
            #   allows the use of chonkie's hub for common recipes
            if "recipe" in params:
                recipe = params.pop("recipe")
                lang = params.pop("lang", None)
                if lang:
                    self.chunker = RecursiveChunker.from_recipe(
                        recipe, lang=lang, **params
                    )
                else:
                    self.chunker = RecursiveChunker.from_recipe(recipe, **params)

            # 2. local recipe path via from_recipe
            elif "recipe_path" in params:
                recipe_path = Path(params.pop("recipe_path"))
                tokenizer_path = params.pop("tokenizer_or_token_counter")

                # Resolve tokenizer path: if it's a relative path starting with "hub/",
                # resolve it relative to the document_common package data
                if tokenizer_path.startswith("hub/"):
                    tokenizer_content = pkg_resources.files("document_common").joinpath(tokenizer_path)
                    tokenizer_file_path = Path(tokenizer_content)
                else:
                    tokenizer_file_path = Path(tokenizer_path)

                self.chunker = RecursiveChunker.from_recipe(
                    tokenizer=Tokenizer.from_file(str(tokenizer_file_path)),
                    path=str(recipe_path.resolve()),
                    **params,
                )

            # 3. inline rules passed directly
            else:
                if hasattr(params, "rules"):
                    # expect params to include "rules" or rely on defaults
                    self.chunker = RecursiveChunker(**params)
                else:
                    # defaults
                    self.chunker = RecursiveChunker(**params)

        else:
            raise ValueError(f"Unsupported chunker type: {chunker_type}")

    def get_chunker(self) -> BaseChunker:
        return self.chunker


@component
class DocumentChunker:
    """
    Component for chunking `Document` objects using the configured chunker.

    It processes text content and returns smaller Document chunks while keeping
    the original metadata of the chunked Document.
    """

    def __init__(self):
        self.chunker = ChunkerFactory().get_chunker()

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Chunk the content of each Document and return new chunked Documents.

        Args:
            documents: Input documents to be chunked.

        Returns:
            dict: Dictionary with a `"documents"` key containing chunked Documents.
        """
        new_docs = []
        for doc in documents:
            chunks = self.chunker(doc.content or "")
            new_docs += [Document(content=chunk.text, meta=doc.meta) for chunk in chunks]  # type: ignore

        return {"documents": new_docs}


@component
class FileChunker:
    """
    Component for chunking file-based documents.

    It expects that each Document has file-level metadata (validated via the `File` model).
    Each output chunk will include metadata modeled by the `Chunk` object, linked to
    the file ID and preserving chunk order.
    """

    def __init__(self):
        self.chunker = ChunkerFactory().get_chunker()
        self.logger = Logger().get_logger("file_chunker")

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Chunk documents that are associated with files.

        It validates the file-level metadata and then chunks the document content,
        producing new Document instances that are our "original" chunks containing
        a file_id to reference the file it originates from and a parent_id set to
        None.

        Args:
            documents (List[Document]): List of documents to chunk.

        Returns:
            dict: Dictionary with a `"documents"` key containing chunked Documents.

        Raises:
            ValueError: If any input document fails file metadata validation.
        """
        new_docs: List[Document] = []
        for doc in documents:
            try:
                File.model_validate(doc.meta)  # We check if it is a File Document
            except ValidationError as e:
                message = f"Validation error while FileChunking of {doc} : {e}"
                self.logger.error(message)
                raise ValueError(message)

            chunks = self.chunker(doc.content or "")
            for i, chunk in enumerate(chunks):
                # Extract source metadata from the file document's metadata
                file_meta = File.model_validate(doc.meta)
                chunk_obj = Chunk(
                    type="original",
                    file_id=doc.id,
                    order=i,
                    core_data_id=file_meta.core_data_id,
                    core_type=file_meta.core_type,
                    core_data_source=file_meta.core_data_source,
                )
                new_docs.append(
                    Document(
                        content=chunk.text,  # type: ignore
                        meta=chunk_obj.model_dump(),
                    )
                )

        return {"documents": new_docs}
