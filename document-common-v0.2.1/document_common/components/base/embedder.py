from typing import Any, Callable, List, Literal, Optional, TypedDict, Union

import numpy as np
from chonkie import BaseEmbeddings
from haystack import Document, component
from openai import OpenAI

from config import config
from document_common.components.base.batched_operations import batch_function_calls
from document_common.utils.logger import Logger


class EmbeddingResult(TypedDict):
    embeddings: List[List[float]]
    embedding_dim: int

@component
class ApiEmbedder:
    """
    ApiEmbedder is a component that generates dense vector embeddings for text input
    by calling a remote embedding API (usually hosted via Docker).

    Attributes:
        url (str): The API endpoint to send embedding requests to.
        dimension (int): The expected dimension of the returned embeddings.
        model (str): The model name to use in the API request. Defaults to "tei".

    Input:
        - texts (Union[str, List[str]]): Text(s) to be embedded.
        - mode (str): Semantic context for the embedder. One of "query", "document", or "instruct".
        - instruction (Optional[str]): Optional instruction text, used when `mode='instruct'`.

    Output:
        - embeddings (List[List[float]]): A list of dense embedding vectors.
        - embedding_dim (int): The dimensionality of each vector.

    Example:
        embedder = ApiEmbedder(url="http://vllm.<CLUSTER_IP>.nip.io/v1/embeddings", model="Qwen/Qwen3-Embedding-4B", dimension=1024)
        result = embedder.run(["This is a test sentence."])
    """

    def __init__(self, url: str, dimension: Optional[int]=None, model: str = "tei"):
        self.client = OpenAI(base_url=url, api_key="no_api_key")
        self.dimension = dimension
        self.model = model
        self.logger = Logger().get_logger("api_embedder")

    @component.output_types(documents=List[Document])
    def run(
        self,
        texts: Union[str, List[str]],
        mode: Literal["query", "document", "instruct"] = "document",
        instruction: Optional[str] = None,
    ) -> EmbeddingResult:
        if isinstance(texts, str):
            texts = [texts]
        
        batch_results = batch_function_calls(
            self._embed_text, 
            [[text] for text in texts]
        )

        embeddings = []
        for result in batch_results:
            if isinstance(result, Exception):
                raise result

            if isinstance(result, list) and all(isinstance(x, float) for x in result):
                embeddings.append(result)
            else:
                raise TypeError(f"Expected embedding to be a list of floats, got {result}")
        
        if self.dimension:
            dimension = self.dimension
        elif embeddings:
            dimension = len(embeddings[0])
        else:
            dimension = 0
        return {
            "embeddings": embeddings,
            "embedding_dim": dimension
        }

    def _embed_text(self, text: str) -> List[float]:
        if self.dimension:
            response = self.client.embeddings.create(model=self.model, input=text, dimensions=self.dimension)
        else:
            response = self.client.embeddings.create(model=self.model, input=text)
        embedding = response.data[0].embedding
        if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
            raise TypeError(f"Expected embedding to be a list of floats, got {embedding}")
        return embedding


class TEIChonkieEmbeddings(BaseEmbeddings):  # type: ignore[misc]
    """
    Chonkie BaseEmbeddings adapter backed by our TEI service.

    Provides:
      - embed(text) -> np.ndarray
      - embed_batch(texts) -> list[np.ndarray]
      - count_tokens / count_tokens_batch via a pluggable counter (defaults to whitespace heuristic)
    """

    def __init__(
        self,
        tei: ApiEmbedder,
        vector_dimension: Optional[int] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        self._tei = tei
        self._dimension = vector_dimension
        self._token_counter = token_counter or self._default_token_counter

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Resolve lazily by embedding a tiny string once.
            v = self.embed("a")
            self._dimension = len(v)
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        vecs = self._tei._embed_text(text)
        return np.asarray(vecs, dtype=np.float32)

    def count_tokens(self, text: str) -> int:
        return self._token_counter(text)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        return [self._token_counter(t) for t in texts]

    def get_tokenizer_or_token_counter(self):
        # Chonkie accepts either; we return the callable used for counts
        return self._token_counter

    def get_tokenizer(self):
        # Required by BaseEmbeddings abstract class
        return self._token_counter

    @classmethod
    def is_available(cls) -> bool:
        # Available if we can make HTTP calls in any supported path
        return True

    def __repr__(self) -> str:
        return f"TEIChonkieEmbeddings(base_url={self._tei.client.base_url!r}, dim={self._dimension})"

    @staticmethod
    def _default_token_counter(text: str) -> int:
        # Simple, deterministic heuristic compatible with Chonkie’s token-counter extension points.
        # Users can override with an accurate counter if desired.
        return max(1, len(text.strip().split()))


class EmbedderFactory:
    """
    A factory class to create an embedder instance based on the current configuration.

    Depending on the embedder configuration, it can return either:
    - DenseEmbedder (in-memory model), or
    - ApiEmbedder (external API service)

    The configuration is expected to come from the global config system
    and should include a "type" field with one of the following:
        - "inmemory"
        - "api"

    Raises:
        ValueError: If an unknown or unsupported embedder type is specified.

    Example:
        factory = EmbedderFactory()
        embedder = factory.get_embedder()
        embeddings = embedder.run(texts=["example text"])
    """

    def __init__(self, embedder_config: Optional[dict[str, Any]]=None):
        if not embedder_config:
            embedder_config = dict(config.get_current_config().embedder)

        embedder_type = embedder_config.pop("type", None)
        if embedder_type == "api":
            # Required fields for API embedder
            url = embedder_config.pop("url", None)
            if not url:
                raise ValueError("URL is required for API embedder configuration")

            dimension = embedder_config.pop("dimension", None)

            model = embedder_config.pop("model", "tei")

            # Validate types
            if not isinstance(url, str):
                raise TypeError(f"URL must be a string, got {type(url)}")
            if isinstance(dimension, int) and dimension <= 0:
                raise TypeError(f"Dimension must be a positive integer, got {dimension}")
            if not isinstance(model, str):
                raise TypeError(f"Model must be a string, got {type(model)}")

            self.embedder = ApiEmbedder(url=url, dimension=dimension, model=model)
        else:
            raise ValueError(
                f"In your embedder configuration, the type is set to {embedder_type} which is not allowed."
            )

    def get_embedder(self):
        return self.embedder


@component
class DocumentEmbedder:
    """
    Component that enriches documents with dense embeddings.

    It takes a single Document or a list of Documents and fills the `embedding`
    attribute of each with a dense vector representation generated by a `DenseEmbedder`.

    Useful for downstream tasks like clustering, similarity search, or deduplication.

    Input:
        - documents (Union[List[Document], Document])

    Output:
        - documents (List[Document]): Same documents, with `embedding` filled.
    """

    def __init__(self, embedder_config: Optional[dict[str, Any]] = None):
        self.embedder = EmbedderFactory(embedder_config).get_embedder()

    @component.output_types(documents=List[Document])
    def run(
        self, documents: Union[List[Document], Document]
    ) -> dict[str, List[Document]]:
        # Normalize input to always be a list
        if isinstance(documents, Document):
            documents = [documents]

        # Embed all documents in one batch (more efficient!)
        texts = [doc.content or "" for doc in documents]
        embeddings = self.embedder.run(texts=texts, mode="document")["embeddings"]

        # Assign embeddings back to documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        return {"documents": documents}


@component
class TextEmbedder:
    """
    A simple wrapper component that generates a dense embedding vector for a single text input.

    This component uses an embedder obtained from `EmbedderFactory`, which may be
    an in-memory model (`DenseEmbedder`) or a remote API (`ApiEmbedder`), depending on the config.

    Input:
        - text (str): The input string to embed.
        - mode (Literal["query", "document", "instruct"], default="document"):
            Contextual mode for the embedding. Use "query" for search queries,
            "document" for content, and "instruct" for instruction-tuned models.
        - instruction (Optional[str]): Optional instruction string used in instruct mode.

    Output:
        - embedding (List[float]): A dense embedding vector representing the input text.

    Example:
        embedder = TextEmbedder()
        result = embedder.run("What is semantic search?")
        vector = result["embedding"]
    """

    def __init__(self):
        self.embedder = EmbedderFactory().get_embedder()

    @component.output_types(embedding=List[float])
    def run(
        self,
        text: str,
        mode: Literal["query", "document", "instruct"] = "document",
        instruction: Optional[str] = None,
    ):
        embedding_result = self.embedder.run(texts=[text], mode=mode, instruction=instruction)
        embeddings = embedding_result["embeddings"]
        if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
            embedding = embeddings[0]
        else:
            raise TypeError(f"Expected embeddings to be a list of lists of floats, got {embeddings}")

        return {"embedding": embedding}
