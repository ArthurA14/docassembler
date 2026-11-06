import time
from typing import Any, Dict, List, Optional

import requests
from haystack import Document, component
from pydantic import BaseModel

from config import config
from document_common.utils.logger import Logger
from document_common.utils.reranker_utils import (
    normalize_scores,
)


class RerankMatch(BaseModel):
    """Single reranked match result"""

    index: int
    score: float
    text: str
    method: str  # "reranker" or "fuzzy"


class RerankerAPIClient:
    """Client for reranker API endpoint.

    Loads configuration from the global config singleton by default.
    Individual parameters can be overridden by passing them explicitly.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ):
        # Load defaults from config
        reranker_config = config.get_current_config().reranker
        api_config = reranker_config.api

        # Use provided values or fall back to config
        self.api_url = api_url or api_config.url
        self.api_key = api_key or api_config.api_key_env_var
        self.model = model or api_config.model
        self.timeout = timeout or api_config.timeout_seconds
        self.max_retries = max_retries or api_config.max_retries
        self.retry_delay = retry_delay or api_config.retry_delay
        self.length_limit = 70000
        self.logger = Logger().get_logger("reranker.api")

    def rerank(self, query: str, texts: List[str]) -> Optional[List[Dict[str, Any]]]:
        """
        Call reranker API to score query against texts.

        Args:
            query: Query text
            texts: List of candidate texts to rank

        Returns:
            List of dicts with 'index' and 'relevance_score' keys, or None on failure
        """
        if not texts:
            return []

        headers = {"Content-Type": "application/json"}

        self.logger.debug(f"Reranking {len(texts)} texts.")

        words = [t.split(' ') for t in texts]
        if sum([len(w) for w in words]) > self.length_limit:
            self.logger.warning("Texts for reranking are too long, they will be truncate.")
            self.logger.debug(f"Original len {sum(len(t) for t in texts)}")

            avg_len = self.length_limit // len(texts)
            texts = [" ".join(w[:avg_len]) for w in words]
            self.logger.debug(f"New len {sum(len(t) for t in texts)}")

        payload = {"model": self.model, "query": query, "documents": texts}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
                latency = time.time() - start_time

                response.raise_for_status()
                result = response.json()

                self.logger.debug(f"Reranker API call successful (latency: {latency:.3f}s, texts: {len(texts)})")

                results = result.get("results", [])
                return results

            except requests.exceptions.Timeout:
                self.logger.warning(f"Reranker API timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                return None

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Reranker API request failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                return None

            except Exception as e:
                self.logger.error(f"Unexpected error in reranker API call: {e}")
                return None

        return None


@component
class DocumentReranker:
    """
    Haystack component for reranking documents using reranker model.

    Used in retrieval pipeline  to improve document ranking quality
    for summary assembly by scoring semantic relevance beyond embedding similarity.
    """

    def __init__(self):
        self.logger = Logger().get_logger("reranker.document")
        reranker_config = config.get_current_config().reranker

        doc_config = reranker_config.document_mode
        enabled = doc_config.get("enabled", True) if isinstance(doc_config, dict) else getattr(doc_config, "enabled", True)
        if not enabled:
            self.logger.warning("Document reranker is disabled in configuration")
            self.enabled = False
            self.top_k_input = 100
            self.top_k_output = 50
            self.min_score = 0.0
            self.normalize = True
            self.fallback_to_embedding = True
            return

        self.enabled = True
        doc_config = reranker_config.document_mode

        def _get_attr(cfg, key, default):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        self.api_client = RerankerAPIClient()

        self.top_k_input = _get_attr(doc_config, "top_k_input", 100)
        self.top_k_output = _get_attr(doc_config, "top_k_output", 50)
        self.min_score = _get_attr(doc_config, "min_score", 0.0)
        self.normalize = _get_attr(doc_config, "normalize_scores", True)
        self.fallback_to_embedding = _get_attr(doc_config, "fallback_to_embedding", True)

        self.logger.debug(f"DocumentReranker initialized (input: {self.top_k_input}, output: {self.top_k_output})")

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document], top_k_output: Optional[int] = None) -> Dict[str, List[Document]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Query string for relevance scoring
            documents: List of retrieved documents

        Returns:
            Dict with 'documents' key containing reranked documents
        """
        if not top_k_output:
            top_k_output = self.top_k_output

        if not self.enabled:
            self.logger.debug("Reranker disabled, returning original documents")
            return {"documents": documents}

        if not documents:
            return {"documents": []}

        # Take top K documents to send to reranker
        input_docs = documents[: self.top_k_input]
        texts = [doc.content or "" for doc in input_docs]

        # Call reranker API
        results = self.api_client.rerank(query=query, texts=texts)

        if results is None:
            self.logger.warning("Reranker API failed, falling back to original order")
            if self.fallback_to_embedding:
                # Sort by embedding score if available
                sorted_docs = sorted(documents, key=lambda d: d.score or 0.0, reverse=True)
                return {"documents": sorted_docs[: top_k_output]}
            return {"documents": documents[: top_k_output]}

        # Process results and update document scores
        scored_docs = []
        for result in results:
            idx = result.get("index", 0)
            score = result.get("relevance_score", 0.0)

            if idx < len(input_docs) and score >= self.min_score:
                doc = input_docs[idx]
                doc.score = score
                scored_docs.append(doc)

        if not scored_docs:
            self.logger.warning("No documents passed minimum score threshold, using fallback")
            return {"documents": documents[: top_k_output]}

        # Normalize scores if requested
        if self.normalize and scored_docs:
            scores = [d.score for d in scored_docs]
            normalized = normalize_scores(scores)
            for doc, norm_score in zip(scored_docs, normalized):
                doc.score = norm_score

        # Sort by score and take top K
        scored_docs.sort(key=lambda d: d.score or 0.0, reverse=True)
        output_docs = scored_docs[: top_k_output]

        self.logger.debug(f"Reranked {len(documents)} documents → returned top {len(output_docs)}")
        return {"documents": output_docs}


