
from pathlib import Path
from typing import Any, List, Optional, Union, cast

import numpy as np
from datasets import Dataset, load_dataset
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import MissingDocumentError
from haystack.document_stores.types import DocumentStore
from document_common.components.base.document_store import DocumentStoreProvider
from document_common.components.base.embedder import EmbedderFactory
from document_common.components.base.writer import Writer
from document_common.utils.logger import Logger


class MSEvalSetup:
    """
    Setup script for MSRS dataset evaluation.

    This script:
    1. Downloads the MSRS dataset from HuggingFace if not already present
    2. Converts meeting-corpus and story-corpus to Haystack Documents
    3. Embeds documents using the configured embedder
    4. Writes embedded documents a the document store
    """

    def __init__(
            self, 
            dataset_path: str, 
            store_name:str, 
            local_dataset_path: Union[str, Path],
            embedder_config: dict[str, Any]
        ):
        self.logger = Logger().get_logger("eval.setup")

        self.dataset_path = dataset_path or "yale-nlp/MSRS"
        self.local_dataset_path = Path("./data/msrs_dataset_setup") if isinstance(local_dataset_path, str) else local_dataset_path
        
        self.document_store_provider = DocumentStoreProvider()
        self.document_store = self.document_store_provider.get_store(store_name, require_config_match=False)

        self.embedder_config = embedder_config

    def download_dataset(self, domain: str) -> Dataset:
        self.logger.info("Downloading all MSRS configs...")

        cache_dir = str(self.local_dataset_path.parent)

        self.logger.info(f"Loading domain: {domain}")
        ds = load_dataset(self.dataset_path, domain, cache_dir=cache_dir)

        return cast(Dataset, ds["corpus"])

    def convert_corpus_to_documents(self, dataset: Dataset, domain: str) -> List[Document]:
        """
        Convert meeting-corpus and story-corpus to Haystack Documents.

        Args:
            dataset: The loaded MSRS dataset

        Returns:
            List of Documents with meta.type = "eval"
        """
        documents: List[Document] = []
        for item in dataset:
            content: str = item.get("text", "") # type: ignore
            id: str = item.get("id", "") # type: ignore

            if content:
                document = Document(
                    id=id,
                    content=content,
                    meta={
                        "type": domain,
                        "source": "MSRS",
                        "dataset": "yale-nlp/MSRS",
                        "embedding_config" : self.embedder_config
                    },
                )
                documents.append(document)

        self.logger.info(f"Created {len(documents)} documents from corpus {domain}")
        return documents

    def embed_documents(
        self, documents: List[Document], max_tokens: int = 1000, embedder_config: Optional[dict[str, Any]]=None
    ) -> List[Document]:
        """
        This function embedds the documents AND if they are longer than max_tokens it chunks them, embedd the chunks and compute an average
        """
        self.logger.info(f"Embedding {len(documents)} documents...")

        embedder = EmbedderFactory(embedder_config).get_embedder()
        self.logger.info(f"Using embedding config :  {embedder_config}")
    
        # Following method from the MSRS paper
        embedded_docs_count = 0
        for doc in documents:
            content = doc.content or ""
            tokens = content.split()
            
            if len(tokens) > max_tokens:
                chunks = [
                    tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)
                ]
                chunks = [" ".join(chunk) for chunk in chunks]
                chunk_embeddings = np.array(
                    embedder.run(texts=chunks)["embeddings"]
                )
                weights = [len(chunk.split()) for chunk in chunks]
                # Calculate the weighted average of the embeddings
                weighted_avg_embedding = np.average(
                    chunk_embeddings, axis=0, weights=weights
                )
                doc.embedding = weighted_avg_embedding.tolist()
            else:
                embedding = embedder.run(texts=[content])["embeddings"][0]
                doc.embedding = embedding
            
            embedded_docs_count += 1

        self.logger.info(f"Successfully embedded {embedded_docs_count} documents")

        return documents

    def write_to_document_store(
        self,
        document_store: DocumentStore, 
        documents: List[Document]
    ) -> list[str]:
        self.logger.info(
            f"Writing {len(documents)} documents to document store '{document_store}'..."
        )

        writer = Writer(document_store)
        count = writer.run(documents=documents, batch_size=1)["doc_ids"]

        self.logger.info(f"Successfully wrote {count} documents to document store")
        return count

    def run(self) -> List[str]:
        self.logger.info("Starting MSRS dataset setup for evaluation...")

        try:
            domains = ["meeting-corpus", "story-corpus"] # in MSRS there is 2 datasets
            written_ids: List[str] = []
            for domain in domains:
                # Step 1: Download dataset
                dataset = self.download_dataset(domain)

                # Step 2: Convert corpus to documents
                documents = self.convert_corpus_to_documents(dataset, domain)

                if not documents:
                    raise ValueError("No documents were created from the corpus")

                # Step 3: Embed documents
                embedded_documents = self.embed_documents(documents=documents, embedder_config=self.embedder_config.copy())
                
                # Step 4: Write to document store
                new_written_ids = []
                new_written_ids = self.write_to_document_store(self.document_store, embedded_documents)

                summary = {
                    "total_documents": len(documents),
                    "embedded_documents": len(embedded_documents),
                    "written_documents": len(new_written_ids),
                    "dataset_path": str(self.local_dataset_path),
                    "message": "MSRS dataset setup completed successfully",
                }

                self.logger.info(f"Setup completed for domain {domain}: {summary}")

                written_ids += new_written_ids

            return written_ids

        except Exception as e:
            self.logger.error(f"Error during setup: {e}", exc_info=True)
            raise
    
    def clean_setup(self, doc_ids: Optional[List[str]] = None):
        if not doc_ids:
            all_docs = self.document_store.filter_documents()
            doc_ids = [doc.id for doc in all_docs]
        
        if doc_ids:
            try :
                self.document_store.delete_documents(document_ids=doc_ids)
            except MissingDocumentError as e:
                self.logger.warning(f"While trying to clean the evaluation document store, a MissingDocumentError was raised : {e}")
