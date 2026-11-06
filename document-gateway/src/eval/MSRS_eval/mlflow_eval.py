
import os
import uuid
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import mlflow
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from haystack import Document as HaystackDocument
from haystack.document_stores.types import DocumentStore
from mlflow.entities import Document, SpanType
from document_common.components.base.document_store import (
    DocumentStoreProvider,
    DocumentDocumentStore,
)
from document_common.components.base.assembly import TextAssembly
from document_common.components.base.prompt import PromptProvider
from document_common.components.base.reranker import DocumentReranker
from document_common.components.base.retriever import Retriever
from document_common.utils.logger import Logger

from config import config
from src.eval.MSRS_eval.mlflow_scorer import MSRS_Scorer

CONFIG_TO_SNAP = {
    "llm": "config/configs/llm/llm_config.yaml",
    "embedder": "config/configs/embedder/embedder.yaml",
    "evaluation": "config/configs/evaluation/evaluation.yaml"
}
RERANKER_CONFIG = "config/configs/reranker/reranker.yaml"

ML_FLOW_URL = "http://mlflow.<CLUSTER_IP>.nip.io" # file:///home/lanthoine/mlruns


os.environ['AWS_ACCESS_KEY_ID'] = 'test'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'test'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = ML_FLOW_URL

class Domain(str, Enum):
    STORY = "story-corpus"
    MEETING = "meeting-corpus"

class MF_Run_Setup:
    document_store: Optional[DocumentStore] = None
    oracle: bool = False
    domain: Domain = Domain.STORY
    with_rerank: bool = False
    only_retrieval: bool = False
    retriever: Retriever | None = None
    reranker: DocumentReranker | None = None
    generator: TextAssembly | None = None
    story_assembly_prompt_name: str | None = None
    meeting_assembly_prompt_name: str | None = None

# add validator for limit

setup = MF_Run_Setup()

# Define a retriever function with proper span type
@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_docs(query: str, gold_documents: List[str]) -> List[Document]:
    if not (setup.document_store and setup.retriever):
        raise ValueError("We can't perform an evaluation without document_store or retriever set")

    docs: List[HaystackDocument] = []

    if setup.oracle:
        conditions = [{"field": "id", "operator": "==", "value": doc_id} for doc_id in gold_documents]
        docs = setup.document_store.filter_documents(filters={
            "operator": "OR",
            "conditions": conditions
        })
        if Counter([doc.id for doc in docs]) != Counter(gold_documents):
            message = "During oracle evaluation, document retrieved did not match the gold documents needed."
            message += f"{[doc.id for doc in docs]} found, {gold_documents} needed"
            #raise ValueError(message)
    else:
        limit = 8 if setup.domain == Domain.STORY else 3

        if setup.with_rerank:
            if not setup.reranker:
                raise ValueError("We can't perform an evaluation with reranking but without reranker")

            docs = setup.retriever.run(
                query=query,
                doc_type=setup.domain,
                top_k = 20 # MSRS rules
            )["documents"]

            docs = setup.reranker.run(query=query, documents=docs)["documents"][:limit]
        else:
            docs = setup.retriever.run(
                query=query,
                doc_type=setup.domain,
                top_k = limit
            )["documents"]
    
    return [Document(page_content=doc.content or "", id=doc.id, metadata= {"chunk_id" : doc.id}) for doc in docs]


# Define your RAG agent
@mlflow.trace
def generate_answer(query: str, gold_documents: List[str]):
    if not setup.generator:
        raise ValueError("We can't perform an evaluation without a generator")

    docs = retrieve_docs(query, gold_documents)
    articles: List[str] = [doc.page_content or "" for doc in docs]
    
    if setup.only_retrieval:
        llm_answer = None
    else :
        truncate_length = None
        if setup.domain == Domain.MEETING:
            truncate_length = 20000
        
        if truncate_length:
            avg_len = truncate_length // len(docs)
            articles = [" ".join(a.split(" ")[:avg_len]) for a in articles]
            
        if setup.domain == Domain.MEETING:
            prompt_name = setup.meeting_assembly_prompt_name
        else:
            prompt_name = setup.story_assembly_prompt_name

        llm_answer = setup.generator.run(
            prompt_name=prompt_name,
            inputs={
                "query": query, 
                "article": "\n".join(articles)
            }
        )['llm_answer']

    return {"response": llm_answer}


class MLFlowEval:
    def __init__(
            self,
            dataset_path: str,
            local_dataset_path: Union[str, Path],
            result_output_path: Union[str, Path],
            store_name: str,
            story_assembly_prompt_name: str,
            meeting_assembly_prompt_name: str,
            truncate: bool,
            truncate_length: int,
            docs_setup: Optional[List[str]]= None,
            g_eval_config: Optional[Dict[str, Any]] = None,
            oracle_run:bool = False,
            with_rerank: bool = False,
            only_retrieval: bool=False
        ):
        self.logger = Logger().get_logger("eval.eval")

        self.dataset_path = dataset_path
        self.local_dataset_path = Path(local_dataset_path) if isinstance(local_dataset_path, str) else local_dataset_path
        self.result_output_path = Path(result_output_path) if isinstance(result_output_path, str) else result_output_path

        setup.story_assembly_prompt_name = story_assembly_prompt_name
        setup.meeting_assembly_prompt_name = meeting_assembly_prompt_name

        PromptProvider(prompt_folder=config.get_current_config().others.prompt_folder, overwrite_existing_prompt=True)

        self.config_snapshot = self._load_config_snapshot()

        self.document_store_provider = DocumentStoreProvider()
        setup.document_store = DocumentDocumentStore(DocumentStoreProvider().get_store(store_name, require_config_match=False), doc_ids=docs_setup)
        setup.retriever = Retriever(document_store=setup.document_store, with_rerank=False)
        setup.reranker = DocumentReranker()
        setup.generator = TextAssembly(prompt_name=story_assembly_prompt_name)
        setup.oracle = oracle_run
        setup.with_rerank = with_rerank
        setup.only_retrieval = only_retrieval

    def _load_config_snapshot(self) -> dict[str, Any]:
        """Load configuration snapshots from YAML files specified in CONFIG_TO_SNAP."""
        config_snapshot: dict[str, Any] = {}
        for key, path in CONFIG_TO_SNAP.items():
            config_path = Path(path)
            with open(config_path, 'r') as f:
                config_snapshot[key] = yaml.safe_load(f)
        
        if setup.with_rerank:
            with open(RERANKER_CONFIG, 'r') as f:
                config_snapshot["reranker"] = yaml.safe_load(f)
        
        return config_snapshot


    def download_dataset(self, domain: Domain) -> DatasetDict:
        if domain == Domain.MEETING:
            cfg = "meeting-qa"
        else:
            cfg = "story-qa"
        self.logger.info(f"Downloading MSRS dataset {cfg}...")

        cache_dir = str(self.local_dataset_path.parent)
        
        ds = load_dataset(self.dataset_path, cfg, cache_dir=cache_dir)

        return cast(DatasetDict, ds)
    
    def convert_dataset_to_mf_dataset(self, dataset: DatasetDict, key:str = "train") -> List[dict[str, Any]]:
        data: Dataset = dataset[key]

        mf_dataset = []
        print(data)
        for item in data:
            # Assuming each item has a 'text' and 'id' field4
            mf_dataset.append({
                "inputs": {
                    "query": item.get("query", ""), # type: ignore
                    "gold_documents": item.get("gold_documents", ""), # type: ignore
                },
                "expectations": {
                    "gold_documents": item.get("gold_documents", ""), # type: ignore
                    "answer": item.get("answer", ""), # type: ignore
                },
            })

        return mf_dataset[:5]

    def run(self):
        try:
            run_id = str(uuid.uuid4())
            self.logger.info(f"Starting evaluation run: {run_id}")

            mlflow.set_tracking_uri(ML_FLOW_URL)
                        
            for domain in [Domain.STORY, Domain.MEETING]:
                try:
                    mlflow.set_experiment(f"MSRS/{domain}")
                    with mlflow.start_run(run_name=f"eval_{domain}_{run_id}"):
                        mlflow.log_params(self.config_snapshot)
                        setup.domain = domain
                        # Step 1: Download datasets
                        self.logger.debug(f"Downloading dataset: {domain}")
                        dataset = self.download_dataset(domain=domain)

                        # Step 2: Convert to GT format (custom object to make steps easier to manage)
                        self.logger.debug("Converting dataset to ground truth format")
                        mf_dataset = self.convert_dataset_to_mf_dataset(dataset=dataset)
                        self.logger.info(f"Loaded {len(mf_dataset)} entries for {domain}")


                        # Run evaluation with multiple RAG judges
                        eval_results = mlflow.genai.evaluate(
                            data=mf_dataset,
                            predict_fn=generate_answer,
                            scorers=[
                                MSRS_Scorer()
                            ],
                        )

                        self.logger.info(f"Completed evaluation for {domain}")

                except Exception as e:
                    self.logger.error(f"Error processing {domain} dataset: {e}", exc_info=True)
                    raise

            self.logger.info(f"Evaluation run {run_id} completed successfully.")
            # print(eval_results)

        except Exception as e:
            self.logger.error(f"Error during eval: {e}", exc_info=True)
            raise