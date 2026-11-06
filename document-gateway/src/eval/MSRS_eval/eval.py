
import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import yaml
from datasets import Dataset, DatasetDict, load_dataset
from haystack import Document
from pydantic import BaseModel
from document_common.components.base.batched_operations import batch_function_calls
from document_common.components.base.document_store import (
    DocumentStoreProvider,
    DocumentDocumentStore,
)
from document_common.components.base.embedder import TextEmbedder
from document_common.components.base.assembly import TextAssembly
from document_common.components.base.prompt import PromptProvider
from document_common.components.base.reranker import DocumentReranker
from document_common.components.base.retriever import Retriever
from document_common.utils.logger import Logger

from config import config
from src.eval.utils.assembly_evaluator import AssemblyEvaluator
from src.eval.utils.utils import evaluate_retrieving

CONFIG_TO_SNAP = {
    "llm": "config/configs/llm/llm_config.yaml",
    "embedder": "config/configs/embedder/embedder.yaml",
    "evaluation": "config/configs/evaluation/evaluation.yaml"
}
RERANKER_CONFIG = "config/configs/reranker/reranker.yaml"

class Ground_Truth(BaseModel):
    id: str
    query: str
    gold_documents: List[str]
    answer: List[str]


class MSEval:
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
        self.document_store_provider = DocumentStoreProvider()

        self.embedder = TextEmbedder()
        self.document_store = DocumentDocumentStore(DocumentStoreProvider().get_store(store_name, require_config_match=False), doc_ids=docs_setup)
        self.retriever = Retriever(document_store=self.document_store, with_rerank=False)
        self.generator = TextAssembly(prompt_name=story_assembly_prompt_name)

        self.story_assembly_prompt_name = story_assembly_prompt_name
        self.meeting_assembly_prompt_name = meeting_assembly_prompt_name
        self.truncate = truncate
        self.truncate_length = truncate_length
        self.g_eval_config = g_eval_config

        self.oracle_run = oracle_run
        self.with_rerank = with_rerank
        self.only_retrieval = only_retrieval

        PromptProvider(prompt_folder=config.get_current_config().others.prompt_folder, overwrite_existing_prompt=True)
        self.reranker = DocumentReranker()

        self.config_snapshot = self._load_config_snapshot()

    def _load_config_snapshot(self) -> dict[str, Any]:
        """Load configuration snapshots from YAML files specified in CONFIG_TO_SNAP."""
        config_snapshot: dict[str, Any] = {}
        for key, path in CONFIG_TO_SNAP.items():
            config_path = Path(path)
            with open(config_path, 'r') as f:
                config_snapshot[key] = yaml.safe_load(f)
        
        if self.with_rerank:
            with open(RERANKER_CONFIG, 'r') as f:
                config_snapshot["reranker"] = yaml.safe_load(f)
        
        return config_snapshot


    def download_dataset(self, config: str) -> DatasetDict:
        self.logger.info(f"Downloading MSRS dataset {config}...")

        cache_dir = str(self.local_dataset_path.parent)
        
        ds = load_dataset(self.dataset_path, config, cache_dir=cache_dir)

        return cast(DatasetDict, ds)
    
    def convert_dataset_to_ground_truths(self, dataset: DatasetDict, key:str = "train") -> List[Ground_Truth]:
        ground_truths: List[Ground_Truth] = []
        
        data: Dataset = dataset[key]

        for item in data:
            # Assuming each item has a 'text' and 'id' field
            ground_truth = Ground_Truth(
                id= item.get("id", ""), # type: ignore
                query= item.get("query", ""), # type: ignore
                gold_documents= item.get("gold_documents", ""), # type: ignore
                answer= item.get("answer", ""), # type: ignore
            )
            ground_truths.append(ground_truth)

        return ground_truths[:5]

    def generate_answer(self, prompt_name: str, query: str, documents: List[Document], truncate_length: Optional[int] = None) -> str:
        articles: List[str] = [doc.content or "" for doc in documents]

        if truncate_length:
            avg_len = truncate_length // len(documents)
            articles = [" ".join(a.split(" ")[:avg_len]) for a in articles]
        
        llm_answer = self.generator.run(
            prompt_name=prompt_name, 
            inputs={
                "query": query, 
                "article": "\n".join(articles)
            }
        )['llm_answer']

        return llm_answer

    def average_retrieval_scores(self, retrieve_scores: List[dict[str, float]]) -> dict[str, float]:
        final_scores: dict[str, float] = {}
        for key in retrieve_scores[0].keys():
            scores = [r[key] for r in retrieve_scores]
            final_scores[key] = sum(scores) / len(retrieve_scores)
        
        return final_scores

    def evaluation(self, gts: List[Ground_Truth], domain: str, oracle_setup: bool = False) -> dict[str, Any]:
        # Step 1 : Getting the retrieved documents
        docs_retrieved: List[List[Document]] = []
        for gt in gts:
            if oracle_setup:
                conditions = [{"field": "id", "operator": "==", "value": doc_id} for doc_id in gt.gold_documents]
                docs = self.document_store.filter_documents(filters={
                    "operator": "OR",
                    "conditions": conditions
                })
                if Counter([doc.id for doc in docs]) != Counter(gt.gold_documents):
                    message = "During oracle evaluation, document retrieved did not match the gold documents needed."
                    message += f"{[doc.id for doc in docs]} found, {gt.gold_documents} needed"
                    self.logger.error(message)
                    #raise ValueError(message)
            else:
                limit = 8 if domain == "story-corpus" else 3 # MSRS rules

                if self.with_rerank:
                    docs = self.retriever.run(
                        query=gt.query,
                        doc_type=domain,
                        top_k = 20 # MSRS rules
                    )["documents"]

                    docs = self.reranker.run(query=gt.query, documents=docs)["documents"][:limit]
                else:
                    docs = self.retriever.run(
                        query=gt.query,
                        doc_type=domain,
                        top_k = limit
                    )["documents"]
            
            docs_retrieved.append(docs)

        if not oracle_setup:
            # 2. Batch call retrieve evaluation
            self.logger.info("Starting retrieval evaluation for all ground truth entries")
            retrieve_scores = batch_function_calls(
                func=lambda docs, gt_docs: evaluate_retrieving(docs, gt_docs),
                positional_args_list=[[docs, gt.gold_documents] for docs, gt in zip(docs_retrieved, gts)],
                keyword_args_list=[None for _ in gts]
            )

            for r in retrieve_scores:
                if isinstance(r, Exception):
                    self.logger.error(f"Error during retrieval evaluation: {r}", exc_info=True)
                    raise r

            retrieve_scores = self.average_retrieval_scores(retrieve_scores)
            self.logger.info(f"Retrieval evaluation completed. Average scores: {retrieve_scores}")
        else:
            self.logger.info("Evaluation under oracle setup, retrieving is skipped")
            retrieve_scores = "oracle_setup"

        if self.only_retrieval:
            assembly_scores = "Evaluation ran without assembly."
        else:
            prompt_name = ""
            if domain == "meeting-corpus":
                prompt_name = self.meeting_assembly_prompt_name
            elif domain == "story-corpus":
                prompt_name = self.story_assembly_prompt_name

            # 3. Batch call answer assembly
            truncate_length = None
            if domain == "meeting-corpus":
                truncate_length = 20000
            if self.truncate and self.truncate_length:
                truncate_length = self.truncate_length
            
            self.logger.info("Starting answer assembly for all ground truth entries")
            assemblies = batch_function_calls(
                func=lambda prompt_name, query, documents, truncate_length: self.generate_answer(prompt_name, query, documents, truncate_length),
                positional_args_list=[[prompt_name, gt.query, docs, truncate_length] for gt, docs in zip(gts, docs_retrieved)],
                keyword_args_list=[None for _ in gts]
            )

            for g in assemblies:
                if isinstance(g, Exception):
                    self.logger.error(f"Error during answer assembly: {g}", exc_info=True)
                    raise g

            # 4. Evaluate the answers
            self.logger.info("Starting answer evaluation")
            assembly_scores = AssemblyEvaluator(
                g_eval_config=self.g_eval_config,
            ).run(predictions=assemblies, references=[gt.answer for gt in gts])

            self.logger.info(f"Answer evaluation completed. Scores: {assembly_scores}")

        return {
            "assembly_scores": assembly_scores,
            "retrieval_scores": retrieve_scores
        }

    def write_eval_results(self, run_id: str, meeting_metrics: Dict[str, Any], story_metrics: Dict[str, Any]) -> None:
        """
        Write evaluation results to src/eval/outputs/<run_id>.json

        Format:
            {
                "run_id": "...",
                "meeting_metrics": { ... },
                "story_metrics": { ... },
                "config_snapshot": { ... }
            }
        """

        output_dir = self.result_output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find the next available run number
        run_number = 0
        while True:
            output_path = output_dir / f"run_{run_number}.json"
            if not output_path.exists():
                break
            run_number += 1

        payload = {
            "run_id": f"run_{run_number}",
            "meeting_metrics": meeting_metrics,
            "story_metrics": story_metrics,
            "config_snapshot": self.config_snapshot,
        }

        with open(output_path, "w") as f:
            json.dump(payload, f, indent=4)

        self.logger.info(f"Saved eval results to {output_path}")

    def run(self):
        try:
            run_id = str(uuid.uuid4())
            self.logger.info(f"Starting evaluation run: {run_id}")

            configs = ["meeting-qa", "story-qa"]

            meeting_metrics = {}
            story_metrics = {}
            for cfg in configs:
                try:
                    self.logger.info(f"Processing {cfg} dataset")

                    # Step 1: Download datasets
                    self.logger.debug(f"Downloading dataset: {cfg}")
                    dataset = self.download_dataset(config=cfg)

                    # Step 2: Convert to GT format (custom object to make steps easier to manage)
                    self.logger.debug("Converting dataset to ground truth format")
                    gts = self.convert_dataset_to_ground_truths(dataset=dataset)
                    self.logger.info(f"Loaded {len(gts)} ground truth entries for {cfg}")

                    # Step 3: Evaluate both datasets (retrieving + llm assembly)
                    self.logger.info(f"Starting evaluation for {cfg}")
                    if cfg == "meeting-qa":
                        meeting_metrics = self.evaluation(gts, "meeting-corpus", self.oracle_run)
                    if cfg == "story-qa":
                        story_metrics = self.evaluation(gts, "story-corpus", self.oracle_run)

                    self.logger.info(f"Completed evaluation for {cfg}")

                except Exception as e:
                    self.logger.error(f"Error processing {cfg} dataset: {e}", exc_info=True)
                    raise

            # Step 4: Write output
            self.logger.info("Writing evaluation results")
            self.write_eval_results(
                run_id=run_id,
                meeting_metrics=meeting_metrics,
                story_metrics=story_metrics
            )

            self.logger.info(f"Evaluation run {run_id} completed successfully.")

        except Exception as e:
            self.logger.error(f"Error during eval: {e}", exc_info=True)
            raise
