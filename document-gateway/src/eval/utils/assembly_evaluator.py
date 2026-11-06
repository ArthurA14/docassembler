from typing import Any, Dict, List, Optional, TypedDict

import torch
from bert_score import scorer
from document_common.components.base.batched_operations import batch_function_calls
from document_common.components.base.prompt import PromptProvider
from document_common.utils.logger import Logger
from torch import Tensor

from config import config
from src.eval.utils.g_eval import GEval
from src.eval.utils.rouge import Rouge


class GEvalResults(TypedDict):
    score: float
    predictions_evaluated: int
    failures: int


class RougeScoreDetail(TypedDict):
    precision: float
    recall: float
    fmeasure: float


class RougeScoreLevel(TypedDict):
    low: RougeScoreDetail
    mid: RougeScoreDetail
    high: RougeScoreDetail


class RougeResults(TypedDict):
    rouge1: RougeScoreLevel
    rouge2: RougeScoreLevel
    rougeL: RougeScoreLevel
    rougeLsum: RougeScoreLevel


class BertResult(TypedDict):
    precision: float
    recall: float
    f1: float


class AssemblyEvaluatorResults(TypedDict):
    rouge: Optional[RougeResults]
    bert: Optional[BertResult]
    g_eval: Dict[str, Optional[GEvalResults]]
    total_documents_evaluated: int


class AssemblyEvaluator:
    """Class to evaluate TEXT compared to another TEXT using ROUGE, BERT, and G-EVAL metrics."""

    def __init__(
        self,
        g_eval_prompts : Optional[Dict[str, Any]] = {
            "relevance" : "G_eval_relevance",
            "consistency" : "G_eval_consistency",
        },
        g_eval_config: Optional[Dict[str, Any]] = None,
    ):
        self.logger = Logger().get_logger("assembly_evaluator")

        # Initialize evaluators
        self.rouge = Rouge()

        # Initialize GEval with configuration if provided
        if g_eval_config:
            self.logger.info(f"Initializing GEval with custom configuration: {g_eval_config}")
            self.g_eval = GEval(
                **g_eval_config
            )
        else:
            self.logger.info("Initializing GEval with default configuration")
            self.g_eval = GEval()

        self.g_eval_prompts = g_eval_prompts or {}

        PromptProvider(prompt_folder=config.get_current_config().others.prompt_folder, overwrite_existing_prompt=True)

        self.bert_scorer = None
        try :
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.bert_scorer = scorer.BERTScorer(model_type="roberta-large", lang="en", device=device)
        except Exception as e:
            self.logger.warning(f'BertScore initialization failed with {e}', exc_info=True)

    def convert_rouge_to_dict(self, rouge_score: Dict) -> None:
        """Convert ROUGE AggregateScore objects to simple dictionaries."""
        for rouge_type in rouge_score:
            agg_score = rouge_score[rouge_type]
            rouge_score[rouge_type] = {
                "low": {
                    "precision": float(agg_score.low.precision),
                    "recall": float(agg_score.low.recall),
                    "fmeasure": float(agg_score.low.fmeasure)
                },
                "mid": {
                    "precision": float(agg_score.mid.precision),
                    "recall": float(agg_score.mid.recall),
                    "fmeasure": float(agg_score.mid.fmeasure)
                },
                "high": {
                    "precision": float(agg_score.high.precision),
                    "recall": float(agg_score.high.recall),
                    "fmeasure": float(agg_score.high.fmeasure)
                }
            }

    def _compute_g_eval_score(self, prediction: str, reference: str, prompt_name: str) -> float:
        result = self.g_eval.run(variables={
                "document": prediction,
                "summary": reference
            },
            prompt_name=prompt_name
        )

        return result["rating"]


    def _parrallele_g_eval(self, predictions: List[str], references: List[List[str]], prompt_name: str) -> GEvalResults:
        flat_predictions = [x for x, sub in zip(predictions, references) for _ in sub]
        flat_references = [y for sub in references for y in sub]

        results = batch_function_calls(
            func=lambda prediction, reference, prompt_name: self._compute_g_eval_score(prediction, reference, prompt_name),
            positional_args_list=[
                [prediction, reference, prompt_name] 
                for prediction, reference in zip(flat_predictions, flat_references)
            ],
            keyword_args_list=[None for _ in flat_references]
        )

        failures = 0
        for i in range(len(results)):
            r = results[i]
            if isinstance(r, Exception):
                results.pop(i)
                failures+=1
        
        return {
            "score" : sum(results)/len(results),
            "predictions_evaluated" : len(results),
            "failures": failures
        }

    def run(self, predictions: List[str], references: List[List[str]]) -> AssemblyEvaluatorResults:
        """Run the evaluation process."""
        rouge_score = None
        # ROUGE evaluation
        try:
            rouge_score = self.rouge._compute(predictions, references)
            self.convert_rouge_to_dict(rouge_score)
            self.logger.info(str(rouge_score))
        except Exception as e:
            self.logger.error(f"Error computing ROUGE: {e}", exc_info=True)
            rouge_score = None

        bert_result = None
        # BERT evaluation
        try:
            if self.bert_scorer:
                p, r, f1 = self.bert_scorer.score(
                    predictions, 
                    references,
                )

                p = p if isinstance(p, Tensor) else Tensor(p)
                r = r if isinstance(r, Tensor) else Tensor(r)
                f1 = f1 if isinstance(f1, Tensor) else Tensor(f1)
                
                bert_result = {
                    "precision": float(p.mean()),
                    "recall": float(r.mean()),
                    "f1": float(f1.mean())
                }
                self.logger.info(bert_result)
        except Exception as e:
            bert_result = None
            self.logger.error(f"Error computing BERT score: {e}", exc_info=True)

        # G-EVALs
        g_eval_results = {}
        for g_eval_name, g_eval_prompt in self.g_eval_prompts.items():
            result = None
            try:
                result = self._parrallele_g_eval(predictions, references, g_eval_prompt)
                self.logger.info(f"G-EVAL {g_eval_name} results with {g_eval_prompt}")
                self.logger.info(result)
            except Exception as e:
                result = None
                self.logger.error(f"Error computing G-EVAL {g_eval_name}: {e}", exc_info=True)
            
            g_eval_results[g_eval_name] = result

        # Compute averages
        self.logger.info("Computing averages...")
        
        final_scores: AssemblyEvaluatorResults = {
            "rouge": rouge_score, # type: ignore
            "bert": bert_result,
            "g_eval": g_eval_results,
            "total_documents_evaluated": len(predictions)
        }

        self.logger.info(f"Evaluation complete. Results : {final_scores}")
        return final_scores