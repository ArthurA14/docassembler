from typing import TYPE_CHECKING, Any, List

from mlflow.genai.scorers.builtin_scorers import BuiltInScorer
from document_common.utils.logger import Logger

if TYPE_CHECKING:
    pass

import json

from mlflow.entities.assessment import Feedback
from mlflow.entities.span import SpanType
from mlflow.entities.trace import Trace
from mlflow.genai.judges.base import JudgeField
from mlflow.genai.utils.trace_utils import (
    extract_request_from_trace,
    extract_response_from_trace,
    extract_retrieval_context_from_trace,
)

from src.eval.utils.assembly_evaluator import AssemblyEvaluator
from src.eval.utils.utils import evaluate_retrieval_ids


class MSRS_Scorer(BuiltInScorer):
    name: str = "retrieval_relevance"
    model: str | None = None
    required_columns: set[str] = {"inputs", "trace"}
    description: str = (
        "Evaluate whether each retrieved context chunk is relevant to the input request."
    )
    def __init__(self, /, **kwargs):
        super().__init__(**kwargs)

    @property
    def instructions(self) -> str:
        """Get the instructions of what this scorer evaluates."""
        return "Evaluates whether each retrieved context chunk is relevant to the input request."

    def get_input_fields(self) -> list[JudgeField]:
        """
        Get the input fields for the RetrievalRelevance judge.

        Returns:
            List of JudgeField objects defining the input fields based on the __call__ method.
        """
        return [
            JudgeField(
                name="trace",
                description=(
                    "The trace of the model's execution. Must contains at least one span with "
                    "type `RETRIEVER`. MLflow will extract the retrieved context from that span. "
                    "If multiple spans are found, MLflow will use the **last** one."
                ),
            ),
            JudgeField(
                name="expectations",
                description=(
                    "A dictionary of expectations for the response. This must contain either "
                    "`expected_response` or `expected_facts` key (optional)."
                ),
            ),
        ]

    def __call__(self, *, trace: Trace, expectations: dict[str, Any] | None = None) -> List[Feedback]:
        """
        Evaluate chunk relevance for each context chunk.

        Args:
            trace: The trace of the model's execution. Must contains at least one span with
                type `RETRIEVER`. MLflow will extract the retrieved context from that span.
                If multiple spans are found, MLflow will use the **last** one.

        Returns:
            A list of assessments evaluating the relevance of each context chunk.
            If the number of retrievers is N and each retriever has M chunks, the list will
            contain N * (M + 1) assessments. Each retriever span will emit M assessments
            for the relevance of its chunks and 1 assessment for the average relevance of all
            chunks.
        """
        logger = Logger().get_logger("eval.eval.mf_scorer")

        expectations = expectations or {}

        request = extract_request_from_trace(trace)
        span_id_to_context = extract_retrieval_context_from_trace(trace)
        span_id = list(span_id_to_context.keys())[0]
        response = extract_response_from_trace(trace)

        logger.info(f"request {request}") # {"query" : "...", "gold_documents": [...]}
        logger.info(f"response {response}") # {"response" : "llm_answer"}
        logger.info(f"expectations {expectations}") # {"gold_documents": [...], 'answer': "..."}
        logger.info(f"OUTPUTS : {trace.search_spans(span_type=SpanType.RETRIEVER)[0].outputs}") # [{'page_content' : '...', 'id' : '..'}]
        
        docs_retrieved = trace.search_spans(span_type=SpanType.RETRIEVER)[0].outputs
        retrieved_doc_ids: List[str] = [d['id'] for d in docs_retrieved]

        gold_documents = expectations.get('gold_documents', '')
        gold_answer = expectations.get('answer', '')

        response = json.loads(response or "")
        llm_answer = response.get('response', '')


        feedbacks: List[Feedback] = []
        if llm_answer:
            feedbacks.extend(self._compute_assembly_feedbacks(llm_answer, gold_answer, span_id))
        feedbacks.extend(self._compute_retrieval_feedbacks(retrieved_doc_ids, gold_documents, span_id))
        return feedbacks

    def _compute_assembly_feedbacks(
        self, llm_answer: str, gold_answer: List[str], span_id: str
    ) -> list[Feedback]:
        """Compute assembly metrics using assemblyEvaluator."""
        logger = Logger().get_logger("eval.eval.mf_scorer")
        
        g_evaluator = AssemblyEvaluator()
        metrics = g_evaluator.run([llm_answer], [gold_answer])
        
        feedbacks: List[Feedback] = []
        # Add response metrics to feedbacks
        if metrics["bert"]:
            bert_f1 = metrics["bert"]["f1"]
            feedbacks.append(Feedback(
                name="bert_f1_score",
                value=(bert_f1*100) or -1,
                span_id=span_id,
            ))
        
        for key, value in metrics['g_eval'].items():
            if value:
                feedbacks.append(Feedback(
                    name=f"g_eval_{key}",
                    value=(value['score']/5)*100,
                    span_id=span_id,
                ))
        
        if metrics["rouge"]:
            rouge1 = metrics["rouge"]["rouge1"]['high']['fmeasure']
            feedbacks.append(Feedback(
                name="rouge1",
                value=rouge1*100,
                span_id=span_id,
            ))
            rouge2 = metrics["rouge"]["rouge2"]['high']['fmeasure']
            feedbacks.append(Feedback(
                name="rouge2",
                value=rouge2*100,
                span_id=span_id,
            ))
        
        logger.info(f"Generated {len(feedbacks)} feedbacks for assembly metrics")
        return feedbacks

    def _compute_retrieval_feedbacks(
        self, retrieved_doc_ids: List[str], gold_documents: List[str], span_id: str
    ) -> list[Feedback]:
        """Compute retrieval metrics using evaluate_retrieval_ids."""
        logger = Logger().get_logger("eval.eval.mf_scorer")

        retrieval_metrics = evaluate_retrieval_ids(doc_ids=retrieved_doc_ids, ground_truth_docs=gold_documents)
        
        feedbacks: List[Feedback] = []
        for metric_name, metric_value in retrieval_metrics.items():
            feedback = Feedback(
                name=f"retrieval_{metric_name.lower()}",
                value=metric_value*100,
                span_id=span_id,
            )
            feedbacks.append(feedback)
        
        logger.info(f"Generated {len(feedbacks)} feedbacks for retrieval metrics")
        return feedbacks
