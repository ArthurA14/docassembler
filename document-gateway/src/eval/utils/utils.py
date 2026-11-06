import math
from pathlib import Path
from typing import List, Union

from haystack import Document
from document_common.utils.logger import Logger

from src.eval.doc_assembler_eval.models import DocRef

logger = Logger().get_logger("eval.utils")

def load_documents(document_dir: str | List[str]) -> List[DocRef]:
    """Load all .txt files from the documents directory."""

    if isinstance(document_dir, str):
        document_paths = [Path(document_dir)]
    else:
        document_paths = [Path(d) for d in document_dir]

    documents: List[DocRef] = []

    for dir in document_paths:
        if not dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {dir}")

        txt_files = list(dir.rglob("*.txt"))
        print(f"Found {len(txt_files)} .txt files in {dir}")

        for file_path in txt_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            doc_ref = DocRef(
                coreDataId = file_path.name,
                coreType = "document",
                coreDataSource = "Dataset Evaluator",
                plainText = content,
                filename = file_path.name
            )
            documents.append(doc_ref)

    if not documents:
        raise FileNotFoundError(f"No .txt files found in {document_paths}")
    return documents

def evaluate_retrieving(retrieved_docs: List[Document], ground_truth_docs: List[str]) -> dict[str, Union[float, int]]:
    doc_ids = [doc.id for doc in retrieved_docs]
    return evaluate_retrieval_ids(doc_ids, ground_truth_docs)

def evaluate_retrieval_ids(doc_ids: List[str], ground_truth_docs: List[str]) -> dict[str, Union[float, int]]:
    retrieved_set = set(doc_ids)
    ground_truth_set = set(ground_truth_docs)
    logger.debug(f"Evaluating retrieved documents: {len(doc_ids)} docs vs {len(ground_truth_docs)} ground truth docs")

    tp = len(retrieved_set.intersection(ground_truth_set))
    fp = len(retrieved_set) - tp
    fn = len(ground_truth_set) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    # NDCG calculation
    dcg = 0
    ideal_dcg = 0
    for i, doc in enumerate(doc_ids):
        rel = 1 if doc in ground_truth_set else 0
        dcg += rel / math.log2(i + 2)  # i starts from 0, log starts from 2
    for i in range(min(len(ground_truth_docs), len(doc_ids))):
        ideal_dcg += 1 / math.log2(i + 2)
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

    # MAP calculation (Average Precision for this query)
    num_relevant = 0
    sum_precision = 0
    for i, doc in enumerate(doc_ids):
        if doc in ground_truth_set:
            num_relevant += 1
            sum_precision += num_relevant / (i + 1)
    ap = sum_precision / len(ground_truth_set) if ground_truth_set else 0

    logger.debug(f"Retrieval metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, NDCG: {ndcg:.4f}, AP: {ap:.4f}")
    return {"Precision": precision, "Recall": recall, "F1": f1, "NDCG": ndcg, "AP": ap}