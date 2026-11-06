# Evaluation Directory

This directory contains evaluation tools and datasets for assessing the performance of the document assembly system. It includes two main subdirectories: `MSRS_eval` and `doc_assembler_eval`.

## MSRS Evaluation (`MSRS_eval`)

The `MSRS_eval` directory implements an evaluation framework for the retrieval and assembly components of the document assembly system, utilizing the publicly available MSRS dataset.

### Overview

This evaluation pipeline assesses both the retrieval system and the LLM performance. It downloads the MSRS dataset automatically and evaluates the quality of retrieval based on the system's embedding and retrieval mechanisms. Then, it generates responses and compares them against the ground truth answers in the MSRS dataset.

### MSRS Dataset Methodology

The evaluation draws inspiration from the MSRS paper and dataset, which provide a comprehensive methodology for evaluating Retrieval-Augmented assembly (RAG) systems. The dataset defines two distinct evaluation scenarios:

#### Stories Scenario
- Contains 1,140 text chunks from various stories
- Includes 635 associated questions
- Provides gold documents for retrieval evaluation and ideal answers for assembly assessment

#### Meetings Scenario
- Contains 231 chunks from meeting transcripts
- Includes 436 related questions
- Provides gold documents for retrieval evaluation and ideal answers for assembly assessment

### Evaluation Metrics

#### Retrieval Metrics
- Precision
- Recall
- F1
- Normalized Discounted Cumulative Gain (nDCG)
- Mean Average Precision (MAP)

#### assembly Metrics
- G-Eval (relevance and consistency)
- ROUGE (n-gram overlap)
- BERTScore (semantic similarity)

### Usage

```bash
# Show help
uv run -m src.eval.MSRS_eval.main --help

# Run full evaluation (setup + eval + cleanup)
uv run -m src.eval.MSRS_eval.main

# Run only evaluation with custom options
uv run -m src.eval.MSRS_eval.main --eval --oracle
uv run -m src.eval.MSRS_eval.main --eval --rerank
uv run -m src.eval.MSRS_eval.main --eval --only_retrieval
```

For more details, see the [MSRS_eval README](MSRS_eval/README.md).

## document assembly Evaluation (`doc_assembler_eval`)

The `doc_assembler_eval` directory enables custom evaluation using custom datasets. Users can configure their own data folder containing documents to be embedded and inserted into the system's database. An experiment configuration file specifies the evaluation runs, where each experiment defines a subject, generates multiple documents, and compares them against a ground truth document.

### Usage

```bash
# Run full evaluation pipeline
uv run -m src.eval.doc_assembler_eval.main --document_dir /path/to/documents

# Run individual steps
uv run -m src.eval.doc_assembler_eval.main --import --document_dir /path/to/docs
uv run -m src.eval.doc_assembler_eval.main --generate --document_dir /path/to/docs
uv run -m src.eval.doc_assembler_eval.main --evaluate --document_dir /path/to/docs
```

For more details, see the [doc_assembler_eval README](doc_assembler_eval/README.md).

### Evaluation Metrics

The same metrics used in MSRS evaluation are applied:
- G-Eval (relevance and consistency)
- ROUGE (n-gram overlap)
- BERTScore (semantic similarity)

