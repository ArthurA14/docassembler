# MSRS Evaluation Scripts

This directory contains scripts for evaluating the MSRS (Multi-Source Retrieval and Summarization) dataset. The evaluation process consists of two main phases: setup and evaluation.

## Overview

The MSRS evaluation framework allows you to:
1. **Setup phase**: Download the MSRS dataset, convert it to Haystack Documents, embed the documents, and store them in a document store
2. **Evaluation phase**: Evaluate retrieval and assembly performance on the MSRS dataset using various metrics

## Directory Structure

```
MSRS_eval/
├── main.py              # Main entry point with CLI interface
├── eval.py              # Evaluation logic and metrics calculation
├── eval_setup.py        # Setup script for dataset preparation
├── data/                # Directory for dataset storage
└── outputs/             # Directory for evaluation results
```

## Prerequisites

- Python 3.8+
- uv (Python package manager)
- Required dependencies (see `pyproject.toml`)

## Configuration

The evaluation uses configuration from `config/configs/evaluation/evaluation.yaml`. Ensure this file contains the correct settings for:
- `dataset_path`: Path to the MSRS dataset
- `local_dataset_path`: Local path for dataset caching
- `store_name`: Document store name
- `result_output_path`: Path for evaluation results
- `story_assembly_prompt_name`: Prompt for story assembly
- `meeting_assembly_prompt_name`: Prompt for meeting assembly
- `g_eval_relevance_prompt_name`: Prompt for G-Eval relevance evaluation
- `g_eval_consistency_prompt_name`: Prompt for G-Eval consistency evaluation

## Usage

The main script provides several command-line options:

```bash
# Show help
uv run -m src.eval.MSRS_eval.main --help

# Run both setup and evaluation (default)
uv run -m src.eval.MSRS_eval.main

# Run only setup
uv run -m src.eval.MSRS_eval.main --setup

# Run only evaluation
uv run -m src.eval.MSRS_eval.main --eval

# Run only clean up
uv run -m src.eval.MSRS_eval.main --clean
```

### Command Line Arguments

- `--setup`: Run only the setup phase (dataset download, embedding, storage)
- `--eval`: Run only the evaluation phase
- `--oracle`: Run evaluation in oracle mode (LLM receives perfect documents)
- `--rerank`: Run evaluation with the reranker enabled
- `--only_retrieval`: Run only the retrieval part of the evaluation (must be used with `--eval`)
- `--clean`: Clean up the document store (removes added documents). Can be used standalone or after other operations.

### Notes

- When no flags are specified, the script runs setup, evaluation, and cleanup by default
- The `--only_retrieval` flag requires `--eval` to be set
- The `--oracle` and `--rerank` flags are only effective when `--eval` is used

## Setup Phase (`eval_setup.py`)

The setup phase performs the following steps:

1. **Download Dataset**: Downloads the MSRS dataset from HuggingFace (`yale-nlp/MSRS`)
2. **Convert to Documents**: Converts meeting-corpus and story-corpus data to Haystack Documents with appropriate metadata
3. **Embed Documents**: Embeds documents using the configured embedder, with chunking for long documents
4. **Store Documents**: Writes embedded documents to the specified document store

Key features:
- Automatic chunking of long documents (>1000 tokens)
- Weighted average embedding for chunked documents
- Metadata preservation for document type tracking

## Evaluation Phase (`eval.py`)

The evaluation phase performs comprehensive evaluation of retrieval and assembly:

1. **Retrieval Evaluation**:
   - Precision, Recall, F1
   - NDCG (Normalized Discounted Cumulative Gain)
   - AP (Average Precision)

2. **assembly Evaluation**:
   - ROUGE scores (precision, recall, F1 for different n-gram levels)
   - BERTScore (precision, recall, F1)
   - G-Eval (relevance and consistency ratings)

3. **Metrics Aggregation**: Calculates average scores across all test cases

The evaluation processes both meeting-corpus and story-corpus datasets and saves results to JSON files in the outputs directory.

## Output Files

Evaluation results are saved in the `outputs/` directory with UUID-named JSON files containing:
- Run ID
- Meeting metrics
- Story metrics
- Failure/success counts
