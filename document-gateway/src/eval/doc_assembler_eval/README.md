# document assembly Evaluation (`doc_assembler_eval`)

This directory contains the evaluation framework for assessing the performance of the document assembly system using custom datasets. It enables end-to-end evaluation by generating documents from provided documents and comparing them against ground truth documents using multiple metrics.

## Overview

The `doc_assembler_eval` module performs automated evaluation of the document assembly pipeline by:

1. **Document importation**: Loading and importing text documents into the system's database via the API
2. **document assembly**: Creating multiple documents for each evaluation scenario using the document assembly API
3. **document Evaluation**: Assessing generated documents against ground truth using ROUGE, BERTScore, and G-Eval metrics

## Prerequisites

- A running instance of the document assembly API (default: `http://localhost:5000`)
- Text documents in `.txt` format
- Ground truth data in JSONL format

## Configuration

The evaluation is configured through the `config/configs/evaluation/evaluation.yaml` file under the `doc_assembler_eval` section:

```yaml
doc_assembler_eval:
  api_base_url: "http://localhost:5000"          # Base URL of the running API
  documents_dir: ["path/to/documents"]          # List of directories containing .txt documents
  documents_dir: "src/eval/doc_assembler_eval/documents"  # Directory to save generated documents
  ground_truth_path: "path/to/ground_truth.jsonl"  # Path to ground truth JSONL file
  n_document_per_gt: 2                             # Number of documents to generate per ground truth entry
  result_dir: "src/eval/doc_assembler_eval/results" # Directory to save evaluation results
```

## Data Preparation

### Documents

Place text documents in `.txt` format in the directories specified in `documents_dir`. Each document will be:

- Loaded and assigned a unique ID
- imported into the system's database via the `/import` API endpoint
- Used for document assembly

### Ground Truth File

Create a JSONL file (one JSON object per line) containing evaluation scenarios. Each entry should follow this structure:

```json
{
  "id": "unique_identifier",
  "subject": "document subject/topic",
  "context": {
    "user_persona": "analyst_type",
    "language": "target_language",
    "etc" : "..."
  },
  "template": "template_id",
  "gold_documents": ["filename1.txt", "filename2.txt"], # not implemented yet
  "answer": ["Ground truth document content in markdown format"]
}
```

- **id**: Unique identifier for the evaluation scenario
- **subject**: The topic or subject for document assembly
- **context**: Optional context variables (e.g., user persona, language)
- **template**: Template ID to use for document assembly
- **gold_documents**: List of document filenames that should be used for this scenario (if `use_gold_documents` is true). Not implemented yet, for now every docs will be used as a base for the document.
- **answer**: Ground truth document content (can be a list of strings for multiple acceptable answers)

## Usage

### Running the Evaluation Pipeline

Execute the main evaluation script:

```bash
uv run -m src.eval.doc_assembler_eval.main --document_dir /path/to/documents
```

This will run all steps (importation, assembly, evaluation) by default.

### Command Line Arguments

- `--document_dir PATH`: Path(s) to document directories (required). Can specify multiple directories.
- `--import`: Run only the importation step (documents to API)
- `--generate`: Run only the document assembly step
- `--evaluate`: Run only the evaluation step

### Examples

```bash
# Run all steps with documents from a directory
uv run -m src.eval.doc_assembler_eval.main --document_dir /path/to/docs

# Run only importation
uv run -m src.eval.doc_assembler_eval.main --import --document_dir /path/to/docs

# Run only assembly (requires documents to be already imported)
uv run -m src.eval.doc_assembler_eval.main --generate --document_dir /path/to/docs

# Run only evaluation (requires documents to be already generated)
uv run -m src.eval.doc_assembler_eval.main --evaluate --document_dir /path/to/docs

# Run with multiple document directories
uv run -m src.eval.doc_assembler_eval.main --document_dir /path/to/docs1 /path/to/docs2
```

## Output

### Generated documents

documents are saved as markdown files in the `documents_dir` with filenames corresponding to their job IDs (e.g., `uuid.md`).

### Evaluation Results

Evaluation metrics are saved as JSON files in the `result_dir` with the following structure:

```json
{
  "rouge": {
    "rouge-1": 0.85,
    "rouge-2": 0.72,
    "rouge-l": 0.81
  },
  "bert": {
    "precision": 0.88,
    "recall": 0.84,
    "f1": 0.86
  },
  "g_eval_relevance": {
    "average": 4.2
  },
  "g_eval_consistency": {
    "average": 3.9
  },
  "total_documents_evaluated": 10
}
```

### Metrics Description

- **ROUGE**: Measures n-gram overlap between generated and ground truth text
- **BERTScore**: Semantic similarity using BERT embeddings
- **G-Eval**: LLM-based evaluation for relevance and consistency (1-5 scale)

## Workflow Details

### Document importation

- Documents are sent to `/import` endpoint in batches
- The system monitors importation status via `/import/{id}/status`
- Process waits until all documents are fully imported

### document assembly

- For each ground truth entry, multiple documents are generated using `/generate-document`
- Each assembly job is monitored via `/documents/{job_id}` endpoint
- documents are converted from structured sections to markdown format

### Evaluation Process

- Generated documents are compared against ground truth answers
- Multiple documents per ground truth are averaged
- Results are aggregated across all evaluation scenarios