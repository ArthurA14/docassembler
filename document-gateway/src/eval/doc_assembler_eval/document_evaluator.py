import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from document_common.components.base.prompt import PromptProvider
from document_common.utils.logger import Logger

from config import config
from src.eval.doc_assembler_eval.models import GroundTruth
from src.eval.utils.assembly_evaluator import AssemblyEvaluator


class DocumentEvaluator:
    """Class to evaluate documents using ROUGE, BERT, and G-EVAL metrics."""

    def __init__(
        self,
        ground_truth_file: str,
        documents_dir: str,
        result_dir: str,
        gt_document_mapping_file: Optional[str] = None,
        gt_document_mapping: Optional[Dict[str, List[str]]] = None,
        g_eval_config: Optional[Dict[str, Any]] = None,
    ):
        self.ground_truth_file = ground_truth_file
        self.documents_dir = documents_dir
        self.gt_document_mapping_file = gt_document_mapping_file
        self.gt_document_mapping = gt_document_mapping
        self.result_dir = result_dir
        self.g_eval_config = g_eval_config

        self.logger = Logger().get_logger("document_evaluator")

        PromptProvider(prompt_folder=config.get_current_config().others.prompt_folder, overwrite_existing_prompt=True)

    def load_ground_truth(self) -> List[GroundTruth]:
        """Load ground truth from JSONL file."""
        return GroundTruth.load_ground_truth(Path(self.ground_truth_file))

    def load_mapping(self) -> Dict[str, List[str]]:
        """Load ground truth to document mapping."""
        if self.gt_document_mapping is not None:
            # Use the provided mapping directly
            self.logger.info("Using provided ground truth to document mapping")
            return self.gt_document_mapping

        if self.gt_document_mapping_file is None:
            # Auto-discover the mapping file
            documents_dir_path = Path(self.documents_dir)
            json_files = list(documents_dir_path.glob("*.json"))

            if len(json_files) == 0:
                raise FileNotFoundError(f"No .json files found in {self.documents_dir} for ground truth to document mapping")
            elif len(json_files) > 1:
                raise ValueError(f"Multiple .json files found in {self.documents_dir}: {[f.name for f in json_files]}. Please specify gt_document_mapping_file explicitly.")
            else:
                self.gt_document_mapping_file = json_files[0]
                self.logger.info(f"Auto-discovered mapping file: {self.gt_document_mapping_file}")
        else:
            self.gt_document_mapping_file = Path(self.gt_document_mapping_file)

        with open(self.gt_document_mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_document(self, document_path: str) -> str:
        """Load document content from markdown file."""
        with open(document_path, 'r', encoding='utf-8') as f:
            return f.read()

    def run(self) -> None:
        """Run the evaluation process."""

        # Load data
        self.logger.info("Loading ground truth data...")
        try:
            ground_truth_list = self.load_ground_truth()
            ground_truth = {str(gt.id): gt for gt in ground_truth_list}
            self.logger.info(f"Loaded {len(ground_truth_list)} ground truth entries")
        except Exception as e:
            self.logger.error(f"Error loading ground truth data: {e}", exc_info=True)
            raise

        self.logger.info("Loading mapping data...")
        try:
            mapping = self.load_mapping()
            self.logger.info(f"Loaded mapping with {len(mapping)} entries")
        except Exception as e:
            self.logger.error(f"Error loading mapping data: {e}", exc_info=True)
            raise

        self.logger.info("Starting evaluation...")

        # For each ground truth ID
        predictions: List[str] = []
        references: List[List[str]] = []
        for gt_id, document_uuids in mapping.items():
            if gt_id not in ground_truth:
                self.logger.warning(f"Ground truth ID {gt_id} in file {self.gt_document_mapping_file} not found in ground truth file {self.ground_truth_file}")
                continue

            gt_obj = ground_truth[gt_id]
            gt_answers = gt_obj.answer

            # For each document UUID mapped to this GT
            for document_uuid in document_uuids:
                document_path = os.path.join(self.documents_dir, f"{document_uuid}.md")
                if not os.path.exists(document_path):
                    self.logger.warning(f"document file {document_path} not found")
                    continue

                self.logger.debug(f"Evaluating document {document_uuid} against GT {gt_id}")

                # Load document content
                try:
                    document_content = self.load_document(document_path)
                    self.logger.debug(f"Loaded document content ({len(document_content)} characters)")
                except Exception as e:
                    self.logger.error(f"Error loading document {document_path}: {e}", exc_info=True)
                    continue
                
                predictions.append(document_content)
                references.append(gt_answers)
        
        evaluator = AssemblyEvaluator(g_eval_config=self.g_eval_config)
        final_scores = evaluator.run(predictions=predictions, references=references)

        # Generate output filename
        output_uuid = str(uuid.uuid4())
        output_path = os.path.join(self.result_dir, f"{output_uuid}.json")

        # Ensure result_dir exists
        os.makedirs(self.result_dir, exist_ok=True)

        # Write results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_scores, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Evaluation complete. Results saved to {output_path}")
