import argparse
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel
from document_common.utils.logger import Logger

from config import config
from src.eval.doc_assembler_eval.importer import Importer
from src.eval.doc_assembler_eval.document_evaluator import DocumentEvaluator
from src.eval.doc_assembler_eval.doc_assembler import DocAssembler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the document assembly evaluation pipeline"
    )
    parser.add_argument(
        "--import",
        dest="importation",
        action="store_true",
        help="Run only the importation step",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run only the document assembly step",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run only the evaluation step",
    )
    parser.add_argument(
        "--document_dir",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to document directories"
    )
    return parser.parse_args()


class EvalConfig(BaseModel):
    api_base_url: str
    documents_dir: List[str] | str
    documents_dir: str
    n_document_per_gt: int
    ground_truth_path: str
    result_dir: str

def main():
    log_level = os.getenv("LOG_LEVEL", "INFO")
    Logger().set_level(level=log_level)
    logger = Logger().get_logger("eval.doc_assembler_eval.main")

    try:
        args = parse_args()

        eval_config = config.get_current_config().evaluation

        rge_conf = EvalConfig.model_validate(eval_config.doc_assembler_eval)

        # Extract G_eval configuration if available
        g_eval_config = {}
        if hasattr(eval_config, 'G_eval') and eval_config.G_eval:
            g_eval_config = dict(eval_config.G_eval)
            logger.info(f"Loaded G_eval configuration: {g_eval_config}")
        else:
            logger.info("No G_eval configuration found, using defaults")

        Path(rge_conf.documents_dir).mkdir(exist_ok=True)
        Path(rge_conf.result_dir).mkdir(exist_ok=True)

        logger.info("Starting document assembly evaluation...")

        gt_document_mapping = None

        if args.document_dir:
            document_dir = args.document_dir
        else :
            document_dir = rge_conf.documents_dir

        # ------------------ Step 1: import ------------------
        if args.importation:
            logger.info("Step 1: importing files")
            try:
                importer = Importer(
                    api_base_url=rge_conf.api_base_url,
                    documents_dir=document_dir,
                )
                importer.run()
                logger.info("File importation completed successfully")
            except Exception as e:
                logger.error(f"Error during file importation: {e}", exc_info=True)
                raise

        # ------------------ Step 2: Generate ------------------
        if args.generate:
            logger.info("Step 2: Generating the documents")
            try:
                generator = DocAssembler(
                    api_base_url=rge_conf.api_base_url,
                    documents_dir=rge_conf.documents_dir,
                    ground_truth_path=rge_conf.ground_truth_path,
                    document_dir=document_dir,
                    n_document_per_gt=rge_conf.n_document_per_gt,
                )
                gt_document_mapping = generator.run()
                logger.info("document assembly completed successfully")
            except Exception as e:
                logger.error(f"Error during document assembly: {e}", exc_info=True)
                raise

        # ------------------ Step 3: Evaluate ------------------
        if args.evaluate:
            logger.info("Step 3: Evaluating the documents")
            try:
                evaluator = DocumentEvaluator(
                    ground_truth_file=rge_conf.ground_truth_path,
                    documents_dir=rge_conf.documents_dir,
                    result_dir=rge_conf.result_dir,
                    gt_document_mapping=gt_document_mapping,
                    g_eval_config=g_eval_config,
                )
                evaluator.run()
                logger.info("document evaluation completed successfully")
            except Exception as e:
                logger.error(f"Error during document evaluation: {e}", exc_info=True)
                raise

        logger.info("Pipeline finished successfully!")

    except Exception as e:
        logger.error(f"Fatal error in document assembly evaluation pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
