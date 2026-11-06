
import argparse
import os

from dotenv import load_dotenv
from pydantic import BaseModel
from document_common.utils.logger import Logger

from config import config
from src.eval.MSRS_eval.eval import MSEval
from src.eval.MSRS_eval.eval_setup import MSEvalSetup
from src.eval.MSRS_eval.mlflow_eval import MLFlowEval


class EvalConfig(BaseModel):
    dataset_path: str
    local_dataset_path: str
    store_name: str
    result_output_path: str
    story_assembly_prompt_name: str
    meeting_assembly_prompt_name: str
    truncate: bool
    truncate_length: int

def main():
    """Main entry point for the MSRS evaluation setup"""
    parser = argparse.ArgumentParser(description="MSRS Evaluation Script")
    parser.add_argument("--setup", action="store_true", help="Run only the setup")
    parser.add_argument("--eval", action="store_true", help="Run only the evaluation")
    parser.add_argument("--oracle", action="store_true", help="Run the evaluation on oracle mode, llm is fed perfect documents")
    parser.add_argument("--rerank", action="store_true", help="Runs the evaluation with the reranker")
    parser.add_argument("--clean", action="store_true", help="Clean setup after running")
    parser.add_argument("--only_retrieval", action="store_true", help="Runs only the retrieval part of the evaluation (must be used with --eval)")
    parser.add_argument("--mlflow", action="store_true", help="Runs the evaluation on ML-Flow")

    args = parser.parse_args()

    # Default to both if no specific option
    if not (args.setup or args.eval or args.clean):
        args.setup = True
        args.eval = True
        args.clean = True
    
    log_level = os.getenv("LOG_LEVEL", "INFO")
    Logger().set_level(level=log_level)
    logger = Logger().get_logger("eval.main")

    load_dotenv()

    try:
        eval_config = config.get_current_config().evaluation
        eval_conf = EvalConfig.model_validate(eval_config.MSRS)
        embedder_config = eval_config.embedder_config

        # Extract G_eval configuration if available
        g_eval_config = {}
        if hasattr(eval_config, 'G_eval') and eval_config.G_eval:
            g_eval_config = dict(eval_config.G_eval)
            logger.info(f"Loaded G_eval configuration: {g_eval_config}")
        else:
            logger.info("No G_eval configuration found, using defaults")

        setup = None
        docs_setup = None

        if args.setup:
            # This sets up the evaluation
            # It downloads the MSRS dataset, converts the data into haystack Documents, embed and writes it to an "eval" collection
            setup = MSEvalSetup(
                dataset_path=eval_conf.dataset_path,
                store_name=eval_conf.store_name,
                local_dataset_path=eval_conf.local_dataset_path,
                embedder_config=embedder_config
            )
            docs_setup = setup.run()
            logger.info("MSRS evaluation setup completed successfully!")

        if args.eval:
            oracle = False
            if args.oracle:
                oracle = True
            rerank = False
            if args.rerank:
                rerank = True
            only_retrieval = False
            if args.only_retrieval:
                only_retrieval = True
            if args.mlflow:
                eval = MLFlowEval(
                    dataset_path=eval_conf.dataset_path,
                    local_dataset_path=eval_conf.local_dataset_path,
                    result_output_path=eval_conf.result_output_path,
                    store_name=eval_conf.store_name,
                    story_assembly_prompt_name=eval_conf.story_assembly_prompt_name,
                    meeting_assembly_prompt_name=eval_conf.meeting_assembly_prompt_name,
                    truncate=eval_conf.truncate,
                    truncate_length=eval_conf.truncate_length,
                    docs_setup=docs_setup,
                    g_eval_config=g_eval_config,
                    oracle_run=oracle,
                    with_rerank=rerank,
                    only_retrieval=only_retrieval
                )
            else:
                eval = MSEval(
                    dataset_path=eval_conf.dataset_path,
                    local_dataset_path=eval_conf.local_dataset_path,
                    result_output_path=eval_conf.result_output_path,
                    store_name=eval_conf.store_name,
                    story_assembly_prompt_name=eval_conf.story_assembly_prompt_name,
                    meeting_assembly_prompt_name=eval_conf.meeting_assembly_prompt_name,
                    truncate=eval_conf.truncate,
                    truncate_length=eval_conf.truncate_length,
                    docs_setup=docs_setup,
                    g_eval_config=g_eval_config,
                    oracle_run=oracle,
                    with_rerank=rerank,
                    only_retrieval=only_retrieval
                )
            eval.run()
            logger.info("MSRS evaluation completed successfully!")

        if args.clean and docs_setup is not None and setup is not None:
            setup.clean_setup(docs_setup)
            logger.info("Setup cleaned successfully!")
        elif args.clean:
            setup = MSEvalSetup(
                dataset_path=eval_conf.dataset_path,
                store_name=eval_conf.store_name,
                local_dataset_path=eval_conf.local_dataset_path,
                embedder_config=embedder_config,
            )
            setup.clean_setup()

    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise


if __name__ == "__main__":
    main()
