import math
from typing import Any, Dict, List, Optional, TypedDict, cast

from haystack import component
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses.chat_message import ChatMessage
from openai.types.chat.chat_completion import ChoiceLogprobs
from document_common.components.base.assembly import ResponseGenerator
from document_common.components.base.prompt import PromptProvider
from document_common.utils.logger import Logger


class GEvalResult(TypedDict):
    raw_answer: Optional[str]
    rating: float
    probs: Dict[str, float]


@component
class GEval:
    """
    GEval component:
    - Based on a prompt name gets the logprob of token "1" to N (ex: "5") and computes weighted average based on softmax of logprobs
    - In short returns rating from LLM as a Judge prompt

    Output:
    {
        "raw_answer": str | None,
        "rating": float | None,
        "probs": dict[str, float]
    }
    """

    def __init__(
        self,
        prompt_name: Optional[str] = None,
        n: int = 5,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        base_url: Optional[str] = None,
        top_p: Optional[float] = None,
    ):
        self.logger = Logger().get_logger("assembly.geval")

        # Initialize ResponseGenerator with optional parameters
        # Only pass parameters that are not None to avoid type errors
        generator_params = {}
        if model is not None:
            generator_params["model"] = model
        if api_key is not None:
            generator_params["api_key"] = api_key
        if temperature is not None:
            generator_params["temperature"] = temperature
        if max_output_tokens is not None:
            generator_params["max_output_tokens"] = max_output_tokens
        if timeout is not None:
            generator_params["timeout"] = timeout
        if base_url is not None:
            generator_params["base_url"] = base_url
        if top_p is not None:
            generator_params["top_p"] = top_p

        self.generator = ResponseGenerator(**generator_params)

        self.n = n
        self.prompt_name = prompt_name

        self.logger.info(f"Initialized GEval with n={n}")
        if model or base_url:
            self.logger.info(f"Using custom model configuration: model={model}, base_url={base_url}")
        else:
            self.logger.info("Using default model configuration")

    @component.output_types(raw_answer=Optional[str], rating=Optional[float], probs=Dict[str, float])
    def run(self, variables: dict[str, Any], prompt_name: Optional[str] = None) -> GEvalResult:
        if not prompt_name:
            if self.prompt_name:
                prompt_name = self.prompt_name
            else:
                raise ValueError("To perform a G Evaluation you have to provide a prompt.")
        
        self.logger.info("Starting GEval run")
        self.logger.debug(f"Prompt messages: {prompt_name}")

        self.prompt_builder = ChatPromptBuilder(template=PromptProvider().get_prompt(prompt_name))
        try:
            prompt_messages: List[ChatMessage] = self.prompt_builder.run(template_variables=variables)["prompt"]
            answer = self.generator.run(prompt_messages, log_prob=True)
        except Exception as e:
            self.logger.error(f"Error calling ResponseGenerator: {e}", exc_info=True)
            raise

        raw_answer = cast(Optional[str], answer.get("reply"))
        log_prob_obj = answer.get("log_prob")

        # If no logprobs → no metric
        if not log_prob_obj or not isinstance(log_prob_obj, ChoiceLogprobs):
            self.logger.warning("No logprobs found in response.")
            raise Exception("No logprob found in response.")

        try:
            # Extract logprobs
            logps: Dict[str, float] = {}

            values = {str(k): k for k in range(1, self.n + 1)}

            for tk in log_prob_obj.content[0].top_logprobs: # type: ignore
                if tk.token in values.keys():
                    logps[tk.token] = tk.logprob

            if not logps:
                self.logger.warning("No relevant tokens (1..n) found in logprobs.")
                raise Exception(f"Incorrect logprob found in response. Found {log_prob_obj.content[0].top_logprobs}") # type: ignore

            # Compute softmax
            m = max(logps.values())
            exps = {k: math.exp(v - m) for k, v in logps.items()}
            Z = sum(exps.values())
            probs = {k: val / Z for k, val in exps.items()}

            # Compute rating
            rating = sum(probs[k] * values[k] for k in probs)

            self.logger.info(f"Computed rating: {rating}")
            self.logger.debug(f"Probability distribution: {probs}")

            return GEvalResult(raw_answer=raw_answer, rating=rating, probs=probs)

        except Exception as e:
            self.logger.error(f"Error computing rating: {e}", exc_info=True)
            raise
