import os
from typing import Any, Dict, List, Optional, Type, Union, cast

from haystack import Document, component
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses.chat_message import ChatMessage
from jinja2.exceptions import UndefinedError
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChoiceLogprobs
from pydantic import BaseModel, ValidationError

from config import config
from document_common.components.base.prompt import PromptProvider
from document_common.utils.logger import Logger
from document_common.utils.prettyprinters import build_chat_messages_str


@component
class ResponseGenerator:
    """
    Wraps OpenAI's `responses.parse` for structured output.
    Can optionally use the Batch API for processing.
    """

    def __init__(
        self,
        base_model: Optional[Type[BaseModel]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        base_url: Optional[str] = None,
        top_p: Optional[float] = None,
    ):
        """
        Initializes the ResponseOpenAIGenerator.

        Args:
            model: The model name to use (e.g., "gpt-4.1-nano-2025-04-14").
            api_key: OpenAI API key.
            use_batch_api: Whether to use the Batch API for assembly.
            batch_endpoint: The OpenAI API endpoint for batch processing (default: "/v1/responses").
            batch_completion_window: The completion window for batch jobs (default: "24h").
            batch_polling_interval: Interval in seconds to poll for batch job status.
            temperature: Sampling temperature for assembly.
            max_output_tokens: Maximum number of tokens to generate.
            timeout: Timeout for synchronous API calls.
        """
        self.logger = Logger().get_logger("assembly.sync_handler")
        llm_config = config.get_current_config().llm
        self.model = model or llm_config.model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or llm_config.api_key
        self.openai_client = OpenAI(
            api_key=self.api_key, 
            base_url=base_url or llm_config.base_url, 
            timeout=timeout or llm_config.timeout
        )
        self.temperature = temperature if temperature is not None else llm_config.temperature
        self.max_output_tokens = max_output_tokens if max_output_tokens is not None else llm_config.max_output_tokens
        self.timeout = timeout or llm_config.timeout
        self.base_model = base_model
        self.top_p = top_p if top_p is not None else llm_config.top_p

        self.logger.debug(
            f"Initialized ResponseGenerator with model: {model}, temperature: {temperature}, max_tokens: {max_output_tokens}"
        )
        self.logger.debug(f"ResponseGenerator config - base_url: {base_url}, timeout: {timeout}, top_p: {top_p}")
        if base_model:
            self.logger.debug(f"Using structured output with BaseModel: {base_model.__name__}")

    def run(
        self, prompt: List[ChatMessage], base_model: Optional[Type[BaseModel]] = None, log_prob: bool = False
    ) -> Dict[str, Union[BaseModel, Optional[str], ChoiceLogprobs]]:
        """
        messages: list of ChatMessage to send as `input`
        text_format: the prompt template to send as `text_format`
        returns dict with key 'structured_reply'
        """
        self.logger.debug(f"Starting assembly with {len(prompt)} messages")
        self.logger.debug(f"Input prompt: {prompt}")

        openai_formatted_messages = [message.to_openai_dict_format() for message in prompt]

        text_format = base_model or self.base_model

        if text_format:
            self.logger.debug(f"Using structured assembly with BaseModel: {text_format.__name__}")
            self.logger.debug(f"Schema: {text_format.model_json_schema()}")

            extra_body = {}
            if "mistral" not in self.model.lower() and "ministral" not in self.model.lower():
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=openai_formatted_messages,  # type: ignore
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": text_format.__name__,
                        "schema": text_format.model_json_schema(),
                    },
                },
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,  # Note: some models use max_completion_tokens instead
                timeout=self.timeout,
                extra_body=extra_body,
            )

            self.logger.debug(f"Received response from OpenAI API for model: {self.model}")
            self.logger.debug(f"Raw response: {response}")

            content = response.choices[0].message.content
            try:
                self.logger.debug(f"Raw response content: {content}")
                parsed_response = text_format.model_validate_json(content)  # type: ignore
                self.logger.debug(f"Successfully parsed response into {text_format.__name__}")
                self.logger.debug(f"Parsed responses: {parsed_response}")
                return {
                    "reply": parsed_response,
                }
            except Exception as e:
                self.logger.error(f"Failed to parse response into {text_format.__name__}: {e}")
                self.logger.debug(f"Failed content: {content}")
                raise e

        else:
            self.logger.info("Using unstructured text assembly")
            openai_formatted_messages = cast(list[ChatCompletionMessageParam], openai_formatted_messages)

            extra_body = {}
            if "mistral" not in self.model.lower() and "ministral" not in self.model.lower():
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

            response: ChatCompletion = self.openai_client.chat.completions.create(
                model=self.model,
                messages=openai_formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,  # Note: some models use max_completion_tokens instead
                logprobs=True if log_prob else None,
                top_logprobs=5 if log_prob else None,
                top_p=self.top_p,
                extra_body=extra_body,
            )

            self.logger.debug(f"Received text response from OpenAI API for model: {self.model}")
            self.logger.debug(f"Raw response: {response}")

            reply_content = response.choices[0].message.content
            self.logger.debug(f"Generated text: {reply_content}")
            if log_prob:
                return {"reply": reply_content, "log_prob": response.choices[0].logprobs}
            else:
                return {"reply": reply_content}


class AsyncResponseGenerator:
    """
    Async wrapper for OpenAI's chat completions with structured output support.
    """

    def __init__(
        self,
        base_model: Optional[Type[BaseModel]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        base_url: Optional[str] = None,
        top_p: Optional[float] = None,
    ):
        self.logger = Logger().get_logger("assembly.async_handler")
        llm_config = config.get_current_config().llm
        self.model = model or llm_config.model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or llm_config.api_key
        self.async_openai_client = AsyncOpenAI(
            api_key=self.api_key, 
            base_url=base_url or llm_config.base_url, 
            timeout=timeout or llm_config.timeout
        )
        self.temperature = temperature if temperature is not None else llm_config.temperature
        self.max_output_tokens = max_output_tokens if max_output_tokens is not None else llm_config.max_output_tokens
        self.timeout = timeout or llm_config.timeout
        self.base_model = base_model
        self.top_p = top_p if top_p is not None else llm_config.top_p

        self.logger.info(
            f"Initialized AsyncResponseGenerator with model: {model}, temperature: {temperature}, max_tokens: {max_output_tokens}"
        )
        self.logger.debug(f"AsyncResponseGenerator config - base_url: {base_url}, timeout: {timeout}, top_p: {top_p}")
        if base_model:
            self.logger.info(f"Using structured output with BaseModel: {base_model.__name__}")

    async def run_async(
        self, prompt: List[ChatMessage], base_model: Optional[Type[BaseModel]] = None
    ) -> Dict[str, Union[BaseModel, Optional[str]]]:
        self.logger.info(f"Starting async assembly request with {len(prompt)} messages")
        self.logger.debug(f"Incoming prompt: {prompt}")

        openai_formatted_messages = [message.to_openai_dict_format() for message in prompt]

        text_format = base_model or self.base_model
        is_structured = text_format is not None

        if is_structured:
            self.logger.info(f"Using structured assembly with model: {text_format.__name__}")
        else:
            self.logger.info("Using unstructured text assembly")

        if text_format:
            self.logger.info(f"Using structured assembly with BaseModel: {text_format.__name__}")
            self.logger.debug(f"Schema: {text_format.model_json_schema()}")
            self.logger.info(f"Calling async OpenAI API for structured assembly with model: {self.model}")

            extra_body = {}
            if "mistral" not in self.model.lower() and "ministral" not in self.model.lower():
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

            response = await self.async_openai_client.chat.completions.create(
                model=self.model,
                messages=openai_formatted_messages,  # type: ignore
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": text_format.__name__, "schema": text_format.model_json_schema()},
                },
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                timeout=self.timeout,
                extra_body=extra_body,
            )

            self.logger.info(f"Received async response from OpenAI API for model: {self.model}")
            self.logger.debug(f"Raw async response: {response}")

            content = response.choices[0].message.content
            try:
                self.logger.debug(f"Raw async response content: {content}")
                parsed_response = text_format.model_validate_json(content)  # type: ignore
                self.logger.info(f"Successfully parsed async response into {text_format.__name__}")
                self.logger.debug(f"Parsed async responses: {parsed_response}")
                return {"reply": parsed_response}
            except Exception as e:
                self.logger.error(f"Failed to parse async response into {text_format.__name__}: {e}")
                self.logger.debug(f"Failed async content: {content}")
                raise e
        else:
            openai_formatted_messages = cast(list[ChatCompletionMessageParam], openai_formatted_messages)

            self.logger.info(f"Calling async OpenAI API for text assembly with model: {self.model}")

            extra_body = {}
            if "mistral" not in self.model.lower() and "ministral" not in self.model.lower():
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

            response = await self.async_openai_client.chat.completions.create(
                model=self.model,
                messages=openai_formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                top_p=self.top_p,
                extra_body=extra_body,
            )

            self.logger.info(f"Received async text response from OpenAI API for model: {self.model}")
            self.logger.debug(f"Raw async response: {response}")

            reply_content = response.choices[0].message.content
            self.logger.debug(f"Generated async text: {reply_content}")
            return {"reply": reply_content}


@component
class StructuredAssembly:
    """
    A structured LLM assembly component that wraps a `ResponseOpenAIGenerator`
    and integrates prompt templating, fixed inputs, and structured output validation.

    This component builds a prompt using a named template (`prompt_name`), merges
    inputs from multiple sources (`document`, `inputs`, and `fixed_inputs`), and
    generates a structured response using the provided `base_model`.

    ### Key Features:
    - Uses `ChatPromptBuilder` with prompt templates registered in `PromptProvider`.
    - Supports checkpointing via optional `Checkpointer` to cache and reuse results.
    - Merges `fixed_inputs` (from init) and `inputs` (from run) for dynamic prompt construction.
    - Validates that the generated response matches the expected `BaseModel` type.
    - Logs full prompt and output for traceability.

    ### Inputs:
    - **document** (`Optional[Union[List[Document], Document]]`): Single or list of documents to be used in prompt.
    - **inputs** (`dict`): Additional variables for the prompt template.
    - **prompt_name** (`Optional[str]`): Overrides the default prompt name provided at init.
    - **base_model** (`Optional[Type[BaseModel]]`): Overrides the base model provided at init.

    ### Output:
    - **structured_reply** (`BaseModel`): The structured and validated response generated by the LLM.

    ### Raises:
    - `ValidationError`: If the generated response doesn't match the expected `BaseModel` structure.
    - `UndefinedError`: If a variable required by the prompt is missing.
    - `Exception`: For any other assembly or runtime errors.
    """

    def __init__(
        self,
        prompt_name: str,
        base_model: Type[BaseModel],
        fixed_inputs: Optional[Dict[str, Any]] = None,
        use_batch_api: bool = False
    ):
        self.logger = Logger().get_logger("assembly.structured")
        self.prompt_name = prompt_name
        self.base_model = base_model
        self.fixed_inputs = fixed_inputs or {}
        self.use_batch_api = use_batch_api  # Store for potential direct use or inspection

        self.generator = ResponseGenerator(base_model=self.base_model)

        self.logger.debug(
            f"Initialized Structuredassembly with prompt_name: {prompt_name}, base_model: {base_model.__name__}"
        )
        self.logger.debug(f"Structuredassembly config - use_batch_api: {use_batch_api}, fixed_inputs: {fixed_inputs}")

    @component.output_types(structured_reply=BaseModel)
    def run(
        self,
        documents: Optional[Union[List[Document], Document]] = None,
        inputs: dict[str, Any] = {},
        prompt_name: Optional[str] = None,
        base_model: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, BaseModel]:

        if not prompt_name:
            prompt_name = self.prompt_name
            self.logger.debug(f"Using default prompt_name: {prompt_name}")

        if not base_model:
            base_model = self.base_model
            self.logger.debug(f"Using default base_model: {base_model.__name__}")

        self.prompt_builder = ChatPromptBuilder(template=PromptProvider().get_prompt(prompt_name))

        if isinstance(documents, Document):
            documents = [documents]
            self.logger.debug(f"Converted single document to list: {len(documents)} document(s)")

        # Ensure fixed_inputs and inputs are properly merged
        template_vars = {"documents": documents}

        if self.fixed_inputs:
            template_vars.update(self.fixed_inputs)
            self.logger.debug(f"Merged fixed inputs: {self.fixed_inputs}")

        if inputs:
            template_vars.update(inputs)
            self.logger.debug(f"Merged dynamic inputs: {inputs}")

        self.logger.info(
            f"Structure assembly. Prompt name: {self.prompt_name} BaseModel: {base_model.__name__} Variables: {list(template_vars.keys())}"
        )
        self.logger.debug(f"Input documents: {documents}")
        self.logger.debug(f"Input variables: {inputs}")
        self.logger.debug(f"Final variables: {template_vars}")

        try:
            self.logger.debug("Building prompt from template")
            prompt_messages = self.prompt_builder.run(template_variables=template_vars)["prompt"]
            self.logger.debug(f"Generated prompt messages:\n{build_chat_messages_str(prompt_messages)}")

            llm_answer = self.generator.run(prompt_messages, base_model=base_model)

        except ValidationError as e:
            message = f"The output of prompt '{self.prompt_name}' could not be match with the model '{base_model.__name__}'. There is probably a problem with the prompt. Details: {e}"
            self.logger.error(message)
            self.logger.debug(f"Template variables that caused validation error: {template_vars}")
            raise ValueError(message) from e

        except UndefinedError as e:
            message = f"Error while loading prompt {self.prompt_name}, a variable specified in the prompt is probably missing: {e}"
            self.logger.error(message)
            self.logger.debug(f"Template variables when undefined error occurred: {template_vars}")
            raise ValueError(message) from e

        except Exception as e:
            message = f"Error while generating '{base_model.__name__}': {e}"
            self.logger.error(message)
            self.logger.debug(f"Context when assembly failed - prompt: {prompt_messages}")
            raise Exception(message) from e

        generated_reply = llm_answer["reply"]

        self.logger.info(f"Successfully generated structured output for {base_model.__name__}")
        self.logger.debug(f"Generated output:\n{generated_reply}")

        if not isinstance(generated_reply, BaseModel):
            error_msg = f"While trying to generate a BaseModel, the model generated a {type(generated_reply)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        return {"structured_reply": generated_reply}


@component
class TextAssembly:
    """
    A simple component for generating plain text responses from a prompt template
    using a language model. This class is ideal for quick, unstructured assemblies
    that do not require response parsing or validation into a `BaseModel`.

    ### Key Features:
    - Retrieves a prompt template using `prompt_name` from `PromptProvider`.
    - Accepts both single or multiple documents as input context.
    - Supports `fixed_inputs` defined at instantiation, merged with dynamic `inputs`.
    - Uses `ResponseOpenAIGenerator` to generate the response.
    - Logs both the constructed prompt and the final output.

    ### Inputs:
    - **documents** (`Optional[Union[List[Document], Document]]`): Input document(s)
      to provide context for the prompt. Can be a single document or a list.
    - **inputs** (`Optional[dict]`): Additional template variables for dynamic prompt assembly.
    - **prompt_name** (`Optional[str]`): Overrides the default prompt name provided at init.

    ### Output:
    - **llm_answer** (`str`): The raw text response generated by the LLM.

    ### Raises:
    - `UndefinedError`: If a required template variable is missing in the inputs.
    - `Exception`: For general assembly errors (e.g., LLM failure, API issues).
    """

    def __init__(
        self,
        prompt_name: str,
        fixed_inputs: Optional[dict[str, str]] = None,
    ):
        self.logger = Logger().get_logger("assembly.text")
        self.prompt_name = prompt_name
        self.generator = ResponseGenerator()
        self.fixed_inputs = fixed_inputs or {}

        self.logger.debug(f"Initialized Textassembly with prompt_name: {prompt_name}")
        self.logger.debug(f"Textassembly config - fixed_inputs: {fixed_inputs}")

    @component.output_types(llm_answer=str)
    def run(
        self,
        documents: Optional[Union[List[Document], Document]] = None,
        inputs: Optional[dict[str, Any]] = None,
        prompt_name: Optional[str] = None,
        initial_query: str = "",
    ) -> dict[str, str]:
        self.initial_query = initial_query
        self.logger.info(f"Starting text assembly with prompt: {prompt_name or self.prompt_name}")
        self.logger.debug(f"Input documents: {documents}")
        self.logger.debug(f"Input variables: {inputs}")
        self.logger.debug(f"Initial query: {initial_query}")

        if not prompt_name:
            prompt_name = self.prompt_name
            self.logger.debug(f"Using default prompt_name: {prompt_name}")

        self.prompt_builder = ChatPromptBuilder(template=PromptProvider().get_prompt(prompt_name))

        template_vars: dict[str, Any] = {}
        if isinstance(documents, list):
            template_vars = {"documents": documents}
            self.logger.debug(f"Using documents list format: {len(documents)} document(s)")
        else:
            template_vars = {"document": documents}
            self.logger.debug("Using single document format")

        if self.fixed_inputs:
            template_vars.update(self.fixed_inputs)
            self.logger.debug(f"Merged fixed inputs: {self.fixed_inputs}")
        if inputs:
            template_vars.update(inputs)
            self.logger.debug(f"Merged dynamic inputs: {inputs}")

        self.logger.debug(f"Final template variables: {template_vars}")

        try:
            self.logger.debug("Building text prompt from template")
            prompt_messages: List[ChatMessage] = self.prompt_builder.run(template_variables=template_vars)["prompt"]
            self.logger.debug(f"Generated prompt messages:\n{build_chat_messages_str(prompt_messages)}")

            llm_answer = self.generator.run(prompt_messages)["reply"]
            llm_answer = cast(str, llm_answer)

        except UndefinedError as e:
            message = (
                f"Error while loading prompt {prompt_name}, a variable specified in the prompt is probably missing: {e}"
            )
            self.logger.error(message)
            self.logger.debug(f"Template variables when undefined error occurred: {template_vars}")
            raise ValueError(message) from e

        except Exception as e:
            message = f"Error while generating with prompt '{prompt_name}': {e}"
            self.logger.error(message)
            self.logger.debug(f"Context when assembly failed - prompt: {prompt_messages}")
            raise Exception(message) from e

        self.logger.debug(f"Generated text output:\n{llm_answer}")
        return {"llm_answer": llm_answer}
