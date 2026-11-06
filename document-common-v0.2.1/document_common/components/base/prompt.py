from pathlib import Path
from typing import Any, List, Optional, Union

import toml
import yaml
from haystack.dataclasses.chat_message import ChatMessage

from document_common.utils.logger import Logger
from document_common.schemas.prompt_template import PromptTemplate, PromptDict

class PromptProvider:
    """
    Singleton class to manage and retrieve prompts defined in YAML files.

    This class loads prompts from one or more folders and allows retrieval by prompt name.
    Prompts must follow a specific format (see `prompts/example.yaml` for reference).
    Once loaded, prompts can be accessed using `get_prompt("prompt_name")`.

    Example:
        provider = PromptProvider("path/to/prompt_folder")
        messages = provider.get_prompt("example_prompt")

    Notes:
        - Only one instance of PromptProvider exists (singleton pattern).
        - Each prompt name must be unique across all loaded folders.
        - You can re-initialize with another folder to load more prompts.
        - Use `overwrite_existing_prompt=False` to overwrite conflicting prompts.
    """

    _instance = None

    def __new__(
        cls,
        prompt_folder: Optional[Union[str, Path]] = None,
        overwrite_existing_prompt: bool = False,
    ):
        """
        Create or retrieve the singleton instance of PromptProvider.

        Args:
            prompt_folder (str | Path, optional): Path to a directory containing YAML prompt files.
                                                  Prompts from this folder will be loaded on instantiation.
            overwrite_existing_prompt (bool): If True, allows prompts with the same name to be overwritten.
                                              If False, raises an error on duplicate prompt names.

        Returns:
            PromptProvider: The singleton instance with loaded prompts (if a folder is provided).

        Raises:
            ValueError: If duplicate prompt names are found and `overwrite_existing_prompt` is False.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.prompts = {}
            cls._instance.logger = Logger().get_logger("PromptProvider")

        if prompt_folder:
            cls._instance._init(Path(prompt_folder), overwrite_existing_prompt)
        return cls._instance

    def _init(self, prompt_folder: Path, overwrite_existing_prompt: bool):
        self.logger = Logger().get_logger("PromptProvider")
        self.prompts: dict[str, List[ChatMessage]] = {}
        self._load_prompts_from_yaml_directory(prompt_folder, overwrite_existing_prompt)

    def _load_prompts_from_yaml_directory(
        self, directory_path: Path, overwrite_existing_prompt: bool = False
    ):
        if not directory_path.is_dir():
            raise ValueError(f"The specified path is not a directory: {directory_path}")

        prompts_loaded = []
        self.logger.info(f"Loading prompts from {directory_path}")
        for yaml_file in directory_path.rglob("*.yaml"):
            try:
                with open(yaml_file, "r") as file:
                    data = yaml.safe_load(file)
                    # Extract version from top level
                    version = data.get("version")
                    if version is None:
                        self.logger.warning(
                            f"No version field found in {yaml_file}, will set it to 1"
                        )
                        version = 1

                    for name, prompt_data in data.items():
                        if name == "version":
                            continue
                        prompts_loaded.append(name)
                        self.add_prompt(
                            name, prompt_data, version, overwrite_existing_prompt
                        )
            except toml.TomlDecodeError as e:
                error_msg = f"Error loading prompts from TOML file {yaml_file}: {e}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            except KeyError as e:
                error_msg = f"Missing key in TOML file {yaml_file}: {e}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        self.logger.info(f"Loaded prompts : {prompts_loaded}")

    def add_prompt(
        self,
        name: str,
        template: dict[str, Any],
        version: int,
        overwrite_existing_prompt: bool = False,
    ) -> None:
        if name in self.prompts:
            if not overwrite_existing_prompt:
                raise ValueError(f"Prompt '{name}' already exists.")
            else:
                self.logger.debug(
                    f"Prompt {name} already exists, it will be overwritten."
                )

        # Add version to template if not already present
        template_with_version = template.copy()
        if "version" not in template_with_version:
            template_with_version["version"] = version

        prompt_template = PromptTemplate(
            name=name, prompt=PromptDict(**template_with_version), version=version
        )

        messages = [ChatMessage.from_system(prompt_template.prompt.system)]

        if prompt_template.prompt.examples:
            for example in prompt_template.prompt.examples:
                role = example.role
                content = example.content
                if role == "user":
                    messages.append(ChatMessage.from_user(content))
                elif role == "assistant":
                    messages.append(ChatMessage.from_assistant(content))

        messages.append(ChatMessage.from_user(prompt_template.prompt.query))

        self.prompts[name] = messages

    def add_prompt_from_str(
        self, prompt: str, overwrite_existing_prompt: bool = False
    ) -> List[str]:
        """
        Add prompts from a YAML string, extracting version and skipping the version field in iteration.

        Args:
            prompt: YAML content string containing one or more prompts
            overwrite_existing_prompt: Whether to overwrite existing prompts

        Returns:
            List[str]: Names of prompts that were loaded
        """
        data = yaml.safe_load(prompt)
        prompts_loaded: list[str] = []

        # Extract version from top level, default to 1 if missing
        version = data.get("version")
        if version is None:
            self.logger.warning(
                "No version field found in YAML content, will set it to 1"
            )
            version = 1

        for name, prompt_data in data.items():
            if name == "version":
                continue
            self.add_prompt(name, prompt_data, version, overwrite_existing_prompt)
            prompts_loaded.append(name)
        self.logger.info(f"Loaded prompts : {prompts_loaded}")
        return prompts_loaded

    def get_prompt(self, prompt_name: str) -> List[ChatMessage]:
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found.")
        return self.prompts[prompt_name]