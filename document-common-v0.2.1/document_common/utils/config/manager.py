import os
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Union

import yaml
from rich import console

from .utils import DotDict, dotdict_representer

yaml.add_representer(DotDict, dotdict_representer)

ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def env_var_constructor(loader, node):
    """Constructor to replace ${VAR} with environment variables."""
    value = loader.construct_scalar(node)

    def replace_env_var(match):
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            raise ConfigError(f"Environment variable '{var_name}' not found")
        return env_value

    return ENV_VAR_PATTERN.sub(replace_env_var, value)


# Register the constructor with SafeLoader
yaml.SafeLoader.add_implicit_resolver("!ENV", ENV_VAR_PATTERN, None)
yaml.SafeLoader.add_constructor("!ENV", env_var_constructor)


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


class YamlConfigLoader:
    """YAML implementation of configuration loader."""

    def load(self, content: Union[str, TextIO]) -> Dict[str, Any]:
        """Load YAML configuration from the given content."""
        try:
            if isinstance(content, str):
                return yaml.load(content, Loader=yaml.SafeLoader)
            return yaml.load(content, Loader=yaml.SafeLoader)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML: {e}")


class YamlConfigWriter:
    """YAML implementation of configuration writer."""

    def write(self, config: Dict[str, Any], file_path: Path) -> None:
        """Write YAML configuration to the given path."""
        try:
            with open(file_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            raise ConfigError(f"Failed to write configuration: {e}")


class ConfigManager:
    """Loads, merges, and provides config files."""

    def __init__(
        self,
        config_directory: str,
        loader: YamlConfigLoader = YamlConfigLoader(),
        writer: YamlConfigWriter = YamlConfigWriter(),
        default_config_file: str = "core.yaml",
    ):
        """Initialize the ConfigManager with its dependencies."""
        self.config_directory = files(config_directory)
        self.loader = loader
        self.writer = writer
        self.default_config_file = default_config_file

        base_config = self.load_base_config()
        self.current_config = self.load_directory_configs(base_config)

        # First check if both configs are present
        has_embedding = "embedding" in self.current_config.keys()
        has_chunker = "chunker" in self.current_config.keys()

        if has_embedding and has_chunker:
            # Both configs exist, create individual configs first
            embedding_config = dict(
                self.current_config.embedding
            )  # Convert to regular dict to avoid serialization issues
            chunker_config = {
                "chunker_type": self.current_config.chunker_type,
                **dict(self.current_config.chunker),  # Convert to regular dict
            }

            # Merge the configs
            merged_config = {**embedding_config, **chunker_config}

            # Write the merged config to both paths
            self.write_config(
                DotDict(merged_config),
                files(self.current_config.embedding_config_dir).joinpath(
                    self.current_config.embedding_config_name
                ),  # TODO: Define common key
            )

        else:
            # Only one config exists, write separately
            if has_embedding:
                # Convert to plain dict to avoid serialization issues
                embedding_config = dict(self.current_config.embedding)
                self.write_config(
                    DotDict(embedding_config),
                    Path(self.current_config.embedding_config_path),
                )

            if has_chunker:
                # Convert to plain dict to avoid serialization issues
                chunker_config = {
                    "chunker_type": self.current_config.chunker_type,
                    **dict(self.current_config.chunker),
                }
                self.write_config(
                    DotDict(chunker_config),
                    files(self.current_config.chunker_config_dir).joinpath(
                        self.current_config.chunker_config_name
                    ),  # TODO: Define common key
                )

    def load_base_config(self) -> DotDict:
        """Loads and returns the base config as a dictionary."""
        return self.load_config_from_path(
            self.config_directory.joinpath(self.default_config_file)
        )

    def load_config_from_path(self, path: Path) -> DotDict:
        """Loads config file from the given path and returns it as a dictionary."""
        try:
            with open(path, "r") as f:
                return DotDict(self.loader.load(f))
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {path}")
        except Exception as e:
            raise ConfigError(f"Failed to load configuration from {path}: {e}")

    def merge_with_override(self, override_file: str) -> None:
        """Merges the current config with the overrides from the specified file."""
        try:
            override_path = self.config_directory / f"{override_file}.yaml"
            override_config = self.load_config_from_path(override_path)
            self.current_config = self.merge_configs(
                self.current_config, override_config
            )
        except ConfigError as e:
            raise ConfigError(f"Failed to merge with override '{override_file}': {e}")

    def merge_configs(self, base_config: DotDict, override_config: DotDict) -> DotDict:
        """Merges two config dictionaries recursively."""
        merged_config = base_config.copy()
        for key, value in override_config.items():
            if (
                key in merged_config
                and isinstance(merged_config[key], dict)
                and isinstance(value, dict)
            ):
                merged_config[key] = self.merge_configs(merged_config[key], value)
            else:
                merged_config[key] = value
        return DotDict(merged_config)

    def get_current_config(self) -> DotDict:
        """Returns the currently loaded config as a dictionary."""
        return self.current_config

    def save_config(self, file_path: Path) -> None:
        """Saves the current config to the specified file path."""
        try:
            self.writer.write(self.current_config, file_path)
        except Exception as e:
            raise ConfigError(f"Failed to save configuration to {file_path}: {e}")

    def write_config(self, config: DotDict, file_path: Path) -> None:
        """Writes a DotDict to the specified file path."""
        try:
            self.writer.write(config, file_path)
        except Exception as e:
            raise ConfigError(f"Failed to save configuration to {file_path}: {e}")

    def load_directory_configs(
        self, config_dict: DotDict, base_path: Optional[Path] = None
    ) -> DotDict:
        """Recursively loads configurations from directories specified in the config.

        This method processes the configuration dictionary, looking for values
        that point to directories. When a directory is found, it loads all YAML files
        from that directory and replaces the directory path with the loaded configurations.

        Args:
            config_dict: The configuration dictionary to process
            base_path: The base path for resolving relative paths (defaults to config_directory)

        Returns:
            The processed configuration dictionary with directories replaced by their contents
        """
        if base_path is None:
            base_path = self.config_directory

        result = config_dict.copy()

        # Process each key-value pair in the configuration
        for key, value in list(config_dict.items()):
            # Recursively process nested dictionaries
            if isinstance(value, dict):
                result[key] = self.load_directory_configs(DotDict(value), base_path)

            # Process string values that might be directory paths
            elif isinstance(value, str) and value.startswith("./"):
                dir_path = base_path / value[2:]  # Remove './' prefix

                # Check if the path exists and is a directory
                if dir_path.exists() and dir_path.is_dir():
                    # Look for a selector key (e.g., {key}_type)
                    selector_key = f"{key}_type"
                    if selector_key in config_dict:
                        # Get the selector value (e.g., 'late')
                        selector_value = config_dict[selector_key]
                        # Construct the path to the specific file
                        file_path = dir_path / f"{selector_value}.yaml"

                        if file_path.exists():
                            # Load the specific file
                            file_config = self.load_config_from_path(file_path)
                            # Recursively process the loaded config with the new base path
                            result[key] = self.load_directory_configs(
                                file_config, dir_path
                            )
                        else:
                            raise ConfigError(
                                f"Selected configuration file not found: {file_path}"
                            )
                    else:
                        # No selector key found, load all YAML files in the directory
                        merged_config = {}

                        # Get all YAML files in the directory
                        yaml_files = [
                            f
                            for f in dir_path.iterdir()
                            if f.is_file()
                            and (
                                f.suffix.lower() == ".yaml"
                                or f.suffix.lower() == ".yml"
                            )
                        ]

                        if not yaml_files:
                            raise ConfigError(
                                f"No YAML files found in directory: {dir_path}"
                            )

                        # Load and merge all YAML files
                        for yaml_file in yaml_files:
                            file_config = self.load_config_from_path(yaml_file)
                            merged_config = self.merge_configs(
                                DotDict(merged_config), file_config
                            )

                        # Recursively process the merged config with the new base path
                        result[key] = self.load_directory_configs(
                            DotDict(merged_config), dir_path
                        )

        return DotDict(result)

    def pprint(self):
        c = console.Console()
        c.log(
            self.current_config,
        )
