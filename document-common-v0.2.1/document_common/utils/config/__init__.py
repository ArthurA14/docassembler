import os

from .manager import ConfigManager, YamlConfigLoader, YamlConfigWriter


def create_config_manager(config_dir: str, default_config_file: str):
    """Create and return a single ConfigManager instance."""

    # Get configuration values from environment variables or use defaults
    config_dir = os.getenv("CONFIG_DIR") or config_dir
    default_config_file = os.getenv("DEFAULT_CONFIG_FILE") or default_config_file

    # Initialize the loader and writer
    config_loader = YamlConfigLoader()
    config_writer = YamlConfigWriter()

    # Initialize and return the ConfigManager
    config_manager = ConfigManager(
        config_directory=config_dir,
        loader=config_loader,
        writer=config_writer,
        default_config_file=default_config_file,
    )

    return config_manager
