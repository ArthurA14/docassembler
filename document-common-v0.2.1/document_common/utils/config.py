import toml
import yaml
from pydantic import BaseModel


class AppConfig(BaseModel):
    checkpoint_dir: str
    saving_directory: str


class Config:
    """Singleton configuration loader that reads settings from a YAML file.

    Initialize with `Config(path)` to load, then use `Config.get()` to retrieve.
    """

    _instance = None
    config: AppConfig = None

    def __new__(cls, config_path: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(config_path)
        return cls._instance

    def _init(self, config_path: str):
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            self.config = AppConfig(**data)  # turn keys into attributes!
        except toml.TomlDecodeError as e:
            error_msg = f"Error loading the config from YAML file {config_path}: {e}"
            # self.logger.error(error_msg)
            raise ValueError(error_msg)

    @classmethod
    def get(cls):
        if cls._instance is None:
            raise RuntimeError("Config has not been initialized.")
        return cls._instance
