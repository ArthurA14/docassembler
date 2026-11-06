import pytest
import tempfile
import os
from pathlib import Path

from document_common.utils.config.manager import (
    ConfigError,
    YamlConfigLoader,
    YamlConfigWriter,
    ConfigManager,
    env_var_constructor,
)
from document_common.utils.config.utils import DotDict


class TestYamlConfigLoader:
    def test_load_simple_dict(self):
        loader = YamlConfigLoader()
        yaml_str = "key: value\nnumber: 42"
        result = loader.load(yaml_str)
        assert result == {"key": "value", "number": 42}

    def test_load_nested_dict(self):
        loader = YamlConfigLoader()
        yaml_str = "outer:\n  inner: value\n  number: 3"
        result = loader.load(yaml_str)
        assert result["outer"]["inner"] == "value"
        assert result["outer"]["number"] == 3

    def test_load_invalid_yaml(self):
        loader = YamlConfigLoader()
        with pytest.raises(ConfigError):
            loader.load("key: [invalid yaml {{{")

    def test_load_with_list(self):
        loader = YamlConfigLoader()
        yaml_str = "items:\n  - a\n  - b\n  - c"
        result = loader.load(yaml_str)
        assert result["items"] == ["a", "b", "c"]


class TestYamlConfigWriter:
    def test_write_yaml(self, tmp_path):
        writer = YamlConfigWriter()
        config = {"key": "value", "nested": {"a": 1}}
        file_path = tmp_path / "config.yaml"
        writer.write(config, file_path)
        assert file_path.exists()
        content = file_path.read_text()
        assert "key: value" in content

    def test_write_dotdict(self, tmp_path):
        writer = YamlConfigWriter()
        config = DotDict({"a": 1, "b": DotDict({"c": 2})})
        file_path = tmp_path / "config.yaml"
        writer.write(config, file_path)
        assert file_path.exists()


class TestConfigManager:
    def test_merge_configs_simple(self):
        base = DotDict({"a": 1, "b": 2})
        override = DotDict({"b": 3, "c": 4})
        manager = ConfigManager.__new__(ConfigManager)
        result = manager.merge_configs(base, override)
        assert result["a"] == 1
        assert result["b"] == 3
        assert result["c"] == 4

    def test_merge_configs_nested(self):
        base = DotDict({"a": {"x": 1, "y": 2}})
        override = DotDict({"a": {"y": 3, "z": 4}})
        manager = ConfigManager.__new__(ConfigManager)
        result = manager.merge_configs(base, override)
        assert result["a"]["x"] == 1
        assert result["a"]["y"] == 3
        assert result["a"]["z"] == 4

    def test_get_current_config_not_initialized(self):
        manager = ConfigManager.__new__(ConfigManager)
        with pytest.raises(AttributeError):
            manager.get_current_config()

    def test_load_config_from_path_not_found(self):
        manager = ConfigManager.__new__(ConfigManager)
        manager.loader = YamlConfigLoader()
        with pytest.raises(ConfigError, match="not found"):
            manager.load_config_from_path(Path("/nonexistent/path.yaml"))


class TestConfigError:
    def test_config_error_is_exception(self):
        e = ConfigError("test error")
        assert isinstance(e, Exception)
        assert str(e) == "test error"
