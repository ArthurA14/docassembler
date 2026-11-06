import json
from enum import Enum
from typing import Any, Dict


class DotDict(dict[str, Any]):
    """Dict subclass supporting attribute-style access (e.g., `d.key`) with recursive nested dict conversion."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)


    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return dict.__repr__(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the DotDict instance to a regular dictionary."""
        result: Dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            if isinstance(value, Enum):
                result[key] = value.name
            else:
                result[key] = value
        return result

    def to_json(self) -> str:
        """Convert the DotDict instance to a JSON string."""
        return json.dumps(self.to_dict())

    def update(self, *args: Any, **kwargs: Any) -> None:
        for key, value in dict(*args, **kwargs).items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
            else:
                self[key] = value


def dotdict_representer(dumper: Any, data: DotDict) -> Any:
    return dumper.represent_dict(data.to_dict())
