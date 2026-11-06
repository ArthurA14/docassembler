import pytest
from enum import Enum

from document_common.utils.config.utils import DotDict


class TestDotDict:
    def test_basic_dict_creation(self):
        dd = DotDict({"a": 1, "b": 2})
        assert dd["a"] == 1
        assert dd["b"] == 2

    def test_attribute_access(self):
        dd = DotDict({"x": 10, "y": 20})
        assert dd.x == 10
        assert dd.y == 20

    def test_attribute_set(self):
        dd = DotDict()
        dd.new_key = "value"
        assert dd["new_key"] == "value"

    def test_nested_dict_becomes_dotdict(self):
        dd = DotDict({"outer": {"inner": "value"}})
        assert dd.outer.inner == "value"

    def test_attribute_missing_raises_attribute_error(self):
        dd = DotDict({"a": 1})
        with pytest.raises(AttributeError):
            _ = dd.nonexistent

    def test_delete_attribute(self):
        dd = DotDict({"a": 1})
        del dd.a
        assert "a" not in dd

    def test_delete_missing_raises_attribute_error(self):
        dd = DotDict({"a": 1})
        with pytest.raises(AttributeError):
            del dd.missing

    def test_repr(self):
        dd = DotDict({"a": 1})
        assert repr(dd) == "{'a': 1}"

    def test_to_dict(self):
        dd = DotDict({"a": 1, "b": {"c": 2}})
        result = dd.to_dict()
        assert result == {"a": 1, "b": {"c": 2}}
        assert isinstance(result, dict)
        assert isinstance(result["b"], dict)

    def test_to_dict_nested_dotdict(self):
        dd = DotDict({"outer": {"inner": DotDict({"deep": "value"})}})
        result = dd.to_dict()
        assert isinstance(result["outer"], dict)
        assert result["outer"]["inner"] == {"deep": "value"}

    def test_to_json(self):
        dd = DotDict({"a": 1, "b": "hello"})
        import json
        result = dd.to_json()
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": "hello"}

    def test_empty_dotdict(self):
        dd = DotDict()
        assert dd == {}
        assert dd.to_dict() == {}

    def test_iteration(self):
        dd = DotDict({"a": 1, "b": 2, "c": 3})
        keys = list(dd.keys())
        assert set(keys) == {"a", "b", "c"}

    def test_len(self):
        dd = DotDict({"a": 1, "b": 2})
        assert len(dd) == 2

    def test_contains(self):
        dd = DotDict({"a": 1})
        assert "a" in dd
        assert "b" not in dd

    def test_get(self):
        dd = DotDict({"a": 1})
        assert dd.get("a") == 1
        assert dd.get("b", "default") == "default"

    def test_update(self):
        dd = DotDict({"a": 1})
        dd.update({"b": 2, "c": {"d": 3}})
        assert dd["b"] == 2
        assert dd.c.d == 3

    def test_to_dict_with_enum(self):
        class MyEnum(Enum):
            FOO = "foo"
            BAR = "bar"

        dd = DotDict({"status": MyEnum.FOO, "count": 5})
        result = dd.to_dict()
        assert result["status"] == "FOO"
        assert result["count"] == 5

    def test_copy(self):
        dd = DotDict({"a": 1, "b": {"c": 2}})
        copied = dd.copy()
        assert copied == dd
        assert copied is not dd
        # Nested should be shared (shallow copy behavior for nested DotDicts)
        assert isinstance(copied["b"], DotDict)
