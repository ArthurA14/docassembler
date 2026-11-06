import pytest
from datetime import datetime, timezone

from haystack import Document
from haystack.document_stores.types.policy import DuplicatePolicy

from document_common.storage.base import PromptStorage, TemplateStorage


class MockDocumentStore:
    """Minimal mock of Haystack DocumentStore for testing."""
    def __init__(self):
        self._docs: dict[str, Document] = {}
        self.written = []
        self.deleted = []

    def write_documents(self, docs, policy=DuplicatePolicy.NONE):
        for doc in docs:
            if policy == DuplicatePolicy.SKIP and doc.id in self._docs:
                continue
            self._docs[doc.id] = doc
        self.written.extend(docs)

    def filter_documents(self, filters=None):
        if filters is None:
            return list(self._docs.values())
        # Handle new-style flat filter format: {"field": "meta.prompt_name", "operator": "==", "value": "..."}
        conditions = filters.get("conditions", [])
        operator = filters.get("operator")
        results = []
        for doc in self._docs.values():
            match = True
            if conditions:
                for cond in conditions:
                    field = cond["field"]
                    value = cond["value"]
                    field_parts = field.split(".")
                    if field_parts[0] == "id":
                        doc_val = doc.id
                    elif field_parts[0] == "meta":
                        doc_val = doc.meta.get(field_parts[1], "")
                    else:
                        doc_val = ""
                    if cond["operator"] == "==":
                        if doc_val != value:
                            match = False
                    elif cond["operator"] == "!=":
                        if doc_val == value:
                            match = False
            else:
                # Handle new-style flat filter directly
                field = filters.get("field")
                op = filters.get("operator")
                value = filters.get("value")
                if field and op:
                    field_parts = field.split(".")
                    if field_parts[0] == "id":
                        doc_val = doc.id
                    elif field_parts[0] == "meta":
                        doc_val = doc.meta.get(field_parts[1], "")
                    else:
                        doc_val = ""
                    if op == "==" and doc_val != value:
                        match = False
                    elif op == "!=" and doc_val == value:
                        match = False
            if match:
                if operator == "OR":
                    results.append(doc)
                    break
                else:
                    results.append(doc)
        return results

    def delete_documents(self, doc_ids):
        for did in doc_ids:
            self._docs.pop(did, None)
        self.deleted.extend(doc_ids)


class TestPromptStorage:
    @pytest.fixture
    def storage(self):
        ds = MockDocumentStore()
        return PromptStorage(ds)

    def test_upsert_prompt(self, storage):
        doc = storage.upsert_prompt("test_prompt", "prompt content", prompt_type="t1", version=1)
        assert doc.id == "test_prompt_v1"
        assert doc.content == "prompt content"
        assert doc.meta["prompt_name"] == "test_prompt"
        assert doc.meta["type"] == "t1"

    def test_upsert_prompt_without_version(self, storage):
        doc = storage.upsert_prompt("test_prompt", "content", version=None)
        assert doc.id == "test_prompt"

    def test_get_prompt(self, storage):
        storage.upsert_prompt("my_prompt", "content v2", version=2)
        doc = storage.get_prompt("my_prompt")
        assert doc is not None
        assert doc.meta.get("version") == 2

    def test_get_prompt_not_found(self, storage):
        doc = storage.get_prompt("nonexistent")
        assert doc is None

    def test_get_all_versions(self, storage):
        storage.upsert_prompt("my_prompt", "content v1", version=1)
        storage.upsert_prompt("my_prompt", "content v2", version=2)
        storage.upsert_prompt("my_prompt", "content v3", version=3)
        versions = storage.get_all_versions("my_prompt")
        assert len(versions) == 3
        assert versions[0].meta["version"] == 3

    def test_delete_prompt(self, storage):
        storage.upsert_prompt("my_prompt", "content", version=1)
        result = storage.delete_prompt("my_prompt")
        assert result is True
        assert storage.get_prompt("my_prompt") is None

    def test_delete_prompt_not_found(self, storage):
        result = storage.delete_prompt("nonexistent")
        assert result is False

    def test_list_all_prompts(self, storage):
        storage.upsert_prompt("prompt_a", "content a", version=1)
        storage.upsert_prompt("prompt_b", "content b", version=1)
        prompts = storage.list_all_prompts()
        assert len(prompts) == 2
        names = {p["name"] for p in prompts}
        assert names == {"prompt_a", "prompt_b"}

    def test_prompt_exists(self, storage):
        storage.upsert_prompt("my_prompt", "content", version=1)
        assert storage.prompt_exists("my_prompt") is True
        assert storage.prompt_exists("nonexistent") is False

    def test_create_prompt_compatibility(self, storage):
        doc = storage.create_prompt("compat_prompt", "content", category="t1")
        assert doc is not None
        assert doc.meta["prompt_name"] == "compat_prompt"

    def test_get_latest_compatibility(self, storage):
        storage.upsert_prompt("my_prompt", "content", version=1)
        doc = storage.get_latest("my_prompt")
        assert doc is not None

    def test_get_latest_content(self, storage):
        storage.upsert_prompt("my_prompt", "my content", version=1)
        content = storage.get_latest_content("my_prompt")
        assert content == "my content"

    def test_get_latest_content_not_found(self, storage):
        with pytest.raises(ValueError, match="not found"):
            storage.get_latest_content("nonexistent")

    def test_preserve_type_on_update(self, storage):
        storage.upsert_prompt("my_prompt", "content v1", prompt_type="t1", version=1)
        storage.upsert_prompt("my_prompt", "content v2", version=2)
        doc = storage.get_prompt("my_prompt")
        assert doc.meta["type"] == "t1"

    def test_version_in_doc_id(self, storage):
        storage.upsert_prompt("my_prompt", "content v1", version=1)
        storage.upsert_prompt("my_prompt", "content v2", version=2)
        all_docs = storage.ds.filter_documents()
        ids = [d.id for d in all_docs]
        assert "my_prompt_v1" in ids
        assert "my_prompt_v2" in ids


class TestTemplateStorage:
    @pytest.fixture
    def storage(self):
        ds = MockDocumentStore()
        return TemplateStorage(ds)

    def test_upsert_template(self, storage):
        doc = storage.upsert_template("test_template", "template content", description="A test")
        assert doc.id == "test_template"
        assert doc.content == "template content"
        assert doc.meta["template_name"] == "test_template"
        assert doc.meta["description"] == "A test"

    def test_upsert_template_no_description(self, storage):
        doc = storage.upsert_template("test_template", "content")
        assert doc.meta["description"] is None

    def test_get_template(self, storage):
        storage.upsert_template("my_template", "content")
        doc = storage.get_template("my_template")
        assert doc is not None
        assert doc.content == "content"

    def test_get_template_not_found(self, storage):
        doc = storage.get_template("nonexistent")
        assert doc is None

    def test_delete_template(self, storage):
        storage.upsert_template("my_template", "content")
        storage.delete_template("my_template")
        assert storage.get_template("my_template") is None

    def test_delete_template_not_found(self, storage):
        with pytest.raises(ValueError, match="not found"):
            storage.delete_template("nonexistent")

    def test_list_all_templates(self, storage):
        storage.upsert_template("template_a", "content a")
        storage.upsert_template("template_b", "content b")
        templates = storage.list_all_templates()
        assert len(templates) == 2
        names = {t.name for t in templates}
        assert names == {"template_a", "template_b"}

    def test_template_exists(self, storage):
        storage.upsert_template("my_template", "content")
        assert storage.template_exists("my_template") is True
        assert storage.template_exists("nonexistent") is False

    def test_upsert_overwrites(self, storage):
        storage.upsert_template("my_template", "content v1")
        storage.upsert_template("my_template", "content v2")
        doc = storage.get_template("my_template")
        assert doc.content == "content v2"

    def test_preserve_created_at_on_update(self, storage):
        storage.upsert_template("my_template", "content v1")
        doc1 = storage.get_template("my_template")
        created_at = doc1.meta["created_at"]
        import time
        time.sleep(0.01)
        storage.upsert_template("my_template", "content v2")
        doc2 = storage.get_template("my_template")
        assert doc2.meta["created_at"] == created_at

    def test_tags_in_meta(self, storage):
        storage.upsert_template("my_template", "content", tags=["tag1", "tag2"])
        doc = storage.get_template("my_template")
        assert doc.meta["tags"] == ["tag1", "tag2"]
