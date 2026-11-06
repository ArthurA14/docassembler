"""
Storage layer for prompts and templates with versioning support.
"""

from document_common.storage.base import PromptStorage, TemplateStorage
from document_common.storage.factory import PromptStorageFactory, TemplateStorageFactory

__all__ = [
    "PromptStorage",
    "TemplateStorage",
    "PromptStorageFactory",
    "TemplateStorageFactory",
]
