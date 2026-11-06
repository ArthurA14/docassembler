
from typing import List, Optional

from pydantic import BaseModel


class TemplateDocument(BaseModel):
    name: str
    description: Optional[str] = None
    tags: List[str] = []
    created_at: str
    updated_at: str
    content: str
