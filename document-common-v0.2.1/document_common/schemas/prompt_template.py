from typing import List, Optional

from pydantic import BaseModel, field_validator


class PromptExample(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        """Validate that role is either 'user' or 'assistant'."""
        if v not in ("user", "assistant"):
            raise ValueError(
                f'All examples roles must be "user" or "assistant". Found "{v}"'
            )
        return v


class PromptDict(BaseModel):
    system: str
    query: str
    examples: Optional[List[PromptExample]] = None
    version: Optional[int] = None


class PromptTemplate(BaseModel):
    name: str
    prompt: PromptDict
    version: int