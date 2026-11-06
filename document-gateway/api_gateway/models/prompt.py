from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PromptItem(BaseModel):
    promptId: str = Field(
        ...,
        examples=["t1_conclusion", "citation_classification"],
        description="Unique identifier of the prompt (snake-case).",
    )
    type: str = Field(
        ...,
        examples=["t1", "citations", "preprocessing"],
        description="Type/category of the prompt.",
    )
    mode: Optional[str] = Field(
        None,
        examples=["think", "no_think"],
        description="Mode of the prompt (think for reasoning, no_think otherwise).",
    )
    version: int = Field(
        ...,
        examples=[1],
        description="Version number of the prompt.",
    )
    content: str = Field(
        ...,
        examples=["Examples can be found in prompts/ directory"],
        description="The YAML content of the prompt.",
    )


class PromptListResponse(BaseModel):
    prompts: List[PromptItem]


class PromptUpsertRequest(BaseModel):
    content: str = Field(
        ...,
        description="Complete YAML content of the prompt including version field",
        examples=[
            """version: 1
my_prompt:
  system: |
    You are a helpful assistant.

  examples:
    - role: user
      content: "Example user message"
    - role: assistant
      content: "Example assistant response"

  query: |
    {{user_input}}
"""
        ],
    )

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Prompt content cannot be empty")
        return v

class PromptUpsertResponse(BaseModel):
    message: str = Field(..., examples=["Prompt 'analyse_text' saved successfully"])
    prompt_name: str = Field(..., examples=["analyse_text"])
    version: int = Field(..., examples=[1])

class PromptDeleteResponse(BaseModel):
    message: str = Field(..., examples=["Prompt 'analyse_text' deleted successfully"])
    prompt_name: str = Field(..., examples=["analyse_text"])
