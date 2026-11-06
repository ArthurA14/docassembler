from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class TemplateItem(BaseModel):
    templateId: str = Field(
        ...,
        examples=["995af007-d96b-4e00-af67-0d78b36e56b0"],
        description="Unique identifier of the template.",
    )
    template: str = Field(
        ...,
        examples=["Examples can be found in config/configs/templates/document"],
        description="The content or reference to the document template.",
    )


class TemplateListResponse(BaseModel):
    templates: List[TemplateItem]


class TemplateUpsertRequest(BaseModel):
    content: str = Field(
        ...,
        description="Complete YAML content of the template",
        examples=[
            """template_name: my_template
description: My custom template
document:
  - section:
      method: direct
      title: "Introduction"
      name: intro
      prompt: my_prompt
      using: []
      num_lines:
        min: 5
        max: 10
"""
        ],
    )
    description: Optional[str] = Field(
        None, description="Optional description (overrides YAML description if provided)"
    )

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Template content cannot be empty")
        return v

class TemplateResponse(BaseModel):
    message: str = Field(
        ..., examples=["Template 't1' deleted/saved successfully"]
    )
    template_name: str = Field(
        ..., examples=["t1"]
    )
