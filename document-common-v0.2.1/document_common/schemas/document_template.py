import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ------ document Template Objects ------
class NumLines(BaseModel):
    """Minimum and maximum line counts for a section."""

    min: int = Field(
        ...,
        ge=1,
        description="Minimum number of lines for the section",
        examples=[5],
    )
    max: int = Field(
        ...,
        ge=1,
        description="Maximum number of lines for the section",
        examples=[10],
    )


class SectionContent(BaseModel):
    """Defines a single section within a document template."""

    title: str = Field(
        ...,
        description="Human-readable title of the section",
        examples=["Introduction"],
    )
    name: str = Field(
        ...,
        description="Snake_case identifier for the section (e.g., 'intro', 'context')",
        examples=["intro"],
    )
    prompt: str = Field(
        ...,
        description="Prompt template ID to use for content assembly",
        examples=["t1_introduction"],
    )
    method: str = Field(
        ...,
        description="assembly method (e.g., 'direct', 'narrative')",
        examples=["direct"],
    )
    method_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional parameters passed to the assembly method",
        examples=[
            {
                "summaries_top_k": 15,
                "originals_top_k": 8,
                "entities_top_k": 10,
                "paragraphs_count": 3,
                "citations": True,
            }
        ],
    )
    using: Optional[List[str]] = Field(
        None,
        description="List of section name identifiers that this section depends on",
        examples=[["main", "context"]],
    )
    num_lines: NumLines = Field(
        ...,
        description="Allowed line count range for the generated section",
        examples=[{"min": 10, "max": 15}],
    )

    @field_validator("name")
    @classmethod
    def name_must_be_snake_case(cls, v: str):
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", v):
            raise ValueError("name must be in snake_case (lowercase, underscore-separated)")
        return v


class SectionBlock(BaseModel):
    """Wraps a SectionContent in a 'section' key, matching YAML structure."""

    section: SectionContent = Field(
        ...,
        description="A section definition wrapped in a 'section' key"
    )


class DocumentTemplate(BaseModel):
    """Complete document template definition."""

    template_name: str = Field(
        ...,
        description="Unique identifier for the template (e.g., 't1', 't2')",
        examples=["t1"],
    )
    description: str = Field(
        ...,
        description="Human-readable description of what the template generates",
        examples=[
            "This template generates a hook, a context, a main body with 3 paragraphs and a conclusion."
        ],
    )
    document: List[SectionBlock] = Field(
        ...,
        description="Ordered list of sections that make up the document"
    )

