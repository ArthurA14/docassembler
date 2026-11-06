from __future__ import annotations


from pydantic import BaseModel, Field, field_validator


class DocRef(BaseModel):
    coreDataId: str = Field(
        ...,
        description="Unique ID of the data (UUID format).",
        examples=["a1b2c3d4-e5f6-7890-1234-567890abcdef"],
    )
    coreType: str = Field(
        ..., description="Type of the data.", examples=["document", "multimedia"]
    )
    coreDataSource: str = Field(
        ...,
        description="Source/Manager responsible for the data.",
        examples=["Document Manager", "SourceBox"],
    )

class ImportDocRef(BaseModel):
    coreDataId: str = Field(
        ...,
        description="Unique ID of the data (UUID format).",
        examples=["a1b2c3d4-e5f6-7890-1234-567890abcdef"],
    )
    coreType: str = Field(
        ..., description="Type of the data.", examples=["document", "multimedia"]
    )
    coreDataSource: str = Field(
        ...,
        description="Source/Manager responsible for the data.",
        examples=[
            "Document Manager",
            "SourceBox",
        ],
    )
    plainText: str = Field(
        ...,
        description="Multipage Content of the document extracted through Tika",
        examples=[
            "La tension persiste dans les deux régions anglophones du Cameroun, où les séparatistes ont appelé à\ndix jours de « ville morte » dans plusieurs localités, a-t-on appris auprès de plusieurs témoins..."
        ],
    )

    @field_validator("plainText")
    @classmethod
    def plain_text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("plainText cannot be empty or whitespace only")
        return v