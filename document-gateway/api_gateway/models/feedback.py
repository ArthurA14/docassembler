from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class SectionContent(BaseModel):
    sectionId: str = Field(
        ..., description="Unique identifier for the section within the document."
    )
    content: str = Field(..., description="The actual text content of the section.")


class SectionFeedback(BaseModel):
    sectionId: str = Field(
        ..., description="Unique identifier of the section being reviewed."
    )
    rating: int = Field(
        ..., ge=1, le=4, description="Rating for the section (1 to 4, where 4 is best)."
    )
    comment: Optional[str] = Field(
        None,
        description="Optional detailed comment or specific feedback about the section.",
    )


class FeedbackSubmission(BaseModel):
    documentId: UUID = Field(
        ...,
        description="Unique identifier of the document this feedback pertains to (corresponds to jobId).",
    )
    createdAt: str = Field(
        ..., description="Timestamp when the feedback was submitted (ISO 8601 format)."
    )
    originalSectionsContent: List[SectionContent] = Field(
        ...,
        description="List of the original generated content for each section of the document.",
    )
    correctedSectionsContent: Optional[List[SectionContent]] = Field(
        None,
        description="Optional list of user-corrected content for each section. Only include sections that were actually corrected by the user.",
    )
    initialUserInputs: Dict = Field(
        ...,
        description="A JSON object containing the initial inputs provided by the user for document assembly (e.g., templateId, selectedDocRefs, context).",
    )
    sectionsFeedback: List[SectionFeedback] = Field(
        ...,
        description="Detailed feedback for each section of the document, including rating and comment.",
    )


class FeedbackSubmissionResponse(BaseModel):
    message: str = Field(
        ..., description="Success message indicating feedback was saved."
    )
    feedbackId: str = Field(
        ..., description="Unique identifier for the stored feedback."
    )


class FeedbackErrorResponse(BaseModel):
    error: str = Field(
        ..., description="Error message describing the validation or processing error."
    )
