from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str = Field(..., examples=["More details on the error"])


class TimeoutErrorResponse(BaseModel):
    errorId: str = Field(
        ...,
        description="Unique ID of the timeout error",
        examples=["a1b2c3d4-e5f6-7890-1234-567890abcdef"],
    )
    errorMessage: str = Field(
        ...,
        description="Description of the timeout error",
        examples=["Request timed out after 30 seconds."],
    )