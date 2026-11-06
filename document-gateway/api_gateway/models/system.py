from typing import List

from haystack import Document
from pydantic import BaseModel, Field


class AvailableFilesResponse(BaseModel):
    files: List[Document]


class FlushSuccessResponse(BaseModel):
    message: str = Field(..., examples=["Data flushed successfully."])
