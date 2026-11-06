# /import
import uuid
from typing import List

from pydantic import BaseModel, Field
from document_common.schemas.api import ImportDocRef
from document_common.schemas.file import FileStatus


class ImportRequest(BaseModel):
    docs: List[ImportDocRef] = Field(
        ..., description="List of document references to import."
    )


class ImportAcceptedResponse(BaseModel):
    message: str = Field(..., examples=["importation process started for the provided documents."])
    # taskId: Optional[str] = Field(default=None, examples=["40b0c46c-68b9-4c64-81c4-c4f156754686"])


class RequestTimeoutException(Exception):
    def __init__(self, message: str="Request timed out.") :
        super().__init__(message)
        self.error_id = str(uuid.uuid4())
        self.message = message


class ImportationStatusResponse(BaseModel):
    coreDataId: str = Field(
        ...,
        description="ID of the imported file.",
        examples=["88651aa9-5a69-4518-beaf-66c8bc34ee1c"],
    )
    status: FileStatus = Field(
        ..., description="Current importation status of the file.", examples=["imported"]
    )