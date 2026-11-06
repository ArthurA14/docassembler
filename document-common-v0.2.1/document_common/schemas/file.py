import mimetypes
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class FileType(str, Enum):
    TXT = ".txt"
    PDF = ".pdf"
    JPG = ".jpg"
    JPEG = ".jpeg"
    PNG = ".png"
    MD = ".md"
    DOCX = ".docx"
    XLSX = ".xlsx"
    UNKNOWN = "unknown"

    # Mapping from MIME types to FileType enum values
    _MIME_MAPPING = {
        "text/plain": TXT,
        "application/pdf": PDF,
        "image/jpeg": JPG,
        "image/png": PNG,
        "text/markdown": MD,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DOCX,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": XLSX,
    }

    @classmethod
    def get_mime_type(cls, file: Any) -> str:
        """
        Get the MIME type from a file object using:
        1. file.content_type
        2. mimetypes.guess_type(file.filename)[0]
        3. "application/octet-stream" (fallback)

        Args:
            file: An object with .filename and optionally .content_type attributes

        Returns:
            The MIME type as a string
        """
        return (
            getattr(file, "content_type", None)
            or (
                mimetypes.guess_type(getattr(file, "filename", ""))[0]
                if hasattr(file, "filename")
                else None
            )
            or "application/octet-stream"
        )

    @classmethod
    def from_file(cls, file: Any) -> "FileType":
        """
        Determine the FileType from a file object.

        Args:
            file: An object with .filename and optionally .content_type attributes

        Returns:
            The matching FileType enum value
        """
        mime_type = cls.get_mime_type(file)

        # Try to find a match in MIME_MAPPING
        if mime_type in cls._MIME_MAPPING:
            return cls._MIME_MAPPING[mime_type]  # type: ignore

        # If no direct match but we have a filename, try extension-based matching
        if hasattr(file, "filename") and file.filename:
            extension = cls._get_extension(file.filename)
            for file_type in cls:
                if file_type.value == extension:
                    return file_type

        # Generic type fallbacks
        if mime_type.startswith("text/"):
            return cls.TXT
        elif mime_type.startswith("image/"):
            return cls.PNG

        return cls.UNKNOWN

    @staticmethod
    def _get_extension(filename: str) -> str:
        """Extract the file extension from a filename."""
        if not filename:
            return ""

        parts = filename.split(".")
        if len(parts) > 1:
            return f".{parts[-1].lower()}"
        return ""

    @classmethod
    def from_mime_type(cls, mime_type: str) -> "FileType":
        """
        Convert a MIME type to the corresponding FileType.

        Args:
            mime_type: MIME type string (e.g., "text/plain")

        Returns:
            The matching FileType enum value
        """
        return cls._MIME_MAPPING.get(mime_type, cls.UNKNOWN)  # type: ignore


class FileStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    importED = "imported"
    FAILED = "failed"


#    DUPLICATE = "duplicate"


class File(BaseModel):
    """Pydantic model for file metadata that will be returned by the component"""

    name: str
    path: str
    type: str
    size: int
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)  # TODO: update method
    status: FileStatus = Field(FileStatus.PENDING)
    message: str = Field("File created")
    importation_type: Optional[List[str]] = None  # the type of importation passed
    doc_types: Optional[List[str]] = None  # the type of chunks created
    content_id: str  # a hashlib id to deduplicate documents
    # Source metadata for traceability back to original file
    core_data_id: Optional[str] = None  # Unique ID of the original data
    core_type: Optional[str] = None  # Type of the original data
    core_data_source: Optional[str] = None  # Source/Manager responsible for the data
    # file_id: str # the id given by the user
    # content: Optional[str] = None
