from typing import List, Optional

from pydantic import BaseModel

import uuid
from datetime import datetime
from enum import Enum

from pydantic import Field
from document_common.schemas.citations import CitationStack, SourceReferences


# ----- document Objects ------
class SectionStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    FAILED = "failed"
    COMPLETED = "completed"


class Section(BaseModel):
    """
    Represents a structured section element with metadata for document assembly.

    Attributes:
        title (str): The title that the section will be preceded by.
        generated_content (Optional[str]): The generated content, typically filled by models like LLMs.
        source_ids (Optional[List[str]]): A optional list of IDs of documents ('original' chunks) used as a source.
        order (int): The position of the section in the final document.
        name (str): An internal identifier used during processing (not included in the final document).
        num_lines (tuple[int, int]): A tuple indicating the minimum and maximum number of lines desired.
        sections (Optional[List[str]]): Reserved for future use or extended structure; optional.
        section_id: Optional[int] = None : Reserved for db persistence
        status: SectionStatus = SectionStatus.PENDING : Reserved for db persistence
        citations (Optional[List[CitationStack]]): List of CitationStack objects extracted from generated content
        citation_extraction_status (Optional[str]): Status of citation extraction process
    """

    title: str
    order: int
    name: str
    generated_content: Optional[str] = None
    source_ids: List[str] = []
    sections: List['Section'] = []
    num_lines_min: Optional[int] = None
    num_lines_max: Optional[int] = None
    job_id: Optional[str] = None
    status: SectionStatus = SectionStatus.PENDING
    section_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Citation fields
    citations: Optional[List[List[CitationStack]]] = None  # List[CitationStack] - using List to avoid circular import
    source_references: Optional[SourceReferences] = None  # Source documents referenced by citations
    citation_extraction_status: Optional[str] = None  # "pending", "completed", "failed", "skipped"
    section_offsets: Optional[List[int]] = None  # Character offsets for each section in generated content


# Enum for document status
class DocumentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    FAILED = "failed"
    COMPLETED = "completed"


# document class definition
class Document(BaseModel):
    status: DocumentStatus = DocumentStatus.PENDING
    sections: Optional[List[Section]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_ids: List[str]  # The files used to generate the document