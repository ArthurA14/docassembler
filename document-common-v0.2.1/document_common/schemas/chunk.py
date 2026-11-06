from typing import Optional

from pydantic import BaseModel


class Chunk(BaseModel):
    type: str  # e.g., "original", "hquery", "summary"
    file_id: str
    version: Optional[int] = None
    project_name: Optional[str] = None
    parent_id: Optional[str] = None
    order: Optional[int] = None  # Where the chunk was placed in the original file.
    # Source metadata for traceability back to original file
    core_data_id: Optional[str] = None  # Unique ID of the original data
    core_type: Optional[str] = None  # Type of the original data
    core_data_source: Optional[str] = None  # Source/Manager responsible for the data
