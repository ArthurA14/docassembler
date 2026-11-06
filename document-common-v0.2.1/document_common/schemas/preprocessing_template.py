from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProcessorTemplate(BaseModel):
    type: str = Field(..., description="Type of subchunking (e.g., 'direct', 'entity_extraction')")
    using: Optional[str] = Field(None, description="Optional reference to another processor by name")


class PreprocessingTemplate(BaseModel):
    preprocessing: List[ProcessorTemplate]
    context: Dict[str, str]
