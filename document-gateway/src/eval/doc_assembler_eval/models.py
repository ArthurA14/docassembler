import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class DocRef(TypedDict):
    coreDataId: str
    coreType: str
    coreDataSource: str
    plainText: str
    filename: str

class GroundTruth(BaseModel):
    id: Any
    subject: str = Field(default="No subject")
    context: Optional[Dict[str, str]] = Field(default=None)
    template: str
    gold_documents: List[str]
    answer: List[str]

    @classmethod
    def load_ground_truth(cls, ground_truth_path: Path) -> List['GroundTruth']:
        """Load ground truth data from JSONL file."""
        ground_truth: List['GroundTruth'] = []
        seen_ids: set[str] = set()
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        gt = GroundTruth(**data)
                        if gt.id in seen_ids:
                            raise ValueError(f"Duplicate ground truth ID '{gt.id}' at line {line_num}")
                        seen_ids.add(gt.id)
                        ground_truth.append(gt)
                    except Exception as e:
                        raise ValueError(f"Error parsing ground truth at line {line_num}: {e}")
        return ground_truth
