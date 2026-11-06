import json
from typing import Any, List

from haystack.dataclasses.chat_message import ChatMessage
from pydantic import BaseModel

# ----------------- Utility Functions -----------------


# Build a formatted string from a list of ChatMessages for better logging
def build_chat_messages_str(data: List[ChatMessage]):
    cm_text = "x" * 36 + " PROMPT " + "x" * 36
    for message in data:
        role = message._role.value.upper()  # Convert role to uppercase
        text = message.text  # Extract message text
        cm_text += f"\n{role}:\n{text}\n" + "-" * 80
    cm_text += "\n" + "x" * 80
    return cm_text


# Print a Pydantic model or a list of models (unused)
def print_model(obj: Any) -> None:
    if isinstance(obj, BaseModel):
        # Convert model to dict and print as formatted JSON
        print(json.dumps(obj.model_dump(), indent=2, ensure_ascii=False))
    elif isinstance(obj, list):
        # Recursively print each item in the list
        for item in obj:
            print_model(item)
    else:
        # Fallback: use standard print
        print(obj)
