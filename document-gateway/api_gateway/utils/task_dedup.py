import hashlib
import json
from typing import Dict


def generate_task_hash(queue: str, payload: Dict) -> str:
    """Generate a deterministic SHA-256 hash for a payload to detect duplicate tasks.
    
    Args:
        queue: Queue name ('importation' or 'docassembler')
        payload: Request payload dict
    
    Returns:
        Hex digest SHA-256 hash string
    """
    if queue == "importation":
        return _generate_import_hash(payload)
    elif queue == "docassembler":
        return _generate_document_hash(payload)
    else:
        raise ValueError(f"Unknown queue: {queue}")


def _generate_import_hash(payload: Dict) -> str:
    if not isinstance(payload, dict) or "docs" not in payload:
        raise ValueError("Invalid import payload")

    docs = payload["docs"]
    if not isinstance(docs, list):
        raise ValueError("docs must be a list")

    sorted_docs = sorted(docs, key=lambda x: x.get("coreDataId", ""))
    payload_str = json.dumps(sorted_docs, sort_keys=True)

    return hashlib.sha256(payload_str.encode()).hexdigest()


def _generate_document_hash(payload: Dict) -> str:
    if not isinstance(payload, dict):
        raise ValueError("Invalid document payload")

    template_id = payload.get("templateId", "")
    docs = payload.get("docs", [])
    context = payload.get("context", {})
    flush = payload.get("flush", False)

    sorted_docs = sorted(docs, key=lambda x: x.get("coreDataId", ""))

    context_str = json.dumps(context, sort_keys=True)
    docs_str = json.dumps(sorted_docs, sort_keys=True)

    payload_str = f"{template_id}:{docs_str}:{context_str}:{flush}"

    return hashlib.sha256(payload_str.encode()).hexdigest()


def check_duplicate_task(redis_client, queue: str, task_hash: str, payload: Dict) -> str:
    """Check if an identical task is already running. Returns the existing task_id if found, else None."""
    key = f"dedup:{queue}:{task_hash}"
    data = redis_client.get(key)
    if not data:
        return None

    try:
        entries = json.loads(data)
        if isinstance(entries, list):
            for entry in entries:
                if entry.get("payload") == payload:
                    return entry.get("task_id")
        # If it's a legacy string (just task_id), we can't verify payload.
        # We return None to allow a new task to be created (safe approach).
    except (json.JSONDecodeError, TypeError):
        pass

    return None


def store_task_hash(
    redis_client, queue: str, task_hash: str, task_id: str, payload: Dict, ttl: int = 86400
):
    """Store a task hash->task_id mapping in Redis for deduplication.
    
    Args:
        ttl: Time-to-live in seconds (default 86400 = 24h)
    """
    key = f"dedup:{queue}:{task_hash}"
    data = redis_client.get(key)
    entries = []
    if data:
        try:
            entries = json.loads(data)
            if not isinstance(entries, list):
                entries = []
        except (json.JSONDecodeError, TypeError):
            entries = []

    entries.append({"payload": payload, "task_id": task_id})
    redis_client.setex(key, ttl, json.dumps(entries))


def cleanup_task_hash(redis_client, queue: str, task_hash: str):
    """Remove a task hash entry from Redis."""
    key = f"dedup:{queue}:{task_hash}"
    redis_client.delete(key)
