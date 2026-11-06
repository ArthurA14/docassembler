from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any
import time
import uuid
import json

app = FastAPI()


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None


class EmbeddingRequest(BaseModel):
    model: str | None = "mock-embedder"
    input: Any | None = None
    text: str | None = None


class RerankRequest(BaseModel):
    model: str | None = "mock-reranker"
    query: str | None = None
    documents: list[Any] = Field(default_factory=list)


def build_mock_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_text = "\n".join(
        str(m.get("content", ""))
        for m in messages
        if isinstance(m, dict)
    ).lower()

    # Étape outline du rapport
    if "document_outline_with_retrieval" in prompt_text or "outline" in prompt_text:
        return {
            "problematic": "Problématique mockée pour test.",
            "paragraphs": [
                {
                    "title": "Paragraphe mocké 1",
                    "query": "Requête mockée 1 pour récupérer les documents utiles.",
                },
                {
                    "title": "Paragraphe mocké 2",
                    "query": "Requête mockée 2 pour récupérer les documents utiles.",
                },
                {
                    "title": "Paragraphe mocké 3",
                    "query": "Requête mockée 3 pour récupérer les documents utiles.",
                },
            ],
        }

    # Étape génération/réécriture de paragraphes
    if (
        "paragraph_generator" in prompt_text
        or "sequential_paragraph_generator" in prompt_text
        or "restructure_paragraphs" in prompt_text
        or "paragraph" in prompt_text
    ):
        return {
            "paragraphs": [
                {
                    "title": "Paragraphe mocké",
                    "content": "Réponse mockée constante pour test.",
                    "source_ids": [],
                }
            ]
        }

    # Étape importation : summaries/refined summaries
    if (
        "extract_summary" in prompt_text
        or "aggregate_summaries" in prompt_text
        or "summary" in prompt_text
    ):
        return {
            "summary": "Réponse mockée constante pour test."
        }

    # Fallback prudent
    return {
        "summary": "Réponse mockée constante pour test."
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": "mock-llm", "object": "model", "created": now, "owned_by": "mock"},
            {"id": "mock-embedder", "object": "model", "created": now, "owned_by": "mock"},
            {"id": "mock-reranker", "object": "model", "created": now, "owned_by": "mock"},
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    payload = build_mock_payload(req.messages)
    content = json.dumps(payload, ensure_ascii=False)
    # content = json.dumps(
    #     {
    #         "summary": "Réponse mockée constante pour test.",
    #     },
    #     ensure_ascii=False,
    # )

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


# @app.post("/v1/embeddings")
# def embeddings(req: EmbeddingRequest):
#     vector = [0.01] * 768

#     return {
#         "object": "list",
#         "model": req.model,
#         "data": [
#             {
#                 "object": "embedding",
#                 "index": 0,
#                 "embedding": vector,
#             }
#         ],
#         "usage": {
#             "prompt_tokens": 1,
#             "total_tokens": 1,
#         },
#     }
@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest):
    inputs = req.input

    if inputs is None:
        inputs = req.text or ""

    if not isinstance(inputs, list):
        inputs = [inputs]

    return {
        "object": "list",
        "model": req.model,
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": [0.01] * 768,
            }
            for i, _ in enumerate(inputs)
        ],
        "usage": {
            "prompt_tokens": len(inputs),
            "total_tokens": len(inputs),
        },
    }


@app.post("/v1/rerank")
def rerank(req: RerankRequest):
    return {
        "model": req.model,
        "results": [
            {
                "index": i,
                "relevance_score": 1.0 - (i * 0.01),
            }
            for i, _ in enumerate(req.documents)
        ],
    }
