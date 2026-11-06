from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from config import config


@dataclass(frozen=True)
class PopcornNerResponse:
    raw: Dict[str, Any]


class PopcornNerClient:
    """
    Minimal REST client for the Popcorn-NER service.

    Config path:
      - document_config.yaml: popcorn_ner: ./popcorn_ner
      - popcorn_ner/popcorn_ner.yaml: url, endpoints, timeouts...
    """

    def __init__(self) -> None:
        cfg = config.get_current_config()

        if isinstance(cfg, dict) and "popcorn_ner" in cfg:
            cfg = cfg["popcorn_ner"]

            # Allow runtime overrides (useful for local dev through Ingress)
            self.base_url = os.getenv("POPCORN_NER_URL", cfg.url).rstrip("/")

            endpoints = getattr(cfg, "endpoints", {}) or {}
            self.health_path = os.getenv("POPCORN_NER_HEALTH_PATH", endpoints.get("health", "/health"))
            self.ner_path = os.getenv("POPCORN_NER_ENDPOINT", endpoints.get("ner", "/extract"))

            request_cfg = getattr(cfg, "request", {}) or {}
            self.timeout = float(os.getenv("POPCORN_NER_TIMEOUT_S", request_cfg.get("timeout_s", 60)))
            self.retries = int(os.getenv("POPCORN_NER_RETRIES", request_cfg.get("retries", 2)))
            self.backoff_s = float(os.getenv("POPCORN_NER_BACKOFF_S", request_cfg.get("backoff_s", 0.5)))

            payload_cfg = getattr(cfg, "payload", {}) or {}
            self.default_lang = os.getenv("POPCORN_NER_DEFAULT_LANG", payload_cfg.get("default_lang", "eng"))
            self.text_field = payload_cfg.get("text_field", "text")
            self.lang_field = payload_cfg.get("lang_field", "lang")

        else:
            raise KeyError("Cannot find 'popcorn_ner' config in current config")


    def health(self) -> Dict[str, Any]:
        url = f"{self.base_url}{self.health_path}"
        r = requests.get(url, timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else {"status": r.text}

    def ner(self, text: str, lang: Optional[str] = None, **extra_payload: Any) -> PopcornNerResponse:
        """
        Calls the NER endpoint.

        Assumed request payload (override via YAML if needed):
          { "text": "...", "lang": "eng" }

        If your service expects a different schema, adjust the YAML or this method.
        """
        url = f"{self.base_url}{self.ner_path}"
        payload: Dict[str, Any] = {
            self.text_field: text,
            self.lang_field: lang or self.default_lang,
            **extra_payload,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout, verify=False)
                r.raise_for_status()
                return PopcornNerResponse(raw=r.json())
            except Exception as e:
                last_exc = e
                if attempt >= self.retries:
                    break
                # Simple backoff
                import time
                time.sleep(self.backoff_s * (attempt + 1))

        raise RuntimeError(f"Popcorn NER call failed after {self.retries + 1} attempts") from last_exc
