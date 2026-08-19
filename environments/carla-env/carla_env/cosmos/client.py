"""HTTP client for the Cosmos frame server."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request

from ..logging import get_logger
from .config import CosmosConfig

logger = get_logger("cosmos.client")


class CosmosClient:
    """Sends camera frames to a remote Cosmos Transfer2.5 server."""

    def __init__(self, config: CosmosConfig):
        self._config = config
        self._base_url = str(config.server_url or "").rstrip("/")

    def _request_json(self, method: str, path: str, payload: dict | None, timeout: float) -> dict:
        if not self._base_url:
            raise RuntimeError("Cosmos server_url is empty")

        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Cosmos request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.timeout):
                raise RuntimeError(f"Cosmos request timed out after {timeout}s") from exc
            raise RuntimeError(f"Cannot connect to Cosmos server at {self._base_url}") from exc

    def health(self) -> bool:
        try:
            data = self._request_json("GET", "/health", None, 5.0)
        except Exception as exc:
            logger.warning("Cosmos health check failed: %s", exc)
            return False
        return bool(data.get("model_loaded", False))

    def stylize(self, frame_base64: str) -> str:
        payload = {
            "image": frame_base64,
            "prompt": self._config.prompt,
            "seed": int(self._config.seed),
            "control": self._config.control,
            "control_weight": float(self._config.control_weight),
        }
        data = self._request_json("POST", "/stylize", payload, float(self._config.timeout))
        image = data.get("image")
        if not image:
            raise RuntimeError("Cosmos server response did not include an image payload")
        return str(image)
