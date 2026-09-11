"""llama.cpp server client: OpenAI-compatible streaming.

Assumes ``llama-server`` is already running at ``llm.base_url``. Start it
yourself so its logs are in front of you:

    llama-server --model <model.gguf> --ctx-size 4096 --n-gpu-layers 999 --port 8080
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator

import httpx

from ..config import LlmConfig

log = logging.getLogger(__name__)

_SENTINEL_DONE = "[DONE]"


class LlamaServerLlm:
    """Streams deltas from ``POST /v1/chat/completions`` with ``stream: true``."""

    def __init__(self, cfg: LlmConfig) -> None:
        self._cfg = cfg
        self._cancel = threading.Event()
        self._client = httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            timeout=httpx.Timeout(cfg.request_timeout_s, connect=cfg.connect_timeout_s),
        )

    def stream(self, prompt: str) -> Iterator[str]:
        self._cancel.clear()
        payload: dict[str, object] = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": self._cfg.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self._cfg.max_tokens,
            "temperature": self._cfg.temperature,
            "top_p": self._cfg.top_p,
            "stream": True,
        }
        if self._cfg.disable_thinking:
            # Qwen3 chat template switch; harmless on servers that ignore it.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            with self._client.stream("POST", "/v1/chat/completions", json=payload) as r:
                if r.status_code != 200:
                    body = r.read().decode("utf-8", "replace")[:500]
                    raise RuntimeError(f"llama-server returned {r.status_code}: {body}")
                for line in r.iter_lines():
                    if self._cancel.is_set():
                        return
                    delta = _parse_sse_delta(line)
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"llama-server unreachable at {self._cfg.base_url} ({exc}). "
                f"Start it, or use --fake-llm."
            ) from exc

    def cancel(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        self._client.close()


def _parse_sse_delta(line: str) -> str | None:
    """Return the content delta, or None for non-data/contentless events."""
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == _SENTINEL_DONE:
        return None
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        log.debug("unparseable SSE payload: %r", data[:200])
        return None
    choices = event.get("choices") or []
    if not choices:
        return None
    content = (choices[0].get("delta") or {}).get("content")
    return content if isinstance(content, str) and content else None
