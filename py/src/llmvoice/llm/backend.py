"""LLM backend contract.

Deliberately narrow: text in, text deltas out. The style channel is absent from
this interface on purpose — voice style must never reach the LLM.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class LlmBackend(Protocol):
    def stream(self, prompt: str) -> Iterator[str]:
        """Yield response text deltas as they are produced."""
        ...

    def cancel(self) -> None:
        """Ask an in-flight ``stream`` to stop promptly."""
        ...

    def close(self) -> None: ...


class EchoLlm:
    """Deterministic fake backend: streams a canned reply word by word.

    Used by the tests and by ``--dry-run`` so the pipeline can be exercised with
    no server and no GPU.
    """

    def __init__(self, reply: str | None = None, chunk_chars: int = 3) -> None:
        self._reply = reply
        self._chunk_chars = chunk_chars
        self._cancelled = False

    def stream(self, prompt: str) -> Iterator[str]:
        self._cancelled = False
        text = self._reply if self._reply is not None else _canned_reply(prompt)
        for i in range(0, len(text), self._chunk_chars):
            if self._cancelled:
                return
            yield text[i : i + self._chunk_chars]

    def cancel(self) -> None:
        self._cancelled = True

    def close(self) -> None:
        pass


def _canned_reply(prompt: str) -> str:
    return (
        f"You asked about {prompt.strip() or 'nothing in particular'}. "
        "Here is the first sentence, which should already be audible. "
        "The second sentence follows, and by now the pipeline is streaming. "
        "A third sentence confirms that segments keep arriving in order."
    )
