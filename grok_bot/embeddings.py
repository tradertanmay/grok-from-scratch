"""Embedding helpers shared by long-term memory and file RAG."""

import array
import math
import time
from collections.abc import Callable
from typing import Any, Protocol

RETRY_AFTER = 300.0
BATCH = 32


class Embedder(Protocol):
    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLMEmbedder:
    """Embeds through any client exposing `embed(texts, model)` (e.g. ProviderRouter)."""

    def __init__(self, llm: Any, model: str):
        self.llm = llm
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self.llm.embed(texts, self.model)


class SafeEmbedder:
    """Never raises: returns None (so callers use keyword search) and backs off after errors."""

    def __init__(
        self,
        embedder: Embedder,
        retry_after: float = RETRY_AFTER,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.inner = embedder
        self.model = embedder.model
        self.retry_after = retry_after
        self._clock = clock
        self._down_until = 0.0

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not texts or self._clock() < self._down_until:
            return None
        try:
            out: list[list[float]] = []
            for i in range(0, len(texts), BATCH):
                out.extend(await self.inner.embed(texts[i : i + BATCH]))
            if len(out) != len(texts):
                raise ValueError("embedding count mismatch")
            return out
        except Exception:
            self._down_until = self._clock() + self.retry_after
            return None

    async def embed_one(self, text: str) -> list[float] | None:
        vecs = await self.embed([text])
        return vecs[0] if vecs else None


def pack(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def unpack(blob: bytes) -> list[float]:
    return array.array("f", blob).tolist()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def top_matches(
    query: list[float], rows: list[tuple[Any, bytes]], k: int, min_score: float
) -> list[Any]:
    scored = []
    for item, blob in rows:
        vec = unpack(blob)
        if len(vec) == len(query):
            scored.append((cosine(query, vec), item))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [item for score, item in scored[:k] if score >= min_score]
