"""Long-term per-user memory: facts in SQLite, embedding recall with FTS5 fallback."""

from typing import Any

from .embeddings import SafeEmbedder, pack, top_matches
from .store import Store
from .tools import Tool

MAX_FACT = 1000


class Memory:
    def __init__(
        self,
        store: Store,
        embedder: SafeEmbedder | None = None,
        k: int = 5,
        min_score: float = 0.35,
    ):
        self.store = store
        self.embedder = embedder
        self.k = k
        self.min_score = min_score

    async def _embed(self, text: str) -> list[float] | None:
        return await self.embedder.embed_one(text) if self.embedder else None

    async def add(self, user_id: str, content: str) -> dict[str, Any]:
        content = " ".join(content.split())[:MAX_FACT]
        if not content:
            raise ValueError("memory is empty")
        existing = self.store.find_memory(user_id, content)
        if existing:
            return existing
        vec = await self._embed(content)
        model = self.embedder.model if vec is not None and self.embedder else None
        return self.store.add_memory(user_id, content, pack(vec) if vec else None, model)

    def all(self, user_id: str) -> list[dict[str, Any]]:
        return self.store.list_memories(user_id)

    def delete(self, user_id: str, memory_id: str) -> bool:
        return self.store.delete_memory(user_id, memory_id)

    async def recall(self, user_id: str, query: str) -> list[dict[str, Any]]:
        if self.embedder is not None:
            rows = self.store.memory_vectors(user_id, self.embedder.model)
            vec = await self._embed(query) if rows else None
            if vec is not None:
                hits = top_matches(vec, rows, self.k, self.min_score)
                if hits:
                    return hits
        return self.store.search_memories(user_id, query, self.k)

    def tools(self, user_id: str) -> list[Tool]:
        async def remember(fact: str) -> str:
            m = await self.add(user_id, fact)
            return f"Saved memory {m['id']}: {m['content']}"

        def forget(memory_id: str) -> str:
            if self.delete(user_id, memory_id):
                return f"Forgot memory {memory_id}."
            raise KeyError(f"no memory with id {memory_id!r}")

        return [
            Tool(
                "remember",
                "Save a durable fact about the user to long-term memory (name, preferences,"
                " ongoing projects, or anything they ask you to remember). One short fact"
                " per call. Never store passwords, keys or other secrets.",
                {
                    "type": "object",
                    "properties": {"fact": {"type": "string", "description": "The fact."}},
                    "required": ["fact"],
                },
                remember,
            ),
            Tool(
                "forget",
                "Delete a long-term memory by its id, when the user asks you to forget"
                " something or a remembered fact is wrong.",
                {
                    "type": "object",
                    "properties": {"memory_id": {"type": "string"}},
                    "required": ["memory_id"],
                },
                forget,
            ),
        ]
