from collections.abc import AsyncIterator
from typing import Any

import pytest


class FakeLLM:
    """Replays scripted turns; each turn is a list of OpenAI-style deltas."""

    def __init__(self, turns: list[list[dict[str, Any]]]):
        self.turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    async def stream_chat(self, messages, tools, model) -> AsyncIterator[dict[str, Any]]:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools, "model": model})
        turn = self.turns.pop(0) if self.turns else [{"content": "(no more turns)"}]
        for delta in turn:
            yield delta

    async def list_models(self) -> list[str]:
        return ["fake-model"]


def text_turn(*pieces: str) -> list[dict[str, Any]]:
    return [{"content": p} for p in pieces]


def tool_turn(name: str, arguments: str, call_id: str = "call_1") -> list[dict[str, Any]]:
    half = len(arguments) // 2
    return [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments[:half]},
                }
            ]
        },
        {"tool_calls": [{"index": 0, "function": {"arguments": arguments[half:]}}]},
    ]


@pytest.fixture
def anyio_backend():
    return "asyncio"
