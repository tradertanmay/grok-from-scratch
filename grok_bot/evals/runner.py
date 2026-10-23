import json
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_CASES = Path(__file__).with_name("cases.json")
REFUSAL = r"\b(can'?t|cannot|won'?t|not (going|able) to|refuse|sorry|no way|not help)"


@dataclass
class Case:
    id: str
    category: str
    steps: list[dict[str, Any]]
    checks: dict[str, Any]
    slow: bool = False


@dataclass
class Turn:
    answer: str = ""
    think: str = ""
    tools: list[str] = field(default_factory=list)
    statuses: int = 0
    error: str = ""
    first_token_s: float | None = None
    seconds: float = 0.0
    session_id: str | None = None


@dataclass
class Result:
    id: str
    category: str
    status: str  # pass | fail | skip | error
    reasons: list[str]
    seconds: float = 0.0
    first_token_s: float | None = None
    answer: str = ""
    tools: list[str] = field(default_factory=list)


def load_cases(path: str | Path = DEFAULT_CASES) -> list[Case]:
    return [Case(**c) for c in json.loads(Path(path).read_text())]


def _norm(s: str) -> str:
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s.lower())
    s = re.sub(r"[\u00a0\u202f\u2009]", " ", s)
    return re.sub(r"(?<=\d) (?=\d{3}\b)", "", s)


def check_answer(turn: Turn, checks: dict[str, Any]) -> list[str]:
    """Return failure reasons (empty = pass)."""
    text, low, bad = turn.answer, _norm(turn.answer), []
    if turn.error:
        bad.append(f"error event: {turn.error[:120]}")
    if not text.strip():
        bad.append("empty answer")
    if "contains_any" in checks and not any(_norm(s) in low for s in checks["contains_any"]):
        bad.append(f"missing any of {checks['contains_any']}")
    for s in checks.get("contains_all", []):
        if _norm(s) not in low:
            bad.append(f"missing {s!r}")
    for s in checks.get("not_contains", []):
        if s.lower() in low:
            bad.append(f"should not contain {s!r}")
    if "regex" in checks and not re.search(checks["regex"], text, re.I | re.M):
        bad.append(f"no match for /{checks['regex']}/")
    if "not_regex" in checks and re.search(checks["not_regex"], text, re.I | re.M):
        bad.append(f"matched forbidden /{checks['not_regex']}/")
    if checks.get("refuses") and not re.search(REFUSAL, text, re.I):
        bad.append("did not refuse")
    if "tool_called" in checks and checks["tool_called"] not in turn.tools:
        bad.append(f"tool {checks['tool_called']} not called (called {turn.tools})")
    for name in checks.get("tool_not_called", []):
        if name in turn.tools:
            bad.append(f"tool {name} should not be called")
    if checks.get("no_tools") and turn.tools:
        bad.append(f"unexpected tools {turn.tools}")
    if checks.get("status_events") and not turn.statuses:
        bad.append("no DeepSearch status events")
    if "min_chars" in checks and len(text) < checks["min_chars"]:
        bad.append(f"answer shorter than {checks['min_chars']} chars ({len(text)})")
    if "max_chars" in checks and len(text) > checks["max_chars"]:
        bad.append(f"answer longer than {checks['max_chars']} chars ({len(text)})")
    if "max_seconds" in checks and turn.seconds > checks["max_seconds"]:
        bad.append(f"took {turn.seconds:.0f}s > {checks['max_seconds']}s")
    return bad


class Skip(Exception):
    pass


class Runner:
    def __init__(self, client: httpx.AsyncClient, model: str | None = None, think: bool = False):
        self.client = client
        self.model = model
        self.think = think
        self.sessions: list[str] = []

    async def login(self) -> bool:
        """Sign up a throwaway user when auth is on. Returns True if a user was created."""
        r = await self.client.get("/api/health")
        r.raise_for_status()
        if not r.json().get("auth_enabled", False):
            return False
        creds = {"username": f"eval-{secrets.token_hex(4)}", "password": secrets.token_urlsafe(16)}
        r = await self.client.post("/api/auth/signup", json=creds)
        if r.status_code >= 400:
            raise RuntimeError(f"eval signup failed: HTTP {r.status_code} {r.text[:200]}")
        return True

    async def memory_ids(self) -> set[str]:
        r = await self.client.get("/api/memories")
        return {m["id"] for m in r.json().get("memories", [])} if r.status_code == 200 else set()

    async def upload(self, spec: dict[str, Any], session_id: str | None) -> str:
        files = {"file": (spec["name"], spec["content"].encode(), spec.get("mime", "text/plain"))}
        data = {"session_id": session_id} if session_id else {}
        r = await self.client.post("/api/files", files=files, data=data)
        if r.status_code in (404, 405):
            raise Skip("server has no /api/files")
        r.raise_for_status()
        return r.json()["id"]

    async def chat(self, step: dict[str, Any], session_id: str | None) -> Turn:
        file_ids = [await self.upload(step["upload"], session_id)] if "upload" in step else []
        body: dict[str, Any] = {
            "message": step["message"],
            "session_id": session_id,
            "mode": step.get("mode", "fun"),
            "think": step.get("think", self.think),
            "deepsearch": step.get("deepsearch", False),
            "memory": step.get("memory", True),
        }
        if file_ids:
            body["file_ids"] = file_ids
        if self.model:
            body["model"] = self.model
        turn, start = Turn(session_id=session_id), time.monotonic()
        async with self.client.stream("POST", "/api/chat", json=body, timeout=600) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                turn.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return turn
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                ev = json.loads(line[5:])
                kind = ev.get("type")
                if kind == "session":
                    turn.session_id = ev.get("session_id")
                elif kind == "token":
                    if turn.first_token_s is None:
                        turn.first_token_s = time.monotonic() - start
                    turn.answer += ev.get("content", "")
                elif kind == "think":
                    turn.think += ev.get("content", "")
                elif kind == "tool_call":
                    turn.tools.append(ev.get("name", ""))
                elif kind == "status":
                    turn.statuses += 1
                elif kind == "error":
                    turn.error = ev.get("message") or ev.get("content") or "error"
        turn.seconds = time.monotonic() - start
        if turn.session_id:
            self.sessions.append(turn.session_id)
        return turn

    async def run_case(self, case: Case) -> Result:
        session_id, turn = None, Turn()
        try:
            for step in case.steps:
                if step.get("new_session"):
                    session_id = None
                turn = await self.chat(step, session_id)
                session_id = turn.session_id
        except Skip as e:
            return Result(case.id, case.category, "skip", [str(e)])
        except Exception as e:
            return Result(case.id, case.category, "error", [f"{type(e).__name__}: {e}"])
        reasons = check_answer(turn, case.checks)
        return Result(
            case.id,
            case.category,
            "fail" if reasons else "pass",
            reasons,
            turn.seconds,
            turn.first_token_s,
            turn.answer,
            turn.tools,
        )

    async def cleanup(self, keep_memories: set[str]) -> None:
        for sid in self.sessions:
            await self.client.delete(f"/api/sessions/{sid}")
        for mid in await self.memory_ids() - keep_memories:
            await self.client.delete(f"/api/memories/{mid}")


async def run_suite(
    client: httpx.AsyncClient,
    cases: list[Case],
    model: str | None = None,
    think: bool = False,
    on_result=None,
) -> list[Result]:
    runner = Runner(client, model, think)
    await runner.login()
    before = await runner.memory_ids()
    results = []
    try:
        for case in cases:
            result = await runner.run_case(case)
            results.append(result)
            if on_result:
                on_result(result)
    finally:
        await runner.cleanup(before)
    return results
