"""MCP client: expose tools from external Model Context Protocol servers to the agent.

Servers are configured by the operator in `GROK_MCP_FILE` (default `mcp.json`):

    {"servers": {
        "fs":   {"command": "npx", "args": ["-y", "@mcp/server-filesystem", "/srv"],
                 "admin_only": true},
        "docs": {"url": "https://mcp.example.com/mcp",
                 "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"}, "timeout": 20}}}

Each server is owned by one background task, which keeps the session open until shutdown.
Tools are registered as `<server>__<tool>`.
"""

import asyncio
import json
import logging
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from .tools.base import Tool

log = logging.getLogger("grok_bot.mcp")
_ENV_VAR = re.compile(r"\$\{(\w+)\}")
_BAD_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def tool_name(server: str, tool: str) -> str:
    """OpenAI function names must match ^[a-zA-Z0-9_-]{1,64}$."""
    return _BAD_NAME_CHARS.sub("_", f"{server}__{tool}")[:64]


def _expand(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _ENV_VAR.sub(lambda m: env.get(m.group(1), ""), value)
    if isinstance(value, list):
        return [_expand(v, env) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, env) for k, v in value.items()}
    return value


@dataclass
class ServerConfig:
    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    admin_only: bool = False
    timeout: float = 30.0
    enabled: bool = True

    @property
    def source(self) -> str:
        return f"mcp:{self.name}"


def load_config(path: str | Path, env: dict[str, str] | None = None) -> list[ServerConfig]:
    """Parse mcp.json (`servers` or Claude-style `mcpServers`); a missing file means none."""
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return []
    if env is None:
        import os

        env = dict(os.environ)
    data = json.loads(p.read_text())
    servers = data.get("servers", data.get("mcpServers", {}))
    out = []
    for name, raw in servers.items():
        raw = _expand(raw, env)
        if bool(raw.get("command")) == bool(raw.get("url")):
            raise ValueError(f"MCP server {name!r}: set exactly one of 'command' or 'url'")
        out.append(
            ServerConfig(
                name=name,
                command=raw.get("command", ""),
                args=[str(a) for a in raw.get("args", [])],
                env={k: str(v) for k, v in raw.get("env", {}).items()},
                cwd=raw.get("cwd"),
                url=raw.get("url", ""),
                headers={k: str(v) for k, v in raw.get("headers", {}).items()},
                admin_only=bool(raw.get("admin_only", False)),
                timeout=float(raw.get("timeout", 30.0)),
                enabled=bool(raw.get("enabled", True)),
            )
        )
    return out


def render_result(result: Any) -> str:
    parts: list[str] = []
    for item in result.content:
        kind = getattr(item, "type", "")
        if kind == "text":
            parts.append(item.text)
        elif kind == "resource":
            res = item.resource
            parts.append(getattr(res, "text", None) or f"[resource {res.uri}]")
        else:
            parts.append(f"[{kind or 'binary'} content omitted]")
    if not parts and getattr(result, "structuredContent", None):
        parts.append(json.dumps(result.structuredContent))
    out = "\n".join(parts)
    if result.isError:
        raise RuntimeError(out or "MCP tool reported an error")
    return out


class MCPManager:
    def __init__(self, servers: list[ServerConfig] | None = None):
        self.servers = [s for s in servers or [] if s.enabled]
        self.status: dict[str, str] = {}
        self._sessions: dict[str, Any] = {}
        self._tools: list[tuple[ServerConfig, Tool]] = []
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    @classmethod
    def from_file(cls, path: str | Path) -> "MCPManager":
        return cls(load_config(path))

    async def start(self) -> None:
        """Connect to every server; one that fails or times out is skipped with a warning."""
        if self._tasks:
            raise RuntimeError("MCPManager is already started")
        self._stop = asyncio.Event()
        for cfg in self.servers:
            ready: asyncio.Future = asyncio.get_running_loop().create_future()
            task = asyncio.create_task(self._own(cfg, ready), name=f"mcp:{cfg.name}")
            try:
                session, tools = await asyncio.wait_for(asyncio.shield(ready), cfg.timeout)
            except Exception as e:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self.status[cfg.name] = f"error: {type(e).__name__}: {e}"
                log.warning("MCP server %s unavailable: %s", cfg.name, self.status[cfg.name])
                continue
            self._tasks.append(task)
            self._sessions[cfg.name] = session
            self._tools += [(cfg, self._wrap(cfg, t)) for t in tools]
            self.status[cfg.name] = "ok"
            log.info("MCP server %s: %d tools", cfg.name, len(tools))

    async def close(self) -> None:
        self._stop.set()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._sessions.clear()
        self._tools.clear()

    async def _own(self, cfg: ServerConfig, ready: asyncio.Future) -> None:
        # The SDK's transports are anyio task groups: enter and exit them in this one task.
        try:
            from mcp import ClientSession

            async with AsyncExitStack() as stack:
                if cfg.url:
                    read, write = await self._open_http(stack, cfg)
                else:
                    from mcp import StdioServerParameters
                    from mcp.client.stdio import stdio_client

                    params = StdioServerParameters(
                        command=cfg.command, args=cfg.args, env=cfg.env or None, cwd=cfg.cwd
                    )
                    read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed = await session.list_tools()
                ready.set_result((session, listed.tools))
                await self._stop.wait()
        except BaseException as e:
            if not ready.done():
                ready.set_exception(e if isinstance(e, Exception) else RuntimeError(repr(e)))
            if not isinstance(e, Exception):
                raise
        finally:
            self._sessions.pop(cfg.name, None)

    @staticmethod
    async def _open_http(stack: AsyncExitStack, cfg: ServerConfig):
        try:
            from mcp.client.streamable_http import streamable_http_client
        except ImportError:  # older mcp releases
            from mcp.client.streamable_http import streamablehttp_client

            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(cfg.url, headers=cfg.headers or None)
            )
            return read, write
        import httpx

        http = await stack.enter_async_context(
            httpx.AsyncClient(headers=cfg.headers, timeout=httpx.Timeout(30, read=300))
        )
        read, write, _ = await stack.enter_async_context(
            streamable_http_client(cfg.url, http_client=http)
        )
        return read, write

    def _wrap(self, cfg: ServerConfig, tool: Any) -> Tool:
        async def call(**arguments: Any) -> str:
            session = self._sessions.get(cfg.name)
            if session is None:
                raise RuntimeError(f"MCP server {cfg.name!r} is not connected")
            result = await session.call_tool(
                tool.name, arguments, read_timeout_seconds=timedelta(seconds=cfg.timeout)
            )
            return render_result(result)

        description = tool.description or getattr(tool, "title", None) or tool.name
        return Tool(
            name=tool_name(cfg.name, tool.name),
            description=f"[MCP {cfg.name}] {description}"[:1024],
            parameters=tool.inputSchema or {"type": "object", "properties": {}},
            func=call,
        )

    def listed(self, user: dict[str, Any] | None) -> list[tuple[str, Tool]]:
        """(source, tool) pairs this user may use; `admin_only` servers need an admin."""
        is_admin = bool(user and user.get("is_admin"))
        return [(c.source, t) for c, t in self._tools if is_admin or not c.admin_only]

    def tools_for(self, user: dict[str, Any] | None) -> list[Tool]:
        return [t for _, t in self.listed(user)]
