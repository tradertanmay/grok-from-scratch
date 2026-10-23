# Chapter 2 — The Agent Loop, Tools and MCP

> **What you build:** the loop that turns a chat model into an agent: stream, run tool calls, feed the results back, repeat. Plus the built-in tools (web search, URL fetch, sandboxed Python) with their security boundaries, and an MCP client that pulls in tools from outside servers. Files: `grok_bot/agent.py`, `grok_bot/tools/`, `grok_bot/mcp_client.py`.

## 2.1 What a tool is

A tool is a name, a description, a JSON Schema for its arguments, and a Python function. Nothing more:

`grok_bot/tools/base.py` (lines 6–59):

```python
MAX_TOOL_OUTPUT = 12_000


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str | Awaitable[str]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    output: str
    is_error: bool = False


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {t.name: t for t in tools or []}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def __iter__(self):
        return iter(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"Unknown tool: {name}", is_error=True)
        try:
            out = tool.func(**arguments)
            if inspect.isawaitable(out):
                out = await out
        except Exception as e:  # tool failures are reported back to the model
            return ToolResult(f"{type(e).__name__}: {e}", is_error=True)
        out = str(out)
        if len(out) > MAX_TOOL_OUTPUT:
            out = out[:MAX_TOOL_OUTPUT] + "\n...[truncated]"
        return ToolResult(out)
```

Two choices here keep the agent robust:

- **Tool failures become text, not exceptions.** If a tool raises, the model sees `ValueError: ...` and can retry or explain. One broken tool never kills the turn.
- **Every output is capped** at `MAX_TOOL_OUTPUT` (12,000 characters). A web page or MCP server that returns megabytes can't blow up the context window.

The default registry is just a list:

`grok_bot/tools/__init__.py` (lines 8–12):

```python
def default_registry(searxng_url: str = "") -> ToolRegistry:
    tools = [web_search_tool(searxng_url), fetch_url_tool()]
    if os.environ.get("GROK_SANDBOX", "auto").lower() != "off":
        tools.append(python_exec_tool())
    return ToolRegistry(tools)
```

## 2.2 The loop

`grok_bot/agent.py` (lines 153–251):

```python
class Agent:
    llm: LLMClient
    tools: ToolRegistry
    model: str
    max_steps: int = 8
    think: bool = False
    new_messages: list[dict[str, Any]] = field(default_factory=list)

    async def run(self, messages: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        """Yield UI events. Every message the agent produces is appended to `messages`
        and to `self.new_messages` (for persistence)."""
        schemas = self.tools.schemas()
        for step in range(self.max_steps):
            use_tools = step < self.max_steps - 1
            content = ""
            reasoning = ""
            streamed: dict[int, _StreamedCall] = {}
            holdback = TagHoldback()
            splitter = ThinkSplitter()
            async for delta in self.llm.stream_chat(
                messages, schemas if use_tools else [], self.model
            ):
                native = delta.get("reasoning") or delta.get("reasoning_content")
                parts = [("think", native)] if native else []
                if delta.get("content"):
                    parts += splitter.feed(delta["content"])
                for kind, text in parts:
                    if kind == "think":
                        reasoning += text
                        if self.think:
                            yield {"type": "think", "content": text}
                        continue
                    content += text
                    visible = holdback.feed(text)
                    if visible:
                        yield {"type": "token", "content": visible}
                for tc in delta.get("tool_calls") or []:
                    sc = streamed.setdefault(tc.get("index", len(streamed)), _StreamedCall())
                    sc.id = tc.get("id") or sc.id
                    fn = tc.get("function") or {}
                    sc.name += fn.get("name") or ""
                    args = fn.get("arguments")
                    sc.arguments += json.dumps(args) if isinstance(args, dict) else (args or "")
            for kind, text in splitter.flush():
                if kind == "think":
                    reasoning += text
                    if self.think:
                        yield {"type": "think", "content": text}
                else:
                    content += text
                    visible = holdback.feed(text)
                    if visible:
                        yield {"type": "token", "content": visible}
            tail = holdback.flush()
            if tail:
                yield {"type": "token", "content": tail}

            calls = [
                ParsedToolCall(sc.id or _new_id(), sc.name, _parse_args(sc.arguments), sc.arguments)
                for _, sc in sorted(streamed.items())
                if sc.name
            ]
            clean_content = content
            if not calls and TOOL_TAG_OPEN in content:
                clean_content, calls = parse_text_tool_calls(content)

            assistant: dict[str, Any] = {"role": "assistant", "content": clean_content.strip()}
            if reasoning and self.think:
                assistant["reasoning"] = reasoning
            if calls:
                assistant["tool_calls"] = [c.as_openai() for c in calls]
            self._append(messages, assistant)
            if not calls:
                return

            for call in calls:
                yield {
                    "type": "tool_call",
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                result = await self.tools.call(call.name, call.arguments)
                yield {
                    "type": "tool_result",
                    "id": call.id,
                    "name": call.name,
                    "output": result.output,
                    "is_error": result.is_error,
                }
                self._append(
                    messages,
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result.output,
                    },
                )
```

One iteration of `for step in range(self.max_steps)` is one model call:

1. Stream deltas. Text goes to the user straight away as `token` events. Reasoning goes out as `think` events, but only in Think mode (Chapter 3).
2. Collect `tool_calls` fragments by `index`. Providers stream a tool call's name and JSON arguments in pieces, so `_StreamedCall` concatenates them.
3. If there were no tool calls, the answer is final: `return`.
4. Otherwise run each call, emit `tool_call`/`tool_result` events for the UI, append `role: "tool"` messages, and loop.

On the **last step, the tools are withheld** (`use_tools = step < self.max_steps - 1`). The model then has to answer with what it has instead of looping forever. `max_steps` comes from `GROK_MAX_TOOL_STEPS` (default 8).

Every produced message goes into `self.new_messages` too, so the web layer can save exactly what this turn added (Chapter 5).

### Models that write tool calls as text

Some open models ignore the `tools` API and write `<tool_call>{"name": ..., "arguments": ...}</tool_call>` into their text. Two small pieces handle this:

`grok_bot/agent.py` (lines 54–66):

```python
def parse_text_tool_calls(text: str) -> tuple[str, list[ParsedToolCall]]:
    """Extract `<tool_call>{"name":..., "arguments":{...}}</tool_call>` blocks from text."""
    calls = []
    for m in _TOOL_TAG_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        calls.append(ParsedToolCall(_new_id(), str(obj["name"]), _parse_args(args)))
    return _TOOL_TAG_RE.sub("", text).strip(), calls
```

`TagHoldback` sits between the stream and the user. It passes text through but holds back anything that is, or *might become*, a `<tool_call>` block, so the user never sees half a tag flash on screen. `_partial_suffix` is what decides whether a trailing `<tool_` could still be the start of a tag.

## 2.3 Built-in tools and their security boundaries

### `web_search` and `fetch_url`: SSRF defense

A model that can fetch any URL can be talked into fetching `http://169.254.169.254/` (cloud metadata) or `http://localhost:8000/api/admin/...`. Every URL, including **every redirect hop** (`MAX_REDIRECTS = 5`, redirects are followed by hand), goes through this check:

`grok_bot/tools/web.py` (lines 30–44):

```python
async def check_public_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts resolving to private/loopback/link-local IPs (SSRF)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only http(s) URLs are allowed")
    if _allow_private():
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    infos = await asyncio.get_running_loop().getaddrinfo(
        parsed.hostname, port, type=socket.SOCK_STREAM
    )
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise ValueError(f"Refusing to fetch non-public address {ip} ({parsed.hostname})")
```

We resolve the hostname ourselves and reject any address that isn't globally routable. This covers loopback, RFC 1918, link-local, CGNAT and IPv6 ULA. `GROK_ALLOW_PRIVATE_FETCH=1` turns it off for local development.

Search tries SearXNG if `SEARXNG_URL` is set, then DuckDuckGo's HTML endpoint, then Wikipedia. That way a free deployment always has *some* working search.

### `python_exec`: a disposable container

`grok_bot/tools/python_exec.py` (lines 58–67):

```python
def docker_command(name: str, memory_mb: int) -> list[str]:
    return [
        "docker", "run", "--rm", "-i", "--name", name,
        "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=64m",
        "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m",
        "--cpus", "1", "--pids-limit", "64", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", "65534:65534",
        "-w", "/tmp", "-e", "HOME=/tmp",
        SANDBOX_IMAGE, "python", "-I", "-",
    ]  # fmt: skip
```

No network, read-only root, a 64 MB tmpfs, memory/CPU/PID limits, all capabilities dropped, and the user `nobody`. There is a 15 s timeout, and the container is killed by name if it runs over. Without Docker there is a fallback: a subprocess with `setrlimit` limits, which is fine for a laptop but not for a public server. `GROK_SANDBOX=off` removes the tool entirely. See `docs/DEPLOY.md` for the Docker-in-Compose setup.

## 2.4 MCP: tools from outside servers

The [Model Context Protocol](https://modelcontextprotocol.io) lets a separate process expose tools: a filesystem, GitHub, a database, your company's docs. Grok Bot is an MCP *client*. It uses the official `mcp` Python SDK (`pip install -e ".[mcp]"`) instead of hand-written JSON-RPC.

**Configuration is operator-only.** A stdio MCP server is a command that runs on your host, so there is deliberately no API for adding servers. You edit `GROK_MCP_FILE` (default `mcp.json`, git-ignored):

`mcp.example.json` (lines 1–19):

```json
{
  "servers": {
    "fs": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/srv/shared"],
      "admin_only": true
    },
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time"]
    },
    "docs": {
      "url": "https://mcp.example.com/mcp",
      "headers": {"Authorization": "Bearer ${DOCS_MCP_TOKEN}"},
      "timeout": 20,
      "enabled": false
    }
  }
}
```

`mcpServers` (Claude Desktop format) is accepted too, so existing configs can be copied in. `${VAR}` is expanded from the environment, so tokens stay out of the file.

### Naming

`grok_bot/mcp_client.py` (lines 32–34):

```python
def tool_name(server: str, tool: str) -> str:
    """OpenAI function names must match ^[a-zA-Z0-9_-]{1,64}$."""
    return _BAD_NAME_CHARS.sub("_", f"{server}__{tool}")[:64]
```

OpenAI-style function names must match `^[a-zA-Z0-9_-]{1,64}$`, and the server prefix keeps two servers' `search` tools from colliding.

### Keeping a session alive

The SDK's transports are anyio task groups. They must be entered and exited **in the same asyncio task**. So each server gets its own owner task, which opens the transport, publishes the session through a future, and then waits for shutdown:

`grok_bot/mcp_client.py` (lines 159–186):

```python
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
```

`start()` waits on that future with the server's `timeout`. A server that fails or hangs is logged and skipped (status `error: ...`), so a broken MCP server never stops the app from booting. The FastAPI lifespan calls `await mcp.start()` on startup and `await mcp.close()` on shutdown.

### Turning MCP tools into ordinary tools

`grok_bot/mcp_client.py` (lines 209–233):

```python
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
```

Each MCP tool becomes a normal `Tool`, so the agent loop, the output cap and error handling all apply unchanged. Each call has a per-server read timeout. `admin_only` servers are filtered per user. Here is where that happens, in `grok_bot/app.py` when a chat turn is built:

`grok_bot/app.py` (lines 868–871):

```python
        turn_tools, recalled = ToolRegistry([*tools, *mcp.tools_for(user)]), None
        if memory is not None and req.memory:
            recalled = await memory.recall(user["id"], req.message)
            turn_tools = ToolRegistry([*turn_tools, *memory.tools(user["id"])])
```

`GET /api/tools` lists the same filtered set and labels each tool with `"source": "builtin"` or `"mcp:<server>"`.

**Known limit:** there is no automatic reconnect. If a server dies mid-run, its tools return `MCP server 'x' is not connected` (or a transport error) until the app restarts.

## Try it

```bash
.venv/bin/pip install -e ".[dev]"     # dev includes the mcp SDK
.venv/bin/pytest -q tests/test_agent.py tests/test_tools.py tests/test_mcp.py
```

`tests/mcp_echo_server.py` is a complete MCP server in ~30 lines, built with the SDK's `FastMCP`. Point the bot at it:

```bash
cat > mcp.json <<'JSON'
{"servers": {"echo": {"command": ".venv/bin/python", "args": ["tests/mcp_echo_server.py"]}}}
JSON
.venv/bin/python - <<'PY'
import asyncio
from grok_bot.mcp_client import MCPManager

async def main():
    m = MCPManager.from_file("mcp.json")
    await m.start()
    print(m.status, [t.name for t in m.tools_for(None)])
    print(await m.tools_for(None)[0].func(a=2, b=3))
    await m.close()

asyncio.run(main())
PY
```

Expected output (the SDK also logs `Processing request ...` lines to stderr):

```text
{'echo': 'ok'} ['echo__add', 'echo__boom', 'echo__slow', 'echo__weird_name']
5
```

Note `weird.name` became `echo__weird_name`. Then run `python -m grok_bot`, ask "use echo add to add 2 and 3", and watch the `tool_call` event.

## Exercises

1. Add a `calculator` tool that evaluates arithmetic with `ast` (no `eval`). Register it in `default_registry` and write a `FakeLLM` test using `tool_turn("calculator", ...)`.
2. `check_public_url` resolves DNS, and then `httpx` resolves it again. That gap is a DNS-rebinding window. How would you close it? (Hint: connect to the IP you checked, and send the original `Host` header.)
3. Run independent tool calls from one step concurrently with `asyncio.gather`. What must stay ordered for the `role: "tool"` messages?
4. Add reconnect to `MCPManager`: when `_own` exits unexpectedly, restart it with backoff. Make sure `close()` still finishes promptly.
5. Add an `allow_tools: [...]` list per MCP server, so an operator can expose only some of a server's tools.
