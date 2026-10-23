import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

from grok_bot.app import create_app
from grok_bot.config import Settings
from grok_bot.mcp_client import MCPManager, ServerConfig, load_config, tool_name
from grok_bot.store import Store
from grok_bot.tools import ToolRegistry
from tests.conftest import FakeLLM, text_turn, tool_turn
from tests.test_auth import signup

ECHO = str(Path(__file__).with_name("mcp_echo_server.py"))


def echo(name="echo", **kw) -> ServerConfig:
    return ServerConfig(name, command=sys.executable, args=[ECHO], **kw)


def test_tool_name_is_openai_safe():
    assert tool_name("fs", "read.file") == "fs__read_file"
    assert len(tool_name("s" * 40, "t" * 40)) == 64


def test_load_config(tmp_path):
    assert load_config(tmp_path / "missing.json") == []
    p = tmp_path / "mcp.json"
    p.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fs": {"command": "npx", "args": ["-y", "x"], "admin_only": True},
                    "docs": {"url": "https://h/mcp", "headers": {"Authorization": "Bearer ${TOK}"}},
                    "off": {"command": "x", "enabled": False},
                }
            }
        )
    )
    fs, docs, off = load_config(p, env={"TOK": "s3cret"})
    assert fs.command == "npx" and fs.args == ["-y", "x"] and fs.admin_only
    assert docs.headers == {"Authorization": "Bearer s3cret"} and not docs.admin_only
    assert MCPManager([fs, docs, off]).servers == [fs, docs]
    p.write_text(json.dumps({"servers": {"bad": {"command": "x", "url": "https://h"}}}))
    with pytest.raises(ValueError, match="exactly one"):
        load_config(p)


@pytest.mark.anyio
async def test_stdio_server_tools_and_failures():
    m = MCPManager([echo(timeout=2), ServerConfig("nope", command="/does/not/exist")])
    await m.start()
    try:
        assert m.status["echo"] == "ok" and m.status["nope"].startswith("error")
        reg = ToolRegistry(m.tools_for(None))
        names = [t.name for t in reg]
        assert names == ["echo__add", "echo__boom", "echo__slow", "echo__weird_name"]
        add = next(iter(reg))
        assert add.description.startswith("[MCP echo] Add two numbers")
        assert set(add.parameters["properties"]) == {"a", "b"}
        assert (await reg.call("echo__add", {"a": 2, "b": 3})).output == "5"
        assert (await reg.call("echo__weird_name", {})).output == "weird ok"
        boom = await reg.call("echo__boom", {})
        assert boom.is_error and "kaboom" in boom.output
        slow = await reg.call("echo__slow", {})
        assert slow.is_error and "Timed out" in slow.output
    finally:
        await m.close()
    assert m.tools_for(None) == []


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.anyio
async def test_streamable_http_server():
    from tests.mcp_echo_server import srv

    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(srv.streamable_http_app(), port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    m = MCPManager([ServerConfig("remote", url=f"http://127.0.0.1:{port}/mcp")])
    try:
        await m.start()
        assert m.status["remote"] == "ok"
        out = await ToolRegistry(m.tools_for(None)).call("remote__add", {"a": 40, "b": 2})
        assert out.output == "42"
    finally:
        await m.close()
        server.should_exit = True
        thread.join(5)


def test_chat_uses_mcp_tools_and_admin_only(tmp_path):
    llm = FakeLLM(
        [tool_turn("echo__add", '{"a": 2, "b": 3}'), text_turn("2 + 3 = 5"), text_turn("hi")]
    )
    app = create_app(
        Settings(llm_model="fake", embed_model=""),
        llm=llm,
        store=Store(str(tmp_path / "t.db")),
        tools=ToolRegistry([]),
        mcp=MCPManager([echo(), echo("adm", admin_only=True)]),
    )
    with TestClient(app) as c:
        assert {t["source"] for t in c.get("/api/tools").json()} == {"mcp:echo"}
        signup(c, "boss")
        admin_tools = {t["name"]: t["source"] for t in c.get("/api/tools").json()}
        assert admin_tools["adm__add"] == "mcp:adm" and admin_tools["echo__add"] == "mcp:echo"

        r = c.post("/api/chat", json={"message": "what is 2+3?"})
        assert r.status_code == 200 and "2 + 3 = 5" in r.text
        tool_msg = next(m for m in llm.calls[1]["messages"] if m["role"] == "tool")
        assert tool_msg["content"] == "5"
        assert "adm__add" in {t["function"]["name"] for t in llm.calls[0]["tools"]}

        c.cookies.clear()
        signup(c, "bob")
        bob_tools = {t["name"] for t in c.get("/api/tools").json()}
        assert "echo__add" in bob_tools and not any(n.startswith("adm__") for n in bob_tools)
        assert c.post("/api/chat", json={"message": "hey"}).status_code == 200
        bob_llm_tools = {t["function"]["name"] for t in llm.calls[2]["tools"]}
        assert "echo__add" in bob_llm_tools and "adm__add" not in bob_llm_tools
