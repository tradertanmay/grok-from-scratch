"""Tiny MCP server for tests: `python tests/mcp_echo_server.py` speaks stdio."""

import asyncio
import sys

from mcp.server.fastmcp import FastMCP

srv = FastMCP("echo")


@srv.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@srv.tool()
def boom() -> str:
    """Always fails."""
    raise ValueError("kaboom")


@srv.tool()
async def slow() -> str:
    """Takes 5 seconds."""
    await asyncio.sleep(5)
    return "late"


@srv.tool(name="weird.name")
def weird() -> str:
    """Has a dot in its name."""
    return "weird ok"


if __name__ == "__main__":
    srv.run(sys.argv[1] if len(sys.argv) > 1 else "stdio")
