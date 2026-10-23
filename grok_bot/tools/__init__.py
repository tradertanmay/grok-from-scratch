import os

from .base import Tool, ToolRegistry
from .python_exec import python_exec_tool
from .web import fetch_url_tool, web_search_tool


def default_registry(searxng_url: str = "") -> ToolRegistry:
    tools = [web_search_tool(searxng_url), fetch_url_tool()]
    if os.environ.get("GROK_SANDBOX", "auto").lower() != "off":
        tools.append(python_exec_tool())
    return ToolRegistry(tools)


__all__ = ["Tool", "ToolRegistry", "default_registry"]
