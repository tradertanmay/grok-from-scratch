"""Interactive terminal CLI for Grok Bot (`python -m grok_bot.cli`)."""

import argparse
import asyncio
import json
import sys
from typing import Any

from .agent import Agent
from .config import Settings
from .llm import LLMClient
from .persona import system_prompt
from .providers import ProviderRouter, load_providers
from .store import Store
from .tools import ToolRegistry, default_registry

BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def format_cli_event(event: dict[str, Any]) -> str:
    """Format a single Agent event for terminal display."""
    etype = event.get("type")
    if etype == "think":
        return f"{DIM}{MAGENTA}{event.get('content', '')}{RESET}"
    if etype == "token":
        return str(event.get("content", ""))
    if etype == "tool_call":
        args = json.dumps(event.get("arguments", {}), ensure_ascii=False)
        return f"\n{YELLOW}⚡ [tool_call] {event.get('name')}({args}){RESET}\n"
    if etype == "tool_result":
        status = f"{RED}ERROR" if event.get("is_error") else f"{GREEN}OK"
        preview = str(event.get("output", "")).strip().replace("\n", " ")[:160]
        return f"{DIM}   ↳ [{status}{RESET}{DIM}] {preview}{RESET}\n"
    if etype == "error":
        return f"\n{RED}✖ Error: {event.get('message')}{RESET}\n"
    return ""


async def run_single_turn(
    prompt: str,
    session_id: str,
    store: Store,
    llm: LLMClient,
    tools: ToolRegistry,
    model: str,
    max_steps: int = 8,
) -> str:
    """Run one user turn through the Agent, persisting messages and streaming to stdout."""
    history = [
        {
            k: m[k]
            for k in ("role", "content", "tool_calls", "tool_call_id", "name")
            if m.get(k) is not None
        }
        for m in store.get_messages(session_id)
    ]
    user_msg = {"role": "user", "content": prompt}
    store.add_message(session_id, user_msg)
    messages = [{"role": "system", "content": system_prompt()}, *history, user_msg]

    agent = Agent(llm, tools, model, max_steps)
    saved = 0
    collected: list[str] = []

    async for event in agent.run(messages):
        for m in agent.new_messages[saved:]:
            store.add_message(session_id, m)
        saved = len(agent.new_messages)
        rendered = format_cli_event(event)
        if rendered:
            sys.stdout.write(rendered)
            sys.stdout.flush()
        if event.get("type") == "token":
            collected.append(str(event.get("content", "")))

    for m in agent.new_messages[saved:]:
        store.add_message(session_id, m)

    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(collected)


async def interactive_loop(settings: Settings, initial_prompt: str | None = None) -> int:
    llm = ProviderRouter(load_providers(settings), list(settings.fallback_models))
    store = Store(settings.db_path)
    tools = default_registry(settings.searxng_url)
    model = settings.llm_model
    session_id = store.create_session("CLI Session")

    if initial_prompt:
        await run_single_turn(
            initial_prompt, session_id, store, llm, tools, model, settings.max_tool_steps
        )
        return 0

    print(f"{BOLD}{CYAN}=== GROK BOT CLI (model: {model}) ==={RESET}")
    print(f"{DIM}Commands: /new, /model <name>, /sessions, /quit{RESET}\n")

    while True:
        try:
            line = input(f"{BOLD}{GREEN}You > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0

        if not line:
            continue
        if line in {"/quit", "/exit", ":q"}:
            return 0
        if line == "/new":
            session_id = store.create_session("CLI Session")
            print(f"{DIM}Started new session: {session_id}{RESET}")
            continue
        if line.startswith("/model "):
            model = line.split(" ", 1)[1].strip()
            print(f"{DIM}Switched model to: {model}{RESET}")
            continue
        if line == "/sessions":
            for s in store.list_sessions()[:10]:
                print(f"  {s['id']}  {s['title']}")
            continue

        sys.stdout.write(f"{BOLD}{CYAN}Grok > {RESET}")
        sys.stdout.flush()
        await run_single_turn(line, session_id, store, llm, tools, model, settings.max_tool_steps)


def main(argv: list[str] | None = None) -> int:
    from dataclasses import replace

    parser = argparse.ArgumentParser(description="Grok Bot Interactive Terminal CLI")
    parser.add_argument("-p", "--prompt", help="Run a single prompt non-interactively and exit")
    parser.add_argument("-m", "--model", help="Override LLM model name")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.model:
        settings = replace(settings, llm_model=args.model)

    return asyncio.run(interactive_loop(settings, args.prompt))


if __name__ == "__main__":
    raise SystemExit(main())
