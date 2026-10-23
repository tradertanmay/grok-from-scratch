"""Run model-written Python in isolation.

Backends (`GROK_SANDBOX`):
- `docker`: throwaway container, no network, read-only FS, CPU/memory/pid caps (real isolation).
- `subprocess`: separate process with timeout + rlimits in a temp dir (NOT a security sandbox).
- `auto` (default): docker if the daemon is reachable, else subprocess.
- `off`: python_exec is not offered to the model at all.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from functools import cache

from .base import Tool

MAX_OUTPUT = 8_000
SANDBOX_IMAGE = os.environ.get("GROK_SANDBOX_IMAGE", "python:3.11-slim")


def _limit_resources(memory_mb: int, cpu_seconds: int):
    def apply():
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            if sys.platform.startswith("linux"):
                limit = memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ImportError, ValueError, OSError):
            pass

    return apply


@cache
def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def sandbox_backend() -> str:
    mode = os.environ.get("GROK_SANDBOX", "auto").lower()
    if mode == "auto":
        return "docker" if docker_available() else "subprocess"
    if mode not in {"docker", "subprocess"}:
        raise ValueError(f"GROK_SANDBOX must be auto, docker or subprocess, not {mode!r}")
    return mode


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


def _format(out: bytes, returncode: int | None) -> str:
    text = out.decode(errors="replace")
    if len(text) > MAX_OUTPUT:
        text = text[:MAX_OUTPUT] + "\n...[truncated]"
    status = "" if returncode == 0 else f"\n[exit code {returncode}]"
    return (text or "(no output)") + status


async def _run_docker(code: str, timeout: float, memory_mb: int) -> str:
    name = f"grok-py-{os.getpid()}-{os.urandom(4).hex()}"
    proc = await asyncio.create_subprocess_exec(
        *docker_command(name, memory_mb),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(code.encode()), timeout=timeout)
    except TimeoutError:
        kill = await asyncio.create_subprocess_exec(
            "docker", "kill", name,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )  # fmt: skip
        await kill.wait()
        proc.kill()
        await proc.wait()
        return f"Error: timed out after {timeout}s"
    return _format(out, proc.returncode)


async def _run_subprocess(code: str, timeout: float, memory_mb: int) -> str:
    limits = _limit_resources(memory_mb, int(timeout) + 1) if os.name == "posix" else None
    with tempfile.TemporaryDirectory(prefix="grok_py_") as cwd:
        env = {"PATH": os.environ.get("PATH", ""), "HOME": cwd, "PYTHONIOENCODING": "utf-8"}
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-c", code,
            cwd=cwd, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            preexec_fn=limits,
        )  # fmt: skip
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Error: timed out after {timeout}s"
    return _format(out, proc.returncode)


async def run_python(
    code: str, timeout: float = 15, memory_mb: int = 512, backend: str | None = None
) -> str:
    backend = backend or sandbox_backend()
    if backend == "docker":
        return await _run_docker(code, timeout, memory_mb)
    return await _run_subprocess(code, timeout, memory_mb)


def python_exec_tool() -> Tool:
    async def python_exec(code: str) -> str:
        return await run_python(code)

    return Tool(
        name="python_exec",
        description=(
            "Execute Python 3 code and return stdout/stderr. Use print() to show results. "
            "Standard library only; no network access; 15s timeout."
        ),
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source code"}},
            "required": ["code"],
        },
        func=python_exec,
    )
