"""Black-box eval harness: runs scripted chats against a live grok-bot server."""

from .runner import Case, Result, check_answer, load_cases, run_suite

__all__ = ["Case", "Result", "check_answer", "load_cases", "run_suite"]
