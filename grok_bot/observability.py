"""Prometheus-style /metrics counters and JSON access logs, no extra dependencies."""

import json
import logging
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger("grok_bot.access")


def _labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    esc = (v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") for _, v in labels)
    return "{" + ",".join(f'{k}="{v}"' for (k, _), v in zip(labels, esc, strict=True)) + "}"


class Metrics:
    HELP = {
        "grok_http_requests_total": ("counter", "HTTP requests by route, method and status"),
        "grok_http_request_seconds_sum": ("counter", "Total seconds spent before response start"),
        "grok_chat_turns_total": ("counter", "Chat turns by model and outcome"),
        "grok_chat_turn_seconds_sum": ("counter", "Total seconds spent streaming chat turns"),
        "grok_chat_tool_calls_total": ("counter", "Tool calls made during chat turns"),
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.started = time.time()

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[name][key] += value

    def turn(self, model: str, outcome: str, seconds: float, tool_calls: int) -> None:
        self.inc("grok_chat_turns_total", model=model, outcome=outcome)
        self.inc("grok_chat_turn_seconds_sum", seconds, model=model)
        if tool_calls:
            self.inc("grok_chat_tool_calls_total", tool_calls, model=model)

    def render(self, gauges: dict[str, float] | None = None) -> str:
        out = []
        with self._lock:
            snapshot = {n: dict(v) for n, v in self._values.items()}
        for name in sorted(snapshot):
            kind, text = self.HELP.get(name, ("counter", name))
            out += [f"# HELP {name} {text}", f"# TYPE {name} {kind}"]
            out += [f"{name}{_labels(k)} {v:g}" for k, v in sorted(snapshot[name].items())]
        for name, value in sorted(
            {"grok_uptime_seconds": time.time() - self.started, **(gauges or {})}.items()
        ):
            out += [f"# TYPE {name} gauge", f"{name} {value:g}"]
        return "\n".join(out) + "\n"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_json_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers[:] = []
        logging.getLogger(name).propagate = True
    logging.getLogger("uvicorn.access").disabled = True


def install(app, metrics: Metrics) -> None:
    @app.middleware("http")
    async def observe(request, call_next):
        started = time.monotonic()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            seconds = time.monotonic() - started
            route = request.scope.get("route")
            path = getattr(route, "path", None) or "<static>"
            metrics.inc(
                "grok_http_requests_total", method=request.method, route=path, status=str(status)
            )
            metrics.inc("grok_http_request_seconds_sum", seconds, route=path)
            if path != "/metrics":
                log.info(
                    "%s %s %s",
                    request.method,
                    path,
                    status,
                    extra={
                        "fields": {
                            "method": request.method,
                            "route": path,
                            "status": status,
                            "ms": round(seconds * 1000, 1),
                            "client": request.client.host if request.client else None,
                        }
                    },
                )
