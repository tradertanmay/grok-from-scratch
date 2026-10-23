"""Concurrent load-test CLI harness for Grok Bot.

Measures TTFT (time-to-first-token), total turn latency, error rate, and
throughput (turns/min) across N concurrent users hitting `POST /v1/chat/completions`
with streaming SSE enabled. Supports `--keys-file` for invite/closed deployments,
respects `Retry-After` on 429 during signup and cleanup, detects mid-stream SSE
errors and empty replies, and aborts if run against an empty DB where the first
throwaway account would become the sole admin.

Usage:
    python -m grok_bot.loadtest --url http://127.0.0.1:8000 --users 20 --model grok-regular
    python -m grok_bot.loadtest --url https://grok.example.com --keys-file keys.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(len(sorted_vals) * pct)))
    return sorted_vals[idx]


def load_keys_file(path: str | Path) -> list[str]:
    """Load personal `gk_...` API keys (one per line, `#` comments ignored)."""
    keys: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        keys.append(line)
    if not keys:
        raise ValueError(f"No API keys found in {path}")
    return keys


async def _request_with_retry_after(
    http: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    max_retries: int = 8,
    **kwargs: Any,
) -> httpx.Response:
    """Send an HTTP request, honoring `Retry-After` on HTTP 429."""
    for attempt in range(max_retries + 1):
        resp = await http.request(method, url, **kwargs)
        if resp.status_code != 429 or attempt == max_retries:
            return resp
        retry_hdr = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
        try:
            wait_s = max(float(retry_hdr), 0.05) if retry_hdr is not None else 1.0
        except ValueError:
            wait_s = 1.0
        await sleep_fn(wait_s)
    return resp


async def run_load_test(
    *,
    base_url: str = "http://127.0.0.1:8000",
    users: int = 20,
    requests_per_user: int = 1,
    model: str = "grok-regular",
    prompt_template: str = "Reply in one short sentence ({i}): What is {i} + {i}?",
    timeout: float = 180.0,
    cleanup: bool = True,
    keys: list[str] | None = None,
    keys_file: str | Path | None = None,
    client: httpx.AsyncClient | None = None,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, Any]:
    """Run a concurrent streaming load test against Grok Bot."""
    own_client = client is None
    http = client or httpx.AsyncClient(base_url=base_url, timeout=timeout)
    run_id = secrets.token_hex(3)
    password = f"pw_{secrets.token_hex(8)}"
    created_accounts: list[dict[str, Any]] = []
    undeleted_users: list[str] = []

    provided_keys: list[str] | None = list(keys) if keys is not None else None
    if provided_keys is None and keys_file is not None:
        provided_keys = load_keys_file(keys_file)

    try:
        if provided_keys is not None:
            if len(provided_keys) < users:
                raise ValueError(
                    f"Provided {len(provided_keys)} API keys, but users={users} requested"
                )
            active_keys = provided_keys[:users]
        else:
            health_r = await http.get("/api/health")
            if health_r.status_code == 200:
                signup_mode = health_r.json().get("signup", "open")
                if signup_mode in ("invite", "closed"):
                    raise RuntimeError(
                        f"Server signup mode is '{signup_mode}'; "
                        "pass --keys-file with existing gk_... keys"
                    )

            active_keys = []
            for i in range(users):
                username = f"lt_{run_id}_{i}"
                r = await _request_with_retry_after(
                    http,
                    "POST",
                    "/api/auth/signup",
                    sleep_fn=sleep_fn,
                    json={"username": username, "password": password},
                )
                if r.status_code != 200:
                    raise RuntimeError(
                        f"Signup failed for {username}: HTTP {r.status_code} {r.text}"
                    )
                body = r.json()
                user_obj = body.get("user") or {}
                cookie = r.cookies.get("grok_session") or ""
                is_admin = bool(user_obj.get("is_admin"))
                created_accounts.append(
                    {
                        "username": username,
                        "cookie": cookie,
                        "is_admin": is_admin,
                    }
                )
                if is_admin:
                    raise RuntimeError(
                        f"First throwaway user '{username}' (password: '{password}') became the "
                        "sole admin on an empty DB and cannot be auto-deleted. Delete the fresh "
                        "SQLite DB file (or sign in with those credentials) and create a real "
                        "admin user first, or pass --keys-file."
                    )
                rk = await _request_with_retry_after(
                    http,
                    "POST",
                    "/api/keys",
                    sleep_fn=sleep_fn,
                    json={"name": f"loadtest-{i}"},
                    headers={"Cookie": f"grok_session={cookie}"},
                )
                if rk.status_code != 200:
                    raise RuntimeError(f"Key creation failed for {username}: HTTP {rk.status_code}")
                active_keys.append(rk.json()["key"])

        async def _single_turn(user_idx: int, req_idx: int, api_key: str) -> dict[str, Any]:
            seq = user_idx * requests_per_user + req_idx + 1
            prompt = prompt_template.format(i=seq, user=user_idx, req=req_idx)
            t0 = time.perf_counter()
            ttft: float | None = None
            chunks = 0
            chars = 0
            finish_reason: str | None = None
            stream_error: str | None = None
            try:
                async with http.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True,
                    },
                ) as resp:
                    if resp.status_code != 200:
                        body_bytes = await resp.aread()
                        return {
                            "user": user_idx,
                            "ok": False,
                            "status": resp.status_code,
                            "error": body_bytes.decode("utf-8", errors="replace")[:200],
                            "total": time.perf_counter() - t0,
                        }
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        try:
                            obj = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        if "error" in obj and obj["error"]:
                            err_val = obj["error"]
                            stream_error = (
                                str(err_val.get("message") or err_val)
                                if isinstance(err_val, dict)
                                else str(err_val)
                            )
                            break
                        choice = (obj.get("choices") or [{}])[0]
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
                        delta = choice.get("delta") or {}
                        piece = (delta.get("content") or "") + (
                            delta.get("reasoning_content") or ""
                        )
                        if piece:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            chunks += 1
                            chars += len(piece)
                total = time.perf_counter() - t0
                if stream_error is not None:
                    return {
                        "user": user_idx,
                        "ok": False,
                        "status": 502,
                        "error": stream_error[:200],
                        "total": total,
                    }
                if ttft is None or chars == 0 or not finish_reason:
                    return {
                        "user": user_idx,
                        "ok": False,
                        "status": 200,
                        "empty": chars == 0,
                        "error": "empty stream or missing finish_reason",
                        "total": total,
                    }
                return {
                    "user": user_idx,
                    "ok": True,
                    "status": 200,
                    "ttft": ttft,
                    "total": total,
                    "chunks": chunks,
                    "chars": chars,
                }
            except Exception as exc:
                return {
                    "user": user_idx,
                    "ok": False,
                    "status": 0,
                    "error": str(exc),
                    "total": time.perf_counter() - t0,
                }

        async def _user_worker(user_idx: int, api_key: str) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for r_idx in range(requests_per_user):
                out.append(await _single_turn(user_idx, r_idx, api_key))
            return out

        wall_t0 = time.perf_counter()
        nested = await asyncio.gather(
            *(_user_worker(idx, key) for idx, key in enumerate(active_keys))
        )
        wall_s = max(time.perf_counter() - wall_t0, 1e-6)
        results = [item for sub in nested for item in sub]

        ok_res = [r for r in results if r["ok"]]
        err_res = [r for r in results if not r["ok"]]
        empty_count = sum(1 for r in err_res if r.get("empty"))
        ttfts = sorted(r["ttft"] for r in ok_res)
        totals = sorted(r["total"] for r in ok_res)

        return {
            "users": users,
            "requests_per_user": requests_per_user,
            "total_requests": len(results),
            "succeeded": len(ok_res),
            "failed": len(err_res),
            "empty_responses": empty_count,
            "model": model,
            "wall_seconds": round(wall_s, 3),
            "throughput_turns_per_min": round(len(ok_res) / wall_s * 60.0, 2),
            "ttft_seconds": {
                "min": round(ttfts[0], 3) if ttfts else 0.0,
                "p50": round(statistics.median(ttfts), 3) if ttfts else 0.0,
                "p95": round(_percentile(ttfts, 0.95), 3) if ttfts else 0.0,
                "max": round(ttfts[-1], 3) if ttfts else 0.0,
            },
            "total_seconds": {
                "min": round(totals[0], 3) if totals else 0.0,
                "p50": round(statistics.median(totals), 3) if totals else 0.0,
                "p95": round(_percentile(totals, 0.95), 3) if totals else 0.0,
                "max": round(totals[-1], 3) if totals else 0.0,
            },
            "errors": err_res[:5],
            "undeleted_users": undeleted_users,
        }
    finally:
        if cleanup and created_accounts:
            for acc in created_accounts:
                try:
                    del_r = await _request_with_retry_after(
                        http,
                        "DELETE",
                        "/api/account",
                        sleep_fn=sleep_fn,
                        json={"password": password},
                        headers={"Cookie": f"grok_session={acc['cookie']}"},
                    )
                    if del_r.status_code != 200:
                        undeleted_users.append(acc["username"])
                except Exception:
                    undeleted_users.append(acc["username"])
            if undeleted_users:
                print(
                    f"WARNING: Could not auto-delete throwaway accounts: {undeleted_users}",
                    file=sys.stderr,
                )
        if own_client:
            await http.aclose()


def format_summary(s: dict[str, Any]) -> str:
    ttft = s["ttft_seconds"]
    tot = s["total_seconds"]
    lines = [
        f"=== Grok Bot Load Test ({s['model']}) ===",
        (
            f"CONCURRENT_USERS: {s['users']} | REQUESTS: {s['succeeded']}/{s['total_requests']} "
            f"(failed={s['failed']}, empty={s.get('empty_responses', 0)}) "
            f"| WALL_TIME: {s['wall_seconds']:.2f}s"
        ),
        (
            f"TTFT   -> min={ttft['min']:.2f}s | p50={ttft['p50']:.2f}s "
            f"| p95={ttft['p95']:.2f}s | max={ttft['max']:.2f}s"
        ),
        (
            f"TOTAL  -> min={tot['min']:.2f}s | p50={tot['p50']:.2f}s "
            f"| p95={tot['p95']:.2f}s | max={tot['max']:.2f}s"
        ),
        f"THROUGHPUT -> {s['throughput_turns_per_min']:.1f} turns/min",
    ]
    if s.get("errors"):
        lines.append(f"ERRORS (first {len(s['errors'])}): {s['errors']}")
    if s.get("undeleted_users"):
        lines.append(f"UNDELETED_USERS: {s['undeleted_users']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m grok_bot.loadtest",
        description="Run a concurrent streaming load test against Grok Bot (/v1/chat/completions).",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Grok Bot base URL")
    parser.add_argument(
        "--users",
        type=int,
        default=None,
        help="Number of concurrent users (default: 20 or len(keys))",
    )
    parser.add_argument(
        "--requests-per-user", type=int, default=1, help="Sequential turns per concurrent user"
    )
    parser.add_argument("--model", default="grok-regular", help="Model id for /v1/chat/completions")
    parser.add_argument(
        "--keys-file",
        dest="keys_file",
        help="Path to file with existing gk_... API keys (one per line)",
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout (s)")
    parser.add_argument("--json", dest="json_path", help="Optional path to write JSON report")
    parser.add_argument(
        "--no-cleanup", action="store_true", help="Keep throwaway load-test accounts"
    )
    args = parser.parse_args(argv)

    keys = load_keys_file(args.keys_file) if args.keys_file else None
    users = args.users if args.users is not None else (len(keys) if keys is not None else 20)

    summary = asyncio.run(
        run_load_test(
            base_url=args.url,
            users=users,
            requests_per_user=args.requests_per_user,
            model=args.model,
            timeout=args.timeout,
            cleanup=not args.no_cleanup,
            keys=keys,
        )
    )
    print(format_summary(summary))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
