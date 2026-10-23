"""python -m grok_bot.evals --url http://localhost:8000 [--model mac/qwen3.6:35b]"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from dataclasses import asdict

import httpx

from .runner import DEFAULT_CASES, Result, load_cases, run_suite

MARK = {"pass": "PASS", "fail": "FAIL", "skip": "skip", "error": "ERR "}


def show(r: Result) -> None:
    ttft = f"{r.first_token_s:5.1f}s" if r.first_token_s is not None else "   -  "
    print(f"{MARK[r.status]}  {r.category:<10} {r.id:<24} {r.seconds:6.1f}s  ttft {ttft}")
    for reason in r.reasons:
        print(f"      - {reason}")
    if r.status == "fail" and r.answer:
        print("      > " + r.answer[:200].replace("\n", " "))


def summary(results: list[Result]) -> str:
    counts = Counter(r.status for r in results)
    graded = counts["pass"] + counts["fail"] + counts["error"]
    rate = counts["pass"] / graded if graded else 0.0
    by_cat: dict[str, list[int]] = {}
    for r in results:
        if r.status != "skip":
            c = by_cat.setdefault(r.category, [0, 0])
            c[0] += r.status == "pass"
            c[1] += 1
    cats = "  ".join(f"{k} {p}/{n}" for k, (p, n) in sorted(by_cat.items()))
    return f"\n{counts['pass']}/{graded} passed ({rate:.0%}), {counts['skip']} skipped\n{cats}"


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grok_bot.evals")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--in-process", action="store_true", help="run against in-process FastAPI app")
    ap.add_argument("--model", help="model or provider/model (default: server default)")
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--only", help="comma-separated categories or case ids")
    ap.add_argument("--fast", action="store_true", help="skip slow cases (DeepSearch, web)")
    ap.add_argument("--think", action="store_true", help="enable Think mode by default")
    ap.add_argument("--json", help="write full results to this file")
    ap.add_argument("--min-pass", type=float, default=0.0, help="exit 1 below this pass rate")
    args = ap.parse_args(argv)

    cases = load_cases(args.cases)
    if args.only:
        want = set(args.only.split(","))
        cases = [c for c in cases if c.id in want or c.category in want]
    if args.fast:
        cases = [c for c in cases if not c.slow]
    target = "in-process" if args.in_process else args.url
    print(f"Running {len(cases)} cases against {target} model={args.model or 'default'}\n")
    if args.in_process:
        import tempfile

        from ..app import create_app
        from ..store import Store

        with tempfile.TemporaryDirectory() as tmp:
            transport = httpx.ASGITransport(app=create_app(store=Store(f"{tmp}/eval.db")))
            async with httpx.AsyncClient(
                transport=transport, base_url="http://eval", timeout=600
            ) as client:
                results = await run_suite(client, cases, args.model, args.think, on_result=show)
    else:
        async with httpx.AsyncClient(base_url=args.url, timeout=600) as client:
            results = await run_suite(client, cases, args.model, args.think, on_result=show)
    print(summary(results))
    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
    graded = [r for r in results if r.status != "skip"]
    rate = sum(r.status == "pass" for r in graded) / len(graded) if graded else 1.0
    return 1 if rate < args.min_pass else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
