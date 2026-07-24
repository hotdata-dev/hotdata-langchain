"""End-to-end, through the real LangChain StructuredTool that make_hotdata_tools builds.

Runs N calls alternating TPCH Q1/Q5 against real sf=1 data, once per cache arm, measured
cold (first-ever write included) and then warm (steady state, every call a hit).

cached() fails open, so a cache backend that is silently broken shows up as a suspiciously
fast arm rather than an error. This installs a log handler to count swallowed failures and
flags any arm they contaminated.

    uv run python -m benchmarks.bench_endtoend
    uv run python -m benchmarks.bench_endtoend --calls 20
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import hotdata_framework as hf

import hotdata_langchain as hl
from benchmarks.backends import LayeredToolCache, SqliteToolCache
from benchmarks.harness import fmt, header, memoize_resolve
from benchmarks.tpch import CACHE_DB_NAME, Q1, Q5, TPCH_DB_NAME, resolve_database_id
from hotdata_langchain.cache import HotdataToolCache


class SwallowedFailureCounter(logging.Handler):
    """Count the warnings cached() emits when it fails open, so they cannot hide."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def take(self) -> list[str]:
        out = list(self.messages)
        self.messages.clear()
        return out


def run_arm(tool: Any, n_calls: int, counter: SwallowedFailureCounter) -> tuple[list[float], int]:
    """Invoke ``tool`` ``n_calls`` times alternating Q1/Q5; return per-call times."""
    counter.take()
    calls = []
    for i in range(n_calls):
        sql = Q1 if i % 2 == 0 else Q5
        start = time.perf_counter()
        out = tool.invoke({"sql": sql})
        calls.append(time.perf_counter() - start)
        if not isinstance(out, str) or len(out) < 50:
            raise SystemExit(f"tool returned a suspiciously small result: {out!r:.200}")
    return calls, len(counter.take())


def report(label: str, calls: list[float], failures: int) -> float:
    total = sum(calls)
    flag = f"  [{failures} cache op(s) FAILED OPEN]" if failures else ""
    print(f"  {label:<34} total={total:7.2f}s   " + " ".join(f"{c:5.2f}" for c in calls) + flag)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calls", type=int, default=12)
    ap.add_argument("--tpch-db", default=TPCH_DB_NAME)
    ap.add_argument("--cache-db", default=CACHE_DB_NAME)
    args = ap.parse_args()

    counter = SwallowedFailureCounter()
    logging.getLogger("hotdata_langchain.cache").addHandler(counter)

    client = hf.from_env()
    tpch_db = resolve_database_id(client, args.tpch_db)
    cache_db = resolve_database_id(client, args.cache_db)
    tmp = Path(tempfile.mkdtemp())
    version = f"e2e-{int(time.time())}"  # fresh each run, so every arm starts cold
    n = args.calls

    header(f"END-TO-END: {n} calls alternating TPCH Q1/Q5, real sf=1 data", width=118)
    print(
        "  Call 1 of the uncached arm absorbs this process's cold start (KEDA scale-from-"
        "zero,\n  auth, TLS). Compare the warm rows for steady state."
    )
    results: dict[str, tuple[float, float]] = {}

    print("\n[A] NO CACHE")
    calls, fails = run_arm(hl.make_hotdata_tools(client, database=tpch_db)[0], n, counter)
    base = report("uncached", calls, fails)
    results["uncached"] = (base, base)

    print("\n[B] SQLITE -- local file, invisible to other processes")
    lite = SqliteToolCache(tmp / "b.db", version=version)
    tool_b = hl.make_hotdata_tools(client, database=tpch_db, cache=lite)[0]
    cold = report("sqlite cold", *run_arm(tool_b, n, counter))
    warm = report("sqlite warm", *run_arm(tool_b, n, counter))
    results["sqlite"] = (cold, warm)

    print("\n[C] HOTDATA -- shared managed table, as shipped (2 round trips per hit)")
    hot = HotdataToolCache(client, database_id=cache_db, version=version)
    tool_c = hl.make_hotdata_tools(client, database=tpch_db, cache=hot)[0]
    cold = report("hotdata cold", *run_arm(tool_c, n, counter))
    warm = report("hotdata warm", *run_arm(tool_c, n, counter))
    results["hotdata"] = (cold, warm)

    print("\n[D] HOTDATA + resolve memoized -- 1 round trip per hit")
    undo = memoize_resolve(client)
    hot_d = HotdataToolCache(client, database_id=cache_db, version=version + "-d")
    tool_d = hl.make_hotdata_tools(client, database=tpch_db, cache=hot_d)[0]
    cold = report("hotdata+memo cold", *run_arm(tool_d, n, counter))
    warm = report("hotdata+memo warm", *run_arm(tool_d, n, counter))
    results["hotdata+memo"] = (cold, warm)

    print("\n[E] LAYERED -- sqlite local tier in front of the shared hotdata tier")
    layered = LayeredToolCache(
        SqliteToolCache(tmp / "e.db", version=version + "-e"),
        HotdataToolCache(client, database_id=cache_db, version=version + "-e"),
    )
    tool_e = hl.make_hotdata_tools(client, database=tpch_db, cache=layered)[0]
    cold = report("layered cold", *run_arm(tool_e, n, counter))
    warm = report("layered warm", *run_arm(tool_e, n, counter))
    print(f"       {layered.counters()}")
    results["layered"] = (cold, warm)
    undo()

    header("SUMMARY", width=118)
    print(
        f"  {'arm':<18} {'cold':>10} {'speedup':>9} {'warm':>10} {'speedup':>9} "
        f"{'per warm call':>16}"
    )
    print("  " + "-" * 78)
    for name, (cold_t, warm_t) in results.items():
        sc = f"{base / cold_t:.2f}x" if cold_t else "-"
        sw = f"{base / warm_t:,.2f}x" if warm_t > 0.001 else ">10000x"
        print(f"  {name:<18} {cold_t:>9.2f}s {sc:>9} {warm_t:>9.2f}s {sw:>9} {fmt(warm_t / n):>16}")

    client.close()


if __name__ == "__main__":
    main()
