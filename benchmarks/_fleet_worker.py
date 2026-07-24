"""One worker process for bench_fleet: make the cached call once, report timing as JSON.

Each invocation is a fresh interpreter on purpose. That is the whole point of the fleet
scenario -- a local cache starts empty in every worker, a shared cache does not -- and it
also makes each worker pay the client bootstrap (auth, TLS handshakes, database resolve)
that a long-lived process pays only once.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import hotdata_framework as hf

from benchmarks.backends import LayeredToolCache, SqliteToolCache
from benchmarks.harness import HttpCounter
from hotdata_langchain.cache import HotdataToolCache, cached

CALLS = {"n": 0}


def build_tool(work_secs: float):  # type: ignore[no-untyped-def]
    """A stand-in for slow external work: a third-party API, a scrape, an LLM call.

    Sleep rather than SQL, for two reasons. The work cost has to be an independent
    variable to find the break-even point, and no TPCH sf=1 query on this engine is slow
    enough to sit above a network round trip anyway.
    """

    def slow_external_tool(resource: str) -> str:
        CALLS["n"] += 1
        time.sleep(work_secs)
        return json.dumps({"resource": resource, "payload": "x" * 200})

    return slow_external_tool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["none", "sqlite", "hotdata", "layered"])
    ap.add_argument("--version", required=True)
    ap.add_argument("--work-secs", type=float, required=True)
    ap.add_argument("--sqlite-path", required=True)
    ap.add_argument("--cache-db-id", required=True)
    args = ap.parse_args()

    process_start = time.perf_counter()
    client = None
    cache: object | None = None

    if args.mode == "sqlite":
        cache = SqliteToolCache(args.sqlite_path, version=args.version)
    elif args.mode == "hotdata":
        client = hf.from_env()
        cache = HotdataToolCache(client, database_id=args.cache_db_id, version=args.version)
    elif args.mode == "layered":
        client = hf.from_env()
        cache = LayeredToolCache(
            SqliteToolCache(args.sqlite_path, version=args.version),
            HotdataToolCache(client, database_id=args.cache_db_id, version=args.version),
        )
    client_init = time.perf_counter() - process_start

    fn = build_tool(args.work_secs)
    if cache is not None:
        fn = cached(fn, cache=cache, tool_name="slow_external_tool")  # type: ignore[arg-type]

    # Constructing the client does no I/O when HOTDATA_WORKSPACE is set, so the per-process
    # cost that matters -- auth token exchange and TCP+TLS handshakes -- lands inside the
    # first cache operation. Count it there rather than around the constructor.
    with HttpCounter() as hc:
        start = time.perf_counter()
        fn("report-42")
        elapsed = time.perf_counter() - start

    if client is not None:
        client.close()

    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "mode": args.mode,
                "client_init": client_init,
                "elapsed": elapsed,
                "http_calls": hc.n,
                "handshakes": hc.new_connections,
                "socket_secs": hc.socket_secs,
                "did_real_work": CALLS["n"] == 1,
            }
        )
    )


if __name__ == "__main__":
    main()
