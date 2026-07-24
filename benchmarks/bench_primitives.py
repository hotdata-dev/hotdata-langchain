"""Where does a cache hit's latency actually go?

For the same logical operation -- look up one key, get one row back -- this measures the
wall time, how many HTTP round trips it took, and how much of the wall time was spent
inside those round trips. A local SQLite file pays none of the network cost, so the gap
between the two backends is the network's contribution, measured rather than assumed.

    uv run python -m benchmarks.bench_primitives
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import hotdata_framework as hf

from benchmarks.backends import SqliteToolCache
from benchmarks.harness import (
    HttpCounter,
    fmt,
    header,
    memoize_resolve,
    report_retries,
    show,
    stats,
    timed,
    timed_retry,
)
from benchmarks.tpch import CACHE_DB_NAME, resolve_database_id
from hotdata_langchain.cache import MISS, HotdataToolCache

TOOL = "hotdata_execute_sql"
ARGS = {"sql": "SELECT 42 AS answer"}
PAYLOAD = {"columns": ["answer"], "rows": [{"answer": 42}]}
VERSION = "bench-primitives"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-db", default=CACHE_DB_NAME)
    ap.add_argument("--remote-samples", type=int, default=12)
    ap.add_argument("--local-samples", type=int, default=5000)
    args = ap.parse_args()

    client = hf.from_env()
    cache_db = resolve_database_id(client, args.cache_db)
    hot = HotdataToolCache(client, database_id=cache_db, version=VERSION)
    tmp = Path(tempfile.mkdtemp())
    lite = SqliteToolCache(tmp / "cache.db", version=VERSION)
    lite_durable = SqliteToolCache(tmp / "durable.db", version=VERSION, synchronous="FULL")

    header("KEY PARITY -- both backends must address the same entry")
    hot_key = hot.make_key(TOOL, ARGS)
    lite_key = lite.make_key(TOOL, ARGS)
    print(f"  hotdata: {hot_key}")
    print(f"  sqlite : {lite_key}")
    print(f"  identical: {hot_key == lite_key}")
    if hot_key != lite_key:
        raise SystemExit("key schemes diverged; the comparison would not be apples-to-apples")

    header("COLD START -- what the first call in a fresh process costs")
    with HttpCounter() as hc:
        secs, _ = timed_retry(lambda: hot.set(hot_key, tool_name=TOOL, args=ARGS, result=PAYLOAD))
    print(
        f"  first set(), including _ensure_ready(): {fmt(secs)}   "
        f"http={hc.n}  in-socket={fmt(hc.socket_secs)}  handshakes={hc.new_connections}"
    )
    hc.dump()

    header("ANATOMY OF ONE CACHE HIT")
    hot.get(hot_key)  # warm the pool so the handshake is not counted below
    with HttpCounter() as hc:
        secs, val = timed(lambda: hot.get(hot_key))
    if val is MISS:
        raise SystemExit("expected a hit; the write above did not land")
    print(f"  wall {fmt(secs)}   http round trips={hc.n}")
    hc.dump()

    header("GET -- HIT")
    remote_hit, remote_socket, remote_http = [], [], []
    for _ in range(args.remote_samples):
        with HttpCounter() as hc:
            secs, val = timed(lambda: hot.get(hot_key))
        if val is MISS:
            raise SystemExit("unexpected miss on a warm key")
        remote_hit.append(secs)
        remote_socket.append(hc.socket_secs)
        remote_http.append(hc.n)
    show("hotdata get() wall", stats(remote_hit))
    show("hotdata get() time inside sockets", stats(remote_socket))
    print(f"  {'hotdata get() round trips per op':<36} {remote_http}")

    lite.set(lite_key, tool_name=TOOL, args=ARGS, result=PAYLOAD)
    local_hit = []
    for _ in range(args.local_samples):
        secs, val = timed(lambda: lite.get(lite_key))
        if val is MISS:
            raise SystemExit("unexpected miss on a warm key")
        local_hit.append(secs)
    show("sqlite get() wall", stats(local_hit))

    header("GET -- MISS")
    miss_key = hot.make_key(TOOL, {"sql": "SELECT 'never cached'"})
    remote_miss = []
    for _ in range(max(4, args.remote_samples // 2)):
        secs, val = timed(lambda: hot.get(miss_key))
        if val is not MISS:
            raise SystemExit("expected a miss")
        remote_miss.append(secs)
    show("hotdata get() miss", stats(remote_miss))
    show(
        "sqlite get() miss",
        stats([timed(lambda: lite.get(miss_key))[0] for _ in range(args.local_samples)]),
    )

    header("SET -- WRITE PATH")
    remote_set, remote_set_http = [], []
    for i in range(5):
        k = hot.make_key(TOOL, {"sql": f"SELECT {i} AS w"})
        with HttpCounter() as hc:
            secs, _ = timed_retry(
                lambda k=k: hot.set(k, tool_name=TOOL, args=ARGS, result=PAYLOAD)  # type: ignore[misc]
            )
        remote_set.append(secs)
        remote_set_http.append(hc.n)
    show("hotdata set() wall", stats(remote_set))
    print(f"  {'hotdata set() round trips per op':<36} {remote_set_http}")

    show(
        "sqlite set() (synchronous=NORMAL)",
        stats(
            [
                timed(
                    lambda i=i: lite.set(  # type: ignore[misc]
                        f"{i:064x}", tool_name=TOOL, args=ARGS, result=PAYLOAD
                    )
                )[0]
                for i in range(min(2000, args.local_samples))
            ]
        ),
    )
    show(
        "sqlite set() (synchronous=FULL)",
        stats(
            [
                timed(
                    lambda i=i: lite_durable.set(  # type: ignore[misc]
                        f"{i:064x}", tool_name=TOOL, args=ARGS, result=PAYLOAD
                    )
                )[0]
                for i in range(500)
            ]
        ),
    )

    header("THE AVOIDABLE ROUND TRIP -- memoizing resolve_managed_database")
    undo = memoize_resolve(client)
    client.resolve_managed_database(cache_db)  # pay it once
    with HttpCounter() as hc:
        secs, _ = timed(lambda: hot.get(hot_key))
    print(f"  wall {fmt(secs)}   http round trips={hc.n}")
    hc.dump()
    memo_hit = [timed(lambda: hot.get(hot_key))[0] for _ in range(args.remote_samples)]
    show("hotdata get() with resolve memoized", stats(memo_hit))
    undo()

    header("ATTRIBUTION")
    hit, sock, memo = stats(remote_hit), stats(remote_socket), stats(memo_hit)
    local = stats(local_hit)
    print(f"  hotdata cache hit, p50                    {fmt(hit['p50'])}")
    print(
        f"    spent inside HTTP sockets               {fmt(sock['p50'])}"
        f"   ({100 * sock['p50'] / hit['p50']:.1f}%)"
    )
    print(
        f"    spent in client-side work               {fmt(hit['p50'] - sock['p50'])}"
        f"   ({100 * (hit['p50'] - sock['p50']) / hit['p50']:.1f}%)"
    )
    print(f"  sqlite cache hit, p50                     {fmt(local['p50'])}")
    print(f"  hotdata / sqlite on a hit                 {hit['p50'] / local['p50']:,.0f}x")
    print()
    print(f"  hotdata hit as shipped (2 round trips)    {fmt(hit['p50'])}")
    print(f"  hotdata hit, resolve memoized (1 trip)    {fmt(memo['p50'])}")
    print(
        f"  available saving                          {fmt(hit['p50'] - memo['p50'])}"
        f"   ({100 * (1 - memo['p50'] / hit['p50']):.0f}% faster)"
    )

    report_retries()
    client.close()


if __name__ == "__main__":
    main()
