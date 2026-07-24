"""Strip the network out: compare server-side execution time only.

Comparing a local SQLite file against a hosted platform partly measures the distance to
us-west-2 rather than the two designs. The engine reports its own execution time per query,
so this uses that instead of wall clock. What remains is geography-independent:

  - what a cache-lookup SELECT costs the engine
  - what the query it replaces costs the engine
  - how both scale with the size of the cached result

If a cache lookup costs the engine about what the query costs, then caching cannot pay off
however close you deploy -- and that is a claim about architecture, not latency.

    uv run python -m benchmarks.bench_serverside
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

import hotdata_framework as hf

from benchmarks.backends import SqliteToolCache
from benchmarks.harness import fmt, header, timed
from benchmarks.tpch import CACHE_DB_NAME, Q1, Q5, TPCH_DB_NAME, resolve_database_id
from hotdata_langchain.cache import MISS, HotdataToolCache

TOOL = "hotdata_execute_sql"
VERSION = "bench-serverside"


def server_ms(client: hf.HotdataClient, sql: str, db: str, n: int = 5) -> tuple[float, float]:
    """Return (median server execution_time_ms, median wall ms) for ``sql``."""
    server, wall = [], []
    for _ in range(n):
        secs, result = timed(lambda: client.execute_sql(sql, database=db))
        wall.append(secs * 1000)
        if result.execution_time_ms is not None:
            server.append(float(result.execution_time_ms))
    return (statistics.median(server) if server else float("nan"), statistics.median(wall))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tpch-db", default=TPCH_DB_NAME)
    ap.add_argument("--cache-db", default=CACHE_DB_NAME)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument(
        "--payload-rows",
        type=int,
        nargs="+",
        default=[1, 100, 1000, 10000],
        help="cached result sizes to sweep, in rows",
    )
    args = ap.parse_args()

    client = hf.from_env()
    tpch_db = resolve_database_id(client, args.tpch_db)
    cache_db = resolve_database_id(client, args.cache_db)
    client.execute_sql("SELECT 1", database=tpch_db)  # warm

    header("SERVER-SIDE EXECUTION TIME -- network excluded entirely")
    print(f"  {'operation':<46} {'server':>10} {'wall':>11} {'network share':>14}")
    print("  " + "-" * 86)

    rows = []
    for label, sql, db in [
        ("SELECT 1 (engine floor)", "SELECT 1 AS one", tpch_db),
        ("TPCH Q1 -- 6M-row aggregation", Q1, tpch_db),
        ("TPCH Q5 -- 6-way join", Q5, tpch_db),
        (
            "cache lookup (SELECT by cache_key)",
            'SELECT result_json, created_at FROM "default"."public"."tool_cache" '
            "WHERE cache_key = '" + "0" * 64 + "'",
            cache_db,
        ),
    ]:
        s, w = server_ms(client, sql, db, args.samples)
        rows.append((label, s, w))
        print(f"  {label:<46} {s:>8.1f}ms {w:>9.1f}ms {100 * (1 - s / w):>13.0f}%")

    floor, q1, q5, lookup = (r[1] for r in rows)
    header("WHAT A CACHE HIT SAVES THE ENGINE, WITH ZERO NETWORK")
    print(f"  engine floor (SELECT 1)                  {floor:8.1f}ms")
    print(
        f"  a cache lookup costs the engine          {lookup:8.1f}ms"
        f"   ({lookup - floor:+.1f}ms above the floor)"
    )
    print(f"  Q1 costs the engine                      {q1:8.1f}ms")
    saving = q1 - lookup
    print(
        f"  saving per hit                           {saving:8.1f}ms   ({q1 / lookup:.2f}x)"
        if lookup
        else ""
    )
    print(f"\n  Q5 costs the engine                      {q5:8.1f}ms")
    print(
        f"  saving per hit                           {q5 - lookup:8.1f}ms   ({q5 / lookup:.2f}x)"
        if lookup
        else ""
    )
    print(
        "\n  This is the ceiling for a co-located deployment: even at zero network latency,\n"
        "  a Hotdata cache hit cannot beat the query by more than this ratio."
    )

    header("DOES THE CACHED RESULT SIZE CHANGE THE ANSWER?")
    hot = HotdataToolCache(client, database_id=cache_db, version=VERSION)
    tmp = Path(tempfile.mkdtemp())
    lite = SqliteToolCache(tmp / "size.db", version=VERSION)

    print(
        f"  {'payload':>10} {'bytes':>10} {'hotdata set':>13} {'hotdata get':>13} "
        f"{'sqlite set':>12} {'sqlite get':>12}"
    )
    print("  " + "-" * 78)
    for n_rows in args.payload_rows:
        payload = [
            {"l_orderkey": i, "l_quantity": i % 50, "l_comment": f"row {i} padding text"}
            for i in range(n_rows)
        ]
        n_bytes = len(json.dumps(payload))
        key = hot.make_key(TOOL, {"sql": f"SELECT payload {n_rows}"})

        set_secs, _ = timed(
            lambda k=key, p=payload: hot.set(k, tool_name=TOOL, args={}, result=p)  # type: ignore[misc]
        )
        get_secs, got = timed(lambda k=key: hot.get(k))  # type: ignore[misc]
        if got is MISS or len(got) != n_rows:
            print(f"  {n_rows:>10} -- readback mismatch, skipping")
            continue
        lset, _ = timed(
            lambda k=key, p=payload: lite.set(k, tool_name=TOOL, args={}, result=p)  # type: ignore[misc]
        )
        lget, _ = timed(lambda k=key: lite.get(k))  # type: ignore[misc]

        print(
            f"  {n_rows:>10} {n_bytes:>10,} {fmt(set_secs):>13} {fmt(get_secs):>13} "
            f"{fmt(lset):>12} {fmt(lget):>12}"
        )

    print(
        "\n  Note: make_hotdata_tools defaults to max_rows=100, so a cached tool result is\n"
        "  capped at 100 rows in the shipped configuration. The larger sizes show what a\n"
        "  higher cap would cost."
    )

    client.close()


if __name__ == "__main__":
    main()
