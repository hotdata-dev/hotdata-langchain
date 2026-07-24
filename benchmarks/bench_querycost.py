"""What does the work we are caching actually cost, and is the baseline honest?

A cache can only ever save the compute, never the network. So the decisive number is not
how fast a cache hit is -- it is how much of an uncached call was compute in the first
place. This establishes that, and rules out the two ways the baseline could be a lie:

  1. A server-side result cache would make the "uncached" arm secretly cached. Detected by
     varying a literal that changes the SQL text but filters no rows.
  2. A query that returns nothing looks fast. Detected by asserting Q1's group
     cardinalities against the known sf=1 answer.

    uv run python -m benchmarks.bench_querycost
"""

from __future__ import annotations

import argparse

import hotdata_framework as hf

from benchmarks.harness import HttpCounter, fmt, header, show, stats, timed
from benchmarks.tpch import (
    HEAVY_QUERIES,
    Q1,
    Q1_EXPECTED_COUNTS_SF1,
    Q5,
    TPCH_DB_NAME,
    resolve_database_id,
    vary,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tpch-db", default=TPCH_DB_NAME)
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()

    client = hf.from_env()
    db = resolve_database_id(client, args.tpch_db)

    header("WARM-UP -- absorbs KEDA scale-from-zero, auth, and TLS handshakes")
    with HttpCounter() as hc:
        secs, _ = timed(lambda: client.execute_sql("SELECT 1 AS warm", database=db))
    print(
        f"  first call in this process: {fmt(secs)}  ({hc.n} round trips, "
        f"{hc.new_connections} handshakes)"
    )
    print("  -> excluded from every number below; a fleet benchmark pays it per process")

    header("CORRECTNESS -- Q1 must return the known sf=1 answer")
    rows = client.execute_sql(Q1, database=db).to_records()
    counts = [r["count_order"] for r in rows]
    print(f"  count_order per group: {counts}")
    print(f"  expected (sf=1):       {Q1_EXPECTED_COUNTS_SF1}")
    ok = counts == Q1_EXPECTED_COUNTS_SF1
    print(f"  correct: {ok}")
    if not ok:
        print("  WARNING: not the canonical sf=1 result -- timings may not be comparable")

    header("IS THERE A SERVER-SIDE RESULT CACHE?")
    floor = [
        timed(lambda: client.execute_sql("SELECT 1 AS one", database=db))[0]
        for _ in range(args.samples)
    ]
    show("SELECT 1 (round trips + engine floor)", stats(floor))

    for name, sql in (("Q1", Q1), ("Q5", Q5)):
        same = [
            timed(lambda s=sql: client.execute_sql(s, database=db))[0]  # type: ignore[misc]
            for _ in range(args.samples)
        ]
        varied = [
            timed(lambda s=sql, i=i: client.execute_sql(vary(s, i), database=db))[0]  # type: ignore[misc]
            for i in range(args.samples)
        ]
        show(f"{name} identical SQL, repeated", stats(same))
        show(f"{name} literal varied each call", stats(varied))
        s, v, f = stats(same), stats(varied), stats(floor)
        verdict = "RESULT CACHE SUSPECTED" if v["p50"] > 1.5 * s["p50"] else "no result cache"
        print(f"    varied vs identical delta: {fmt(v['p50'] - s['p50'])}  -> {verdict}")
        print(
            f"    {name} p50 total {fmt(s['p50'])} = floor {fmt(f['p50'])} "
            f"+ compute {fmt(max(0.0, s['p50'] - f['p50']))}"
        )
        print(f"    network+floor is {100 * f['p50'] / s['p50']:.0f}% of this query\n")

    header("HOW HEAVY CAN THE WORK GET? -- compute vs one round trip")
    f = stats(floor)
    print(f"  floor (round trips + engine startup): {fmt(f['p50'])}")
    print(f"\n  {'query':<42} {'wall':>11} {'compute above floor':>21}")
    print("  " + "-" * 76)
    for name, sql in HEAVY_QUERIES.items():
        try:
            secs, _ = timed(lambda s=sql: client.execute_sql(s, database=db))  # type: ignore[misc]
            print(f"  {name:<42} {fmt(secs):>11} {fmt(max(0.0, secs - f['p50'])):>21}")
        except Exception as e:
            print(f"  {name:<42} {'FAILED':>11}   {str(e)[:36]}")
    print(
        "\n  A remote cache can only save the compute column. Where compute is smaller than\n"
        "  a round trip, caching the result remotely cannot pay for itself."
    )

    client.close()


if __name__ == "__main__":
    main()
