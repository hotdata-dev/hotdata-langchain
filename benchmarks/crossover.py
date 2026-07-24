"""When does each cache backend win? A model over the measured constants.

Pure arithmetic, no network -- so it runs anywhere, and the constants it takes are exactly
the ones the other benchmarks measure. Feed it your own measurements to see how the answer
moves for your latency to the API and your workload.

    uv run python -m benchmarks.crossover
    uv run python -m benchmarks.crossover --remote-hit 0.33 --workers 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class Constants:
    """Measured per-operation costs, in seconds."""

    work: float  # cost of the real tool call being cached
    remote_hit: float  # a Hotdata cache hit
    remote_write: float  # a Hotdata cache write
    local_hit: float  # a SQLite cache hit
    local_write: float  # a SQLite cache write
    bootstrap: float  # per-process client setup a remote cache forces (auth, TLS, resolve)


def total_uncached(c: Constants, calls: int, workers: int) -> float:
    return calls * c.work


def total_local(c: Constants, calls: int, workers: int) -> float:
    """Per-process local cache: each worker misses once, then hits."""
    return workers * (c.work + c.local_write) + max(0, calls - workers) * c.local_hit


def total_remote(c: Constants, calls: int, workers: int) -> float:
    """Shared remote cache: one miss overall, then a network hop per hit."""
    return c.work + c.remote_write + (calls - 1) * c.remote_hit + workers * c.bootstrap


def total_layered(c: Constants, calls: int, workers: int) -> float:
    """Shared remote tier with a local tier in front: one remote hit per worker."""
    remote_hits = max(0, workers - 1)
    local_hits = max(0, calls - 1 - remote_hits)
    return (
        c.work
        + c.remote_write
        + c.local_write
        + remote_hits * c.remote_hit
        + local_hits * c.local_hit
        + workers * c.bootstrap
    )


STRATEGIES = {
    "no cache": total_uncached,
    "sqlite (per process)": total_local,
    "hotdata (shared)": total_remote,
    "layered": total_layered,
}

# Server-side costs measured by bench_serverside, in ms. Network-free, so these hold
# wherever the app is deployed; only the round trip changes.
SERVER_MS = {"cache_lookup": 75.0, "q1": 100.0, "q5": 267.0}

# Round trips the SDK spends per operation, measured by bench_primitives.
TRIPS_PER_QUERY = 2

# Realistic deployments for a LangChain app talking to a hosted Hotdata.
RTT_SCENARIOS = [
    (0.5, "same cluster / sidecar"),
    (2.0, "same AWS region"),
    (15.0, "same continent, different region"),
    (80.0, "cross-continent"),
    (265.0, "this benchmark machine -> us-west-2"),
]


def rtt_sweep() -> None:
    """How the answer moves with distance, holding the measured server costs fixed.

    The point of the sweep: a local cache's advantage is almost entirely the round trip it
    skips, so it shrinks as the app moves closer to the platform -- but the cache-lookup
    floor does not shrink, and that is what caps the co-located case.
    """
    print("\nSPEEDUP FROM A HOTDATA TOOL CACHE, BY DEPLOYMENT DISTANCE")
    print(
        f"  server-side: cache lookup {SERVER_MS['cache_lookup']:.0f}ms, "
        f"Q1 {SERVER_MS['q1']:.0f}ms, Q5 {SERVER_MS['q5']:.0f}ms "
        f"({TRIPS_PER_QUERY} round trips per op)"
    )
    print(
        f"\n  {'deployment':<38} {'RTT':>7} {'hit':>9} {'Q1':>9} {'Q1 gain':>9} "
        f"{'Q5':>9} {'Q5 gain':>9}"
    )
    print("  " + "-" * 96)
    for rtt, label in RTT_SCENARIOS:
        net = TRIPS_PER_QUERY * rtt
        hit = net + SERVER_MS["cache_lookup"]
        q1 = net + SERVER_MS["q1"]
        q5 = net + SERVER_MS["q5"]
        print(
            f"  {label:<38} {rtt:>6.1f}ms {hit:>7.0f}ms {q1:>7.0f}ms "
            f"{q1 / hit:>8.2f}x {q5:>7.0f}ms {q5 / hit:>8.2f}x"
        )
    print(
        "\n  A local cache serves any of these in ~0.1ms, so it always wins on latency alone.\n"
        "  What it cannot do is be shared: on a multi-replica or serverless backend its hit\n"
        "  rate collapses, which is the case the shared tier exists for."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--work",
        type=float,
        default=0.672,
        help="cost of the uncached tool call (default: measured TPCH Q1)",
    )
    ap.add_argument("--remote-hit", type=float, default=0.616)
    ap.add_argument("--remote-write", type=float, default=4.02)
    ap.add_argument("--local-hit", type=float, default=0.0000053)
    ap.add_argument("--local-write", type=float, default=0.0000153)
    ap.add_argument(
        "--bootstrap",
        type=float,
        default=2.0,
        help="per-process client setup a remote cache forces; 0 if long-lived",
    )
    ap.add_argument("--calls", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    c = Constants(
        work=args.work,
        remote_hit=args.remote_hit,
        remote_write=args.remote_write,
        local_hit=args.local_hit,
        local_write=args.local_write,
        bootstrap=args.bootstrap,
    )

    print(f"constants: work={c.work}s remote_hit={c.remote_hit}s remote_write={c.remote_write}s")
    print(f"           local_hit={c.local_hit * 1e6:.1f}us bootstrap={c.bootstrap}s\n")

    print(f"AT {args.calls} CALLS ACROSS {args.workers} WORKER PROCESS(ES)")
    print(f"  {'strategy':<24} {'total':>10} {'vs no cache':>13}")
    print("  " + "-" * 50)
    base = total_uncached(c, args.calls, args.workers)
    for name, fn in STRATEGIES.items():
        t = fn(c, args.calls, args.workers)
        print(f"  {name:<24} {t:>9.2f}s {base / t:>12.2f}x")

    print("\nBREAK-EVEN: how expensive must the work be before hotdata beats no cache?")
    print("  (long-lived process, bootstrap amortised to zero)")
    print(f"  {'calls':>7} {'work must exceed':>18}")
    print("  " + "-" * 28)
    for calls in (2, 3, 5, 10, 100, 1000):
        # work*calls > work + remote_write + (calls-1)*remote_hit
        threshold = (c.remote_write + (calls - 1) * c.remote_hit) / (calls - 1)
        print(f"  {calls:>7} {threshold:>17.2f}s")
    print(
        f"\n  Asymptote as calls -> infinity: {c.remote_hit:.3f}s -- a remote cache can never\n"
        f"  help work cheaper than one round trip, no matter how often it repeats."
    )

    rtt_sweep()

    print("\nSENSITIVITY TO WORKER COUNT (100 calls, work as given)")
    print(f"  {'workers':>8} " + " ".join(f"{k:>21}" for k in STRATEGIES))
    print("  " + "-" * 96)
    for w in (1, 2, 4, 10, 50, 100):
        row = " ".join(f"{fn(c, 100, w):>20.2f}s" for fn in STRATEGIES.values())
        print(f"  {w:>8} {row}")


if __name__ == "__main__":
    main()
