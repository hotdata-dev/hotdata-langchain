"""The scenario a local cache structurally cannot serve: a fleet of short-lived workers.

N fresh processes each make the same tool call, sequentially, so a shared cache written by
worker 1 is visible to worker 2. Swept across work costs, because the winner inverts:

  per-process sqlite  every worker starts empty, so every worker pays the full work cost
  shared sqlite       one file, one host: 1 miss then hits at local latency
  hotdata             shared across hosts: 1 miss then hits, but a network hop each
  layered             shared across hosts, and local latency on repeats within a process

Each worker also reports its bootstrap time, which is what a long-lived process amortises
away and a short-lived one does not.

    uv run python -m benchmarks.bench_fleet
    uv run python -m benchmarks.bench_fleet --workers 6 --work-secs 0.12 3.0 10.0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import hotdata_framework as hf

from benchmarks.tpch import CACHE_DB_NAME, resolve_database_id

ARMS = [
    ("no cache", "none", False),
    ("sqlite, per-process file", "sqlite", False),
    ("sqlite, shared file (one host)", "sqlite", True),
    ("hotdata (shared, cross-host)", "hotdata", False),
    ("layered sqlite -> hotdata", "layered", False),
]


def run_fleet(
    mode: str, work_secs: float, shared_file: bool, workers: int, cache_db: str, tag: str
) -> tuple[list[float], int, list[int], list[int]]:
    """Run ``workers`` fresh processes.

    Returns (per-worker elapsed, how many ran the real work, http calls, handshakes).
    """
    tmp = Path(tempfile.mkdtemp())
    version = f"fleet-{tag}-{int(time.time() * 1000)}"
    elapsed: list[float] = []
    http: list[int] = []
    handshakes: list[int] = []
    real_work = 0
    for w in range(workers):
        path = tmp / ("shared.db" if shared_file else f"worker{w}.db")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks._fleet_worker",
                mode,
                "--version",
                version,
                "--work-secs",
                str(work_secs),
                "--sqlite-path",
                str(path),
                "--cache-db-id",
                cache_db,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        if not lines:
            print(f"      worker {w} failed: {proc.stderr.strip()[-300:]}")
            elapsed.append(float("nan"))
            continue
        rec = json.loads(lines[-1])
        elapsed.append(rec["elapsed"])
        http.append(rec["http_calls"])
        handshakes.append(rec["handshakes"])
        real_work += int(rec["did_real_work"])
    return elapsed, real_work, http, handshakes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-db", default=CACHE_DB_NAME)
    ap.add_argument(
        "--work-secs",
        type=float,
        nargs="+",
        default=[0.12, 3.0],
        help="work costs to sweep; 0.12 matches a TPCH sf=1 query's compute",
    )
    args = ap.parse_args()

    client = hf.from_env()
    cache_db = resolve_database_id(client, args.cache_db)
    client.close()

    for work_secs in args.work_secs:
        print("\n" + "=" * 112)
        print(f"FLEET OF {args.workers} FRESH PROCESSES, same call, work cost = {work_secs}s")
        print("=" * 112)
        print(
            f"  {'arm':<32} {'per-worker elapsed':<40} {'total':>9} {'ran work':>9} "
            f"{'http/worker':>12} {'handshakes':>11}"
        )
        print("  " + "-" * 120)
        for name, mode, shared in ARMS:
            el, real, http, hs = run_fleet(
                mode, work_secs, shared, args.workers, cache_db, f"{mode}-{shared}-{work_secs}"
            )
            per = "  ".join(f"{t:6.3f}s" for t in el)
            http_desc = "-" if not http else f"{min(http)}-{max(http)}"
            hs_desc = "-" if not hs else f"{min(hs)}-{max(hs)}"
            print(
                f"  {name:<32} {per:<40} {sum(el):8.2f}s "
                f"{real:>4}/{args.workers:<4} {http_desc:>12} {hs_desc:>11}"
            )
        print(
            f"\n  'ran work' = how many workers actually executed the {work_secs}s call.\n"
            "  A shared cache should show 1; a per-process cache shows every worker.\n"
            "  'handshakes' = TCP+TLS setups this worker paid because its process was new.\n"
            "  Those, plus the auth token exchange, are why a fresh worker's cache hit costs\n"
            "  several times what a warm process's does."
        )


if __name__ == "__main__":
    main()
