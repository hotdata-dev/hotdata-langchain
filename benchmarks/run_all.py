"""Run the whole suite in the order the argument builds, writing a combined transcript.

uv run python -m benchmarks.run_all
uv run python -m benchmarks.run_all --out results.txt --skip fleet
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Order matters: establish what the work costs before claiming a cache improved it.
STAGES = [
    ("querycost", "benchmarks.bench_querycost", "What does the work cost? Is the baseline honest?"),
    ("primitives", "benchmarks.bench_primitives", "Where does a cache hit's latency go?"),
    ("endtoend", "benchmarks.bench_endtoend", "Through the real StructuredTool, all arms"),
    ("fleet", "benchmarks.bench_fleet", "Short-lived workers: shared vs local"),
    ("crossover", "benchmarks.crossover", "Model over the measured constants"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None, help="also write the transcript here")
    ap.add_argument("--skip", nargs="*", default=[], help="stage names to skip")
    ap.add_argument("--only", nargs="*", default=None, help="run only these stages")
    args = ap.parse_args()

    chunks: list[str] = []
    for name, module, blurb in STAGES:
        if name in args.skip or (args.only and name not in args.only):
            print(f"--- skipping {name}")
            continue
        banner = f"\n{'#' * 100}\n# {name.upper()}  --  {blurb}\n{'#' * 100}"
        print(banner, flush=True)
        chunks.append(banner)
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", module], capture_output=True, text=True, check=False
        )
        took = time.perf_counter() - start
        print(proc.stdout, end="", flush=True)
        chunks.append(proc.stdout)
        if proc.returncode != 0:
            tail = proc.stderr.strip()[-2000:]
            print(f"\n!! {name} exited {proc.returncode}\n{tail}", flush=True)
            chunks.append(f"\n!! {name} exited {proc.returncode}\n{tail}")
        footer = f"\n[{name} took {took:.1f}s]"
        print(footer, flush=True)
        chunks.append(footer)

    if args.out:
        args.out.write_text("\n".join(chunks))
        print(f"\nTranscript written to {args.out}")


if __name__ == "__main__":
    main()
