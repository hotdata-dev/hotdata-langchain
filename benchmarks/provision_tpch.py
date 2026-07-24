"""Create the TPCH fixture this benchmark suite measures against.

Idempotent: skips generation and loading entirely if the tables are already there.

    uv run --with duckdb python -m benchmarks.provision_tpch
    uv run --with duckdb python -m benchmarks.provision_tpch --scale 1 --force
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import hotdata_framework as hf

from benchmarks.harness import fmt, timed
from benchmarks.tpch import (
    CACHE_DB_NAME,
    TABLES,
    TPCH_DB_NAME,
    generate_tpch,
    tpch_is_loaded,
)


def ensure_database(client: hf.HotdataClient, name: str, tables: list[str]) -> str:
    """Resolve a managed database by name, creating it if it does not exist."""
    try:
        db = client.resolve_managed_database(name)
        print(f"  found existing database {name!r} -> {db.id}")
        return str(db.id)
    except KeyError:
        db = client.create_managed_database(description=name, schema="public", tables=tables)
        print(f"  created database {name!r} -> {db.id}")
        return str(db.id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=float, default=1.0, help="TPCH scale factor")
    ap.add_argument("--force", action="store_true", help="reload even if tables exist")
    args = ap.parse_args()

    client = hf.from_env()
    print(f"workspace: {client.workspace_id}")

    print("\nCache database (holds tool-cache entries):")
    cache_db = ensure_database(client, CACHE_DB_NAME, ["tool_cache"])

    print("\nTPCH database:")
    tpch_db = ensure_database(client, TPCH_DB_NAME, TABLES)

    if not args.force and tpch_is_loaded(client, tpch_db):
        print(f"\n  all {len(TABLES)} TPCH tables already present -- nothing to load.")
        print(f"\nReady.\n  TPCH_DB_ID={tpch_db}\n  CACHE_DB_ID={cache_db}")
        client.close()
        return

    with tempfile.TemporaryDirectory() as tmp:
        print(f"\nGenerating TPCH sf={args.scale} locally with DuckDB...")
        secs, paths = timed(lambda: generate_tpch(Path(tmp), args.scale))
        print(f"  generated {len(paths)} parquet files in {fmt(secs)}")

        print("\nLoading into the managed database (largest last):")
        for table in TABLES:
            secs, _ = timed(
                lambda t=table: client.load_managed_table(  # type: ignore[misc]
                    tpch_db, t, schema="public", file=str(paths[t]), mode="replace"
                )
            )
            size_mb = paths[table].stat().st_size / 1e6
            print(f"  {table:<12} {size_mb:7.1f} MB  {fmt(secs)}")

    print(f"\nReady.\n  TPCH_DB_ID={tpch_db}\n  CACHE_DB_ID={cache_db}")
    client.close()


if __name__ == "__main__":
    main()
