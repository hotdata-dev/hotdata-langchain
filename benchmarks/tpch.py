"""TPCH fixture: canonical queries, and provisioning them into a managed database.

Data is generated locally with DuckDB's built-in ``dbgen`` -- no download, deterministic,
and the same dataset ``hotdata-ibis``'s README references -- then loaded into a managed
database once and reused across runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

TPCH_DB_NAME = "tpch_sf1"
CACHE_DB_NAME = "langchain_tool_cache"

TABLES = ["region", "nation", "supplier", "customer", "orders", "lineitem"]

# Canonical TPCH Q1: pricing summary report. Single-table aggregation over every
# lineitem row. At sf=1 the answer is fixed, so it doubles as a correctness check.
Q1 = """
SELECT l_returnflag, l_linestatus,
       sum(l_quantity) AS sum_qty,
       sum(l_extendedprice) AS sum_base_price,
       sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
       sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
       avg(l_quantity) AS avg_qty,
       avg(l_extendedprice) AS avg_price,
       avg(l_discount) AS avg_disc,
       count(*) AS count_order
FROM "default"."public"."lineitem"
WHERE l_shipdate <= DATE '1998-09-02'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus
"""

# Canonical TPCH Q5: local supplier volume. Six-way join, filtered to Asia / 1994.
Q5 = """
SELECT n_name, sum(l_extendedprice * (1 - l_discount)) AS revenue
FROM "default"."public"."customer",
     "default"."public"."orders",
     "default"."public"."lineitem",
     "default"."public"."supplier",
     "default"."public"."nation",
     "default"."public"."region"
WHERE c_custkey = o_custkey
  AND l_orderkey = o_orderkey
  AND l_suppkey = s_suppkey
  AND c_nationkey = s_nationkey
  AND s_nationkey = n_nationkey
  AND n_regionkey = r_regionkey
  AND r_name = 'ASIA'
  AND o_orderdate >= DATE '1994-01-01'
  AND o_orderdate < DATE '1995-01-01'
GROUP BY n_name
ORDER BY revenue DESC
"""

# The known-correct Q1 group cardinalities at sf=1. Asserting on these keeps a
# "fast because it returned nothing" result from being mistaken for a fast query.
Q1_EXPECTED_COUNTS_SF1 = [1478493, 38854, 2920374, 1478870]

# Progressively heavier work, used to find where compute exceeds a network round trip.
HEAVY_QUERIES = {
    "self-join lineitem on orderkey": """
        SELECT count(*) AS n
        FROM "default"."public"."lineitem" a
        JOIN "default"."public"."lineitem" b ON a.l_orderkey = b.l_orderkey
    """,
    "count distinct l_comment (6M strings)": """
        SELECT count(DISTINCT l_comment) AS n FROM "default"."public"."lineitem"
    """,
    "global sort of 6M rows": """
        SELECT l_orderkey, l_extendedprice
        FROM "default"."public"."lineitem"
        ORDER BY l_extendedprice DESC, l_orderkey LIMIT 20
    """,
    "group by orderkey having sum > 300": """
        SELECT l_orderkey, sum(l_quantity) AS tot
        FROM "default"."public"."lineitem"
        GROUP BY l_orderkey HAVING sum(l_quantity) > 300
        ORDER BY tot DESC LIMIT 20
    """,
    "LIKE scan over 6M comments": """
        SELECT count(*) AS n FROM "default"."public"."lineitem"
        WHERE l_comment LIKE '%special%requests%'
    """,
    "4-way join + aggregate": """
        SELECT a.l_returnflag, count(*) AS n
        FROM "default"."public"."lineitem" a
        JOIN "default"."public"."orders" o ON a.l_orderkey = o.o_orderkey
        JOIN "default"."public"."customer" c ON o.o_custkey = c.c_custkey
        JOIN "default"."public"."supplier" s ON a.l_suppkey = s.s_suppkey
        GROUP BY a.l_returnflag
    """,
}


def vary(sql: str, i: int) -> str:
    """Return ``sql`` with a predicate that excludes no rows but changes the SQL text.

    ``l_quantity`` is always at least 1, so ``l_quantity > -1-i`` filters nothing: same
    rows, same work, different text. Used to detect a server-side result cache -- if
    repeating identical SQL is fast but varying the text is not, results are being cached
    and the "uncached" baseline is not really uncached.
    """
    if "WHERE l_shipdate <= DATE '1998-09-02'" in sql:
        return sql.replace(
            "WHERE l_shipdate <= DATE '1998-09-02'",
            f"WHERE l_shipdate <= DATE '1998-09-02' AND l_quantity > {-1 - i}",
        )
    return sql.replace("r_name = 'ASIA'", f"r_name = 'ASIA' AND l_quantity > {-1 - i}")


def resolve_database_id(client: Any, name_or_id: str) -> str:
    """Resolve a managed database by name or ID, raising a useful error if absent."""
    try:
        return str(client.resolve_managed_database(name_or_id).id)
    except KeyError as e:
        raise SystemExit(
            f"No managed database named or with id {name_or_id!r} in this workspace.\n"
            f"Run:  uv run --with duckdb python -m benchmarks.provision_tpch"
        ) from e


def tpch_is_loaded(client: Any, database_id: str) -> bool:
    """True when every TPCH table exists in ``database_id``."""
    present = {t.table for t in client.list_managed_tables(database_id)}
    return all(t in present for t in TABLES)


def generate_tpch(out_dir: Path, scale: float = 1.0) -> dict[str, Path]:
    """Generate TPCH parquet files locally with DuckDB. Returns table -> path."""
    try:
        import duckdb
    except ImportError as e:
        raise SystemExit(
            "duckdb is needed to generate TPCH data. Run this module as:\n"
            "  uv run --with duckdb python -m benchmarks.provision_tpch"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={scale})")
    paths = {}
    for table in TABLES:
        path = out_dir / f"{table}.parquet"
        con.execute(f"COPY {table} TO '{path}' (FORMAT PARQUET)")
        paths[table] = path
    con.close()
    return paths
