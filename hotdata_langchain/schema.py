"""Schema introspection so an agent can learn what it is allowed to query."""

from __future__ import annotations

import json
import re

from hotdata_framework import HotdataClient
from langchain_core.tools import StructuredTool

DEFAULT_DESCRIBE_TOOL_NAME = "hotdata_describe_tables"

#: Cap on columns returned for a single table, so one wide table cannot flood the context.
DEFAULT_MAX_COLUMNS = 200

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _split_table(table: str) -> tuple[str | None, str]:
    """Split ``schema.table`` or a bare ``table`` into its parts, validating both."""
    parts = table.split(".")
    if len(parts) > 2:
        raise ValueError(
            "table must be 'table' or 'schema.table' (the database is already scoped), "
            f"got {table!r}"
        )
    for part in parts:
        if not _IDENTIFIER_RE.fullmatch(part):
            raise ValueError(f"table must be made of bare SQL identifiers, got {table!r}")
    return (parts[0], parts[1]) if len(parts) == 2 else (None, parts[0])


def table_overview_sql() -> str:
    """Return SQL listing every table in the scoped database with its column count."""
    return (
        "SELECT table_schema, table_name, COUNT(column_name) AS column_count "
        "FROM information_schema.columns "
        "GROUP BY table_schema, table_name "
        "ORDER BY table_schema, table_name"
    )


def table_columns_sql(table: str, *, limit: int = DEFAULT_MAX_COLUMNS) -> str:
    """Return SQL listing one table's columns and types, in declaration order."""
    schema, name = _split_table(table)
    where = f"WHERE table_name = '{name}'"
    if schema is not None:
        where += f" AND table_schema = '{schema}'"
    return (
        f"SELECT table_schema, table_name, column_name, data_type "
        f"FROM information_schema.columns {where} "
        f"ORDER BY table_schema, table_name, ordinal_position "
        f"LIMIT {limit}"
    )


def describe_tables_json(
    client: HotdataClient,
    *,
    table: str | None = None,
    database: str | None = None,
    max_columns: int = DEFAULT_MAX_COLUMNS,
) -> str:
    """Describe the scoped database's tables, or one table's columns, as JSON.

    Without ``table`` this returns every table with its column count — a cheap map of
    what exists. With ``table`` it returns that table's columns and types in
    declaration order, capped at ``max_columns`` so a wide table cannot flood the
    model's context; the payload says so when the cap truncated the list.
    """
    if table is None:
        result = client.execute_sql(table_overview_sql(), database=database)
        tables = [
            {
                "table": f"{row['table_schema']}.{row['table_name']}",
                "column_count": row["column_count"],
            }
            for row in result.to_records()
        ]
        return json.dumps({"tables": tables}, indent=2)

    result = client.execute_sql(table_columns_sql(table, limit=max_columns), database=database)
    records = result.to_records()
    if not records:
        return json.dumps(
            {"table": table, "columns": [], "error": f"no table named {table!r} in this database"},
            indent=2,
        )
    payload: dict[str, object] = {
        "table": f"{records[0]['table_schema']}.{records[0]['table_name']}",
        "columns": [{"name": r["column_name"], "type": r["data_type"]} for r in records],
    }
    if len(records) == max_columns:
        payload["truncated_at"] = max_columns
    return json.dumps(payload, indent=2)


def default_describe_description() -> str:
    """Return the agent-facing description for the schema tool."""
    return (
        "Discover what data is available before writing a query. Called with no "
        "arguments it lists every table with how many columns it has; called with a "
        "table name ('listings' or 'public.listings') it returns that table's columns "
        "and their types.\n"
        "Use it whenever you are unsure a table or column exists — guessing a column "
        "name that is not there makes the query fail."
    )


def make_hotdata_describe_tables_tool(
    client: HotdataClient,
    *,
    database: str | None = None,
    name: str = DEFAULT_DESCRIBE_TOOL_NAME,
    description: str | None = None,
    max_columns: int = DEFAULT_MAX_COLUMNS,
) -> StructuredTool:
    """Return a LangChain tool that reports the scoped database's tables and columns."""

    def hotdata_describe_tables(table: str | None = None) -> str:
        """List the tables in the database, or one table's columns and types."""
        return describe_tables_json(client, table=table, database=database, max_columns=max_columns)

    return StructuredTool.from_function(
        func=hotdata_describe_tables,
        name=name,
        description=description or default_describe_description(),
    )
