"""Schema introspection so an agent can learn what it is allowed to query."""

from __future__ import annotations

import json

from hotdata_framework import HotdataClient, ManagedDatabase
from langchain_core.tools import StructuredTool

from hotdata_langchain._sql import validate_identifier
from hotdata_langchain.databases import query_scope, resolve_database_by_id

DEFAULT_DESCRIBE_TOOL_NAME = "hotdata_describe_tables"

#: Cap on columns returned for a single table, so one wide table cannot flood the context.
DEFAULT_MAX_COLUMNS = 200


def _split_table(table: str) -> tuple[str | None, str]:
    """Split ``schema.table`` or a bare ``table`` into its parts, validating both."""
    parts = table.split(".")
    if len(parts) > 2:
        raise ValueError(
            "table must be 'table' or 'schema.table' (the database is already scoped), "
            f"got {table!r}"
        )
    for part in parts:
        validate_identifier(part, label="table")
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
    database: ManagedDatabase | None = None,
    max_columns: int = DEFAULT_MAX_COLUMNS,
) -> str:
    """Describe the scoped database's tables, or one table's columns, as JSON.

    Without ``table`` this returns every table with its column count — a cheap map of
    what exists. With ``table`` it returns that table's columns and types in
    declaration order, capped at ``max_columns`` so a wide table cannot flood the
    model's context; the payload says so when the cap truncated the list.

    ``database`` is a resolved ``ManagedDatabase``, not an id or a name — resolve one
    with :func:`hotdata_langchain.databases.resolve_database_by_id`.

    Raises ``ValueError`` for a non-positive ``max_columns``.
    """
    if max_columns < 1:
        raise ValueError(f"max_columns must be >= 1, got {max_columns}")
    scope = query_scope(database)
    if table is None:
        result = client.execute_sql(table_overview_sql(), database=scope)
        tables = [
            {
                "table": f"{row['table_schema']}.{row['table_name']}",
                "column_count": row["column_count"],
            }
            for row in result.to_records()
        ]
        return json.dumps({"tables": tables}, indent=2)

    # One row past the cap, so a table with exactly `max_columns` columns is reported as
    # complete rather than flagged as truncated — telling the model part of the schema is
    # missing is the one thing likely to send it back to guessing.
    result = client.execute_sql(table_columns_sql(table, limit=max_columns + 1), database=scope)
    records = result.to_records()
    if not records:
        return json.dumps(
            {"table": table, "columns": [], "error": f"no table named {table!r} in this database"},
            indent=2,
        )
    truncated = len(records) > max_columns
    records = records[:max_columns]
    payload: dict[str, object] = {
        "table": f"{records[0]['table_schema']}.{records[0]['table_name']}",
        "columns": [{"name": r["column_name"], "type": r["data_type"]} for r in records],
    }
    if truncated:
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
    database_id: str | ManagedDatabase | None = None,
    name: str = DEFAULT_DESCRIBE_TOOL_NAME,
    description: str | None = None,
    max_columns: int = DEFAULT_MAX_COLUMNS,
) -> StructuredTool:
    """Return a LangChain tool that reports the scoped database's tables and columns.

    ``database_id`` scopes the introspection to one managed database, by id and never by
    name; it is resolved once here. Pass an already-resolved ``ManagedDatabase`` to skip
    the lookup.

    Fails fast on a non-positive ``max_columns`` rather than at first invocation.
    """
    if max_columns < 1:
        raise ValueError(f"max_columns must be >= 1, got {max_columns}")
    database = resolve_database_by_id(client, database_id) if database_id is not None else None

    def hotdata_describe_tables(table: str | None = None) -> str:
        """List the tables in the database, or one table's columns and types."""
        return describe_tables_json(client, table=table, database=database, max_columns=max_columns)

    return StructuredTool.from_function(
        func=hotdata_describe_tables,
        name=name,
        description=description or default_describe_description(),
    )
