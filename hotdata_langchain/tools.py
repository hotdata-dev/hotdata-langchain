"""LangChain tools built on hotdata-framework."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from hotdata_framework import DEFAULT_SCHEMA, HotdataClient, QueryResult
from langchain_core.tools import StructuredTool

from hotdata_langchain.databases import (
    create_managed_database,
    list_managed_databases_json,
    load_managed_table,
    load_result_summary,
    managed_database_summary,
)
from hotdata_langchain.schema import (
    DEFAULT_DESCRIBE_TOOL_NAME,
    make_hotdata_describe_tables_tool,
)
from hotdata_langchain.search import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEARCH_TOOL_NAME,
    make_hotdata_search_tool,
)


def sql_tool_description(
    search_tool_name: str | None = None,
    describe_tool_name: str | None = DEFAULT_DESCRIBE_TOOL_NAME,
) -> str:
    """Return the agent-facing description for the SQL tool.

    States the engine's capabilities positively rather than listing what is absent, so
    the description does not turn into a false claim as the SQL surface grows. The two
    constraints it does name are the ones that silently produce wrong tool calls: SQL
    cannot rank text, and an aggregate query that references no column is rejected.

    ``search_tool_name`` is named as the place to do text matching only when a search
    tool is actually registered alongside this one. When it is, `LIKE` is framed as a
    filter on text you already know rather than a way to find relevant rows: stating
    only that it "works" was observed to pull the model into `ILIKE '%word%'` instead of
    searching, which returns unranked results and misses related wording.
    """
    text_guidance = (
        f"To find which rows are about something, use the {search_tool_name} tool — it "
        f"ranks by relevance — and then pass the values it returns into SQL as literals. "
        f"SQL cannot rank text. LIKE and ILIKE only test for a literal substring you "
        f"already know, so they are a filter, not a substitute for searching: "
        f"ILIKE '%word%' returns unranked rows and misses the related wording a search "
        f"would find."
        if search_tool_name
        else "LIKE and ILIKE test for a literal substring, but SQL cannot rank rows by "
        "how well their text matches a phrase."
    )
    discovery = (
        f"Do not guess table or column names — get them from the {describe_tool_name} tool"
        if describe_tool_name
        else "Do not guess table or column names — read them from "
        "information_schema.tables and information_schema.columns, or DESCRIBE <table>"
    )
    return (
        "Run a read-only SQL query and return the rows as JSON. PostgreSQL dialect: "
        "joins, CTEs, subqueries, GROUP BY, window functions, ORDER BY/LIMIT and the "
        "usual scalar functions all work.\n"
        f"{text_guidance}\n"
        "An aggregate query must reference at least one column: COUNT(*) and COUNT(1) "
        "are rejected on their own, so write COUNT(<column>) or add a GROUP BY.\n"
        "Tables are addressed as catalog.schema.table; inside a managed database the "
        "catalog is always 'default' (schema-qualified names also resolve). "
        f"{discovery}."
    )


def result_rows_for_llm(result: QueryResult, *, max_rows: int = 20) -> list[dict[str, Any]]:
    return result.to_records(max_rows=max_rows)


def execute_sql_json(
    client: HotdataClient,
    sql: str,
    *,
    max_rows: int = 100,
    database: str | None = None,
) -> str:
    result = client.execute_sql(sql, database=database)
    payload = {
        "metadata": result.metadata_dict(),
        "rows": result.to_records(max_rows=max_rows),
    }
    return json.dumps(payload, indent=2)


def make_hotdata_tools(
    client: HotdataClient,
    *,
    max_rows: int = 100,
    database: str | None = None,
    search_table: str | None = None,
    search_column: str | None = None,
    search_columns: Sequence[str] | None = None,
    search_k: int = DEFAULT_SEARCH_LIMIT,
    search_tool_name: str = DEFAULT_SEARCH_TOOL_NAME,
    describe_tables: bool = True,
) -> list[StructuredTool]:
    """Return LangChain tools for SQL and managed database workflows.

    ``describe_tables`` (on by default) adds a schema-introspection tool, so the agent
    can look up tables and columns instead of guessing them. It reads
    ``information_schema`` in whichever database the tools are scoped to.

    Passing both ``search_table`` and ``search_column`` appends a full-text search tool
    bound to that column, which requires a BM25 index on it. ``search_columns`` selects
    the columns each hit returns (default: the searched column). Supplying only one of
    ``search_table``/``search_column`` raises ``ValueError``.

    For more than one searchable corpus, call
    :func:`hotdata_langchain.search.make_hotdata_search_tool` directly per corpus and
    extend this list.
    """
    if (search_table is None) != (search_column is None):
        raise ValueError("search_table and search_column must be provided together")

    def hotdata_execute_sql(sql: str) -> str:
        """Run SQL against the Hotdata workspace and return JSON rows."""
        return execute_sql_json(client, sql, max_rows=max_rows, database=database)

    def hotdata_list_managed_databases() -> str:
        """List Hotdata-managed databases in the workspace."""
        return list_managed_databases_json(client)

    def hotdata_create_managed_database(
        name: str,
        schema_name: str = DEFAULT_SCHEMA,
        tables: str = "",
    ) -> str:
        """Create a managed database and optionally declare tables (comma/newline separated)."""
        table_names = [t.strip() for t in tables.replace(",", "\n").splitlines() if t.strip()]
        db = create_managed_database(
            client,
            name=name,
            schema=schema_name or DEFAULT_SCHEMA,
            tables=table_names or None,
        )
        return json.dumps(managed_database_summary(db), indent=2)

    def hotdata_load_managed_table(
        database: str,
        table: str,
        file: str,
        schema_name: str = DEFAULT_SCHEMA,
    ) -> str:
        """Load a local parquet file into a declared managed table."""
        loaded = load_managed_table(
            client,
            database=database,
            table=table,
            file=file,
            schema=schema_name or DEFAULT_SCHEMA,
        )
        return json.dumps(load_result_summary(loaded), indent=2)

    has_search = search_table is not None and search_column is not None
    tools = [
        StructuredTool.from_function(
            func=hotdata_execute_sql,
            name="hotdata_execute_sql",
            description=sql_tool_description(
                search_tool_name if has_search else None,
                DEFAULT_DESCRIBE_TOOL_NAME if describe_tables else None,
            ),
        ),
        StructuredTool.from_function(
            func=hotdata_list_managed_databases,
            name="hotdata_list_managed_databases",
            description=(
                "List the managed databases in this workspace. Returns each database's "
                "'id' and its human-readable 'description'. Names are display labels and "
                "are not unique — pass the 'id' to other tools, never the description."
            ),
        ),
        StructuredTool.from_function(
            func=hotdata_create_managed_database,
            name="hotdata_create_managed_database",
            description=(
                "Create a managed database to hold tables you load. 'name' is a display "
                "label; the response carries the 'id' to use with the other tools. Declare "
                "the tables you intend to load up front as a comma- or newline-separated "
                "list, so data loads straight into them."
            ),
        ),
        StructuredTool.from_function(
            func=hotdata_load_managed_table,
            name="hotdata_load_managed_table",
            description=(
                "Load a parquet file from the local filesystem into a table that was "
                "declared on a managed database, replacing whatever the table held. "
                "'database' should be a database id. Only local parquet paths are "
                "accepted — not URLs, and not other file formats."
            ),
        ),
    ]

    if describe_tables:
        tools.append(make_hotdata_describe_tables_tool(client, database=database))

    if has_search:
        assert search_table is not None and search_column is not None
        tools.append(
            make_hotdata_search_tool(
                client,
                table=search_table,
                column=search_column,
                columns=search_columns,
                k=search_k,
                name=search_tool_name,
                max_rows=max_rows,
                database=database,
            )
        )

    return tools
