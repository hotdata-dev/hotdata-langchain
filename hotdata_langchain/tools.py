"""LangChain tools built on hotdata-framework."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from hotdata_framework import DEFAULT_SCHEMA, HotdataClient, ManagedDatabase, QueryResult
from langchain_core.tools import StructuredTool

from hotdata_langchain.databases import (
    create_managed_database,
    list_managed_databases_json,
    load_managed_table,
    load_result_summary,
    managed_database_summary,
    query_scope,
    resolve_database_by_id,
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
    *,
    search_table: str | None = None,
    search_column: str | None = None,
) -> str:
    """Return the agent-facing description for the SQL tool.

    States the engine's capabilities positively rather than listing what is absent, so
    the description does not turn into a false claim as the SQL surface grows. The one
    constraint it names is the one that silently produces wrong tool calls: an aggregate
    query that references no column is rejected.

    ``search_tool_name`` is named as a place to do text matching only when a search tool
    is actually registered alongside this one. `LIKE` is framed as a filter on text you
    already know rather than a way to find relevant rows: stating only that it "works"
    was observed to pull the model into `ILIKE '%word%'` instead of searching, which
    returns unranked results and misses related wording.

    Text ranking is also reachable *inside* SQL: ``bm25_search`` is a table-valued
    function, so a cohort identified by relevance can be joined and aggregated in one
    query. An agent given only the tool framing was observed to call search and then
    paste the returned ids back as SQL literals — correct, but capped by the tool's row
    limit and quadratic in prompt size. Naming the function, and preferring it whenever
    the answer aggregates over the matches, is what makes the composed form reachable.
    ``search_table``/``search_column`` are woven into the text when known, so the model
    is told which column is actually BM25-indexed rather than guessing one.

    Table references are asked for in full. A two-part `schema.table` reference resolves
    and returns correct rows, but the engine's index-lookup rewrite matches on the
    reference as written, so the short form can silently forfeit an index and fall back
    to a scan (datafusion-vector-search-ext#32). The wording states the preference rather
    than the current defect, so it stays accurate once that is fixed.
    """
    if search_table and search_column:
        bm25_example = (
            f"Here the BM25-indexed column is '{search_column}' on {search_table}, so "
            f"the call is bm25_search('{search_table}', '{search_column}', "
            f"'<query text>', <k>)."
        )
    else:
        bm25_example = (
            "The call is bm25_search('catalog.schema.table', '<column>', '<query text>', "
            "<k>), over a column that has a BM25 index."
        )
    composable = (
        f"To rank rows by how well their text matches a phrase, call bm25_search inside "
        f"SQL: it is a table-valued function returning the matched rows' columns plus a "
        f"`score`, so it joins, groups and nests in subqueries like any other table. "
        f"{bm25_example} Prefer this whenever the answer aggregates over the matches "
        f"rather than listing them — it keeps the whole cohort in the query instead of "
        f"passing ids back as literals."
    )
    text_guidance = (
        f"{composable} To simply list the most relevant rows, the "
        f"{search_tool_name} tool does the same ranking and returns them directly. "
        f"LIKE and ILIKE only test for a literal substring you already know, so they "
        f"are a filter, not a substitute for searching: ILIKE '%word%' returns "
        f"unranked rows and misses the related wording a search would find."
        if search_tool_name
        else f"{composable} LIKE and ILIKE only test for a literal substring, so they "
        f"are a filter, not a way to rank rows by relevance."
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
        "Address tables with all three parts: catalog.schema.table. Inside a managed "
        "database the catalog is always 'default'. A two-part schema.table reference "
        "resolves to the same rows but is not always index-accelerated, so write the "
        "full form. "
        f"{discovery}."
    )


def result_rows_for_llm(result: QueryResult, *, max_rows: int = 20) -> list[dict[str, Any]]:
    return result.to_records(max_rows=max_rows)


def execute_sql_json(
    client: HotdataClient,
    sql: str,
    *,
    max_rows: int = 100,
    database: ManagedDatabase | None = None,
) -> str:
    """Run SQL scoped to an already-resolved managed database and return JSON.

    ``database`` is a resolved ``ManagedDatabase``, not an id or a name — resolve one
    with :func:`hotdata_langchain.databases.resolve_database_by_id`.
    """
    result = client.execute_sql(sql, database=query_scope(database))
    payload = {
        "metadata": result.metadata_dict(),
        "rows": result.to_records(max_rows=max_rows),
    }
    return json.dumps(payload, indent=2)


def make_hotdata_tools(
    client: HotdataClient,
    *,
    max_rows: int = 100,
    database_id: str | ManagedDatabase | None = None,
    search_table: str | None = None,
    search_column: str | None = None,
    search_columns: Sequence[str] | None = None,
    search_k: int = DEFAULT_SEARCH_LIMIT,
    search_tool_name: str = DEFAULT_SEARCH_TOOL_NAME,
    describe_tables: bool = True,
) -> list[StructuredTool]:
    """Return LangChain tools for SQL and managed database workflows.

    ``database_id`` scopes every query these tools run to one managed database. It is a
    database id, never a name: names are display labels and are not unique. The id is
    resolved once here and the resolved record is what each query carries, so a
    non-existent id fails at build time rather than on the agent's first query. Pass an
    already-resolved ``ManagedDatabase`` to skip the lookup. Ids come from
    ``client.list_managed_databases()`` or the ``hotdata_list_managed_databases`` tool.

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

    database = resolve_database_by_id(client, database_id) if database_id is not None else None

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
        database_id: str,
        table: str,
        file: str,
        schema_name: str = DEFAULT_SCHEMA,
    ) -> str:
        """Load a local parquet file into a declared managed table."""
        loaded = load_managed_table(
            client,
            database_id=database_id,
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
                search_table=search_table if has_search else None,
                search_column=search_column if has_search else None,
            ),
        ),
        StructuredTool.from_function(
            func=hotdata_list_managed_databases,
            name="hotdata_list_managed_databases",
            description=(
                "List the managed databases in this workspace. Returns each database's "
                "'id' and its human-readable 'description'. Names are display labels and "
                "are not unique — pass the 'id' to other tools, never the description. "
                "An id cannot be guessed or built from a name; it only comes from here or "
                "from creating a database."
            ),
        ),
        StructuredTool.from_function(
            func=hotdata_create_managed_database,
            name="hotdata_create_managed_database",
            description=(
                "Create a managed database to hold tables you load. 'name' is a display "
                "label only and is not an identifier; the response carries the 'id', which "
                "is what every other tool needs — keep it. Declare the tables you intend "
                "to load up front as a comma- or newline-separated list, so data loads "
                "straight into them."
            ),
        ),
        StructuredTool.from_function(
            func=hotdata_load_managed_table,
            name="hotdata_load_managed_table",
            description=(
                "Load a parquet file from the local filesystem into a table that was "
                "declared on a managed database, replacing whatever the table held. "
                "'database_id' must be a database id returned by "
                "hotdata_list_managed_databases or hotdata_create_managed_database — call "
                "one of those first if you do not have an id. A database name is rejected: "
                "names are not unique, and this load overwrites the table, so the wrong "
                "target would destroy data. Only local parquet paths are accepted — not "
                "URLs, and not other file formats."
            ),
        ),
    ]

    if describe_tables:
        tools.append(make_hotdata_describe_tables_tool(client, database_id=database))

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
                database_id=database,
            )
        )

    return tools
