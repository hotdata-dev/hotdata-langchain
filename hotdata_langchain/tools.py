"""LangChain tools built on hotdata-framework."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from hotdata_framework import DEFAULT_SCHEMA, HotdataClient, ManagedDatabase, QueryResult
from langchain_core.embeddings import Embeddings
from langchain_core.tools import StructuredTool

from hotdata_langchain._sql import format_pattern_warnings
from hotdata_langchain.databases import (
    create_managed_database,
    list_managed_databases_json,
    load_managed_table,
    load_result_summary,
    managed_database_summary,
    query_catalogs,
    query_scope,
    resolve_database_by_id,
)
from hotdata_langchain.errors import HotdataToolError, engine_error_message, with_error_feedback
from hotdata_langchain.results import result_json
from hotdata_langchain.schema import (
    DEFAULT_DESCRIBE_TOOL_NAME,
    make_hotdata_describe_tables_tool,
)
from hotdata_langchain.search import (
    DEFAULT_KEY_COLUMN,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEARCH_TOOL_NAME,
    DEFAULT_SEMANTIC_TOOL_NAME,
    SearchRoute,
    SearchStrategy,
    make_hotdata_search_tool,
    resolve_search_route,
)

DEFAULT_SQL_TOOL_NAME = "hotdata_execute_sql"
DEFAULT_LIST_DATABASES_TOOL_NAME = "hotdata_list_managed_databases"
DEFAULT_CREATE_DATABASE_TOOL_NAME = "hotdata_create_managed_database"
DEFAULT_LOAD_TABLE_TOOL_NAME = "hotdata_load_managed_table"


def sql_tool_description(
    search_tool_name: str | None = None,
    describe_tool_name: str | None = DEFAULT_DESCRIBE_TOOL_NAME,
    *,
    search_table: str | None = None,
    search_column: str | None = None,
    search_route: SearchRoute | None = None,
    catalogs: Sequence[str] | None = None,
    max_rows: int | None = None,
) -> str:
    """Return the agent-facing description for the SQL tool.

    States the engine's capabilities positively rather than listing what is absent, so
    the description does not turn into a false claim as the SQL surface grows. The
    constraints it does name are the ones that silently produce wrong tool calls rather
    than errors the model can read and retry against.

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
    is told which column is actually indexed rather than guessing one.

    ``search_route`` says which function that column is reachable through, and the
    paragraph is rewritten around it. The two descriptions arrive in one prompt, so a SQL
    description naming ``bm25_search`` beside a tool that ranks by meaning would tell the
    model to call a function that has no index on the column it was just given. The
    semantic wording also carries the sort, because ``vector_search`` returns its rows
    unsorted and a trailing ``LIMIT`` without ``ORDER BY _distance`` was measured
    returning arbitrary rows rather than the nearest ones. A plain vector index gets no
    composable paragraph at all: writing that search needs a query vector, which an agent
    writing SQL cannot produce.

    Table references are asked for in full. A two-part `schema.table` reference resolves
    and returns correct rows, but the engine's index-lookup rewrite matches on the
    reference as written, so the short form can silently forfeit an index and fall back
    to a scan (datafusion-vector-search-ext#32). The wording states the preference rather
    than the current defect, so it stays accurate once that is fixed.

    ``catalogs`` names the catalogs the tools are scoped to, so the description can state
    the catalog outright. There is no universal answer to state instead: a managed
    database answers to `default`, an attached source answers to its attachment alias,
    and the database record reports `default` either way. With none supplied the model is
    pointed at `information_schema` rather than given a rule that holds for one of the two
    kinds.

    The dialect is named as DataFusion rather than PostgreSQL. Calling it PostgreSQL
    reinforced the prior that produced the one measured silent-wrong-value failure: the
    model wrote valid PostgreSQL date formatting and got a column of literal format
    strings back. Naming the engine gives a prior that holds for divergences not yet
    found, where a list of exceptions only covers the ones already measured.

    The date/time wording says a PostgreSQL pattern does not reliably raise, rather than
    that it echoes. Echoing is a current defect that the engine may fix, so asserting it
    would put a false claim in the model's contract the day it is fixed. Behaviour that is
    being corrected upstream is stated as an absence of a guarantee, never as a rule.

    ``max_rows`` states the row cap and what ``row_count`` counts. The gap between the two
    is arithmetic a model can do — one was measured spotting it and paginating unprompted
    — but nothing said where the cap fell, so it guessed the boundary and re-read rows it
    already had. The number is stated rather than left to be discovered.
    """
    semantic = search_route is not None and search_route.semantic
    prefer = (
        "Prefer this whenever the answer aggregates over the matches rather than listing "
        "them — it keeps the whole cohort in the query instead of passing ids back as "
        "literals."
    )
    if semantic and search_route is not None and not search_route.composable:
        # No composable form: this index needs a query vector, which SQL cannot express.
        composable = (
            f"Ranking rows by meaning is not available in SQL here — it needs the query "
            f"as a vector, which SQL cannot express — so use the {search_tool_name} tool "
            f"for it and aggregate over what it returns."
            if search_tool_name
            else "Ranking rows by meaning is not available in SQL here: it needs the "
            "query as a vector, which SQL cannot express."
        )
    elif semantic:
        if search_table and search_column:
            example = (
                f"Here the column searchable by meaning is '{search_column}' on "
                f"{search_table}, so the call is vector_search('{search_table}', "
                f"'{search_column}', '<query text>', <k>)."
            )
        else:
            example = (
                "The call is vector_search('catalog.schema.table', '<column>', "
                "'<query text>', <k>), over a column that is searchable by meaning."
            )
        composable = (
            f"To rank rows by how close their meaning is to a phrase, call vector_search "
            f"inside SQL: it is a table-valued function returning the matched rows' "
            f"columns plus a `_distance` where smaller is nearer, so it joins, groups and "
            f"nests in subqueries like any other table. {example} Its rows come back "
            f"unsorted, so add ORDER BY _distance ASC — a trailing LIMIT without that "
            f"sort returns arbitrary rows rather than the nearest ones. {prefer}"
        )
    else:
        if search_table and search_column:
            example = (
                f"Here the BM25-indexed column is '{search_column}' on {search_table}, so "
                f"the call is bm25_search('{search_table}', '{search_column}', "
                f"'<query text>', <k>)."
            )
        else:
            example = (
                "The call is bm25_search('catalog.schema.table', '<column>', "
                "'<query text>', <k>), over a column that has a BM25 index."
            )
        composable = (
            f"To rank rows by how well their text matches a phrase, call bm25_search "
            f"inside SQL: it is a table-valued function returning the matched rows' "
            f"columns plus a `score`, so it joins, groups and nests in subqueries like "
            f"any other table. {example} {prefer}"
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
    known = list(catalogs or ())
    if len(known) == 1:
        catalog_rule = f"Here the catalog is '{known[0]}'."
    elif known:
        catalog_rule = (
            f"This database exposes more than one catalog ({', '.join(known)}) — read "
            f"table_catalog from information_schema.tables to see which one holds a table."
        )
    else:
        catalog_rule = (
            "Read table_catalog from information_schema.tables rather than assuming a "
            "catalog name: a managed database answers to 'default', an attached source "
            "answers to its own name."
        )
    row_cap = (
        ""
        if max_rows is None
        else (
            f"At most {max_rows} rows come back. metadata.row_count is how many the query "
            f"matched before that cap, so a row_count above the number of rows you "
            f"received means you are reading a prefix of the answer: aggregate in SQL "
            f"when the question is about the whole set, and page with LIMIT/OFFSET only "
            f"when you genuinely need the rest of the rows."
        )
    )
    return (
        "Run a read-only SQL query and return the rows as JSON. The engine is Apache "
        "DataFusion, whose SQL follows PostgreSQL closely: joins, CTEs, subqueries, "
        "GROUP BY, window functions, ORDER BY/LIMIT and the usual scalar functions all "
        "work. Where the two differ, DataFusion is what runs, so prefer a DataFusion "
        "function over a PostgreSQL-only one when you are unsure.\n"
        f"{text_guidance}\n"
        "Date and time handling is one place they differ: format patterns are strftime, "
        "so write to_char(<date>, '%Y-%m-%d'). A PostgreSQL pattern like 'YYYY-MM-DD' is "
        "wrong here and does not reliably raise — it can come back as the literal text on "
        "every row — so never assume a bad pattern will announce itself. There is no "
        "date_sub or date_add: subtract an interval, as in <date> - INTERVAL '6 days'. "
        "date_trunc, now() and current_date work.\n"
        "Some tables reject a projection naming none of their own columns: COUNT(*), "
        "COUNT(1) and SELECT 1 can fail with 'must either specify a row count or at "
        "least one column'. Naming a column always works, so prefer COUNT(<column>).\n"
        "Identifiers are lowercased when stored, so a name that looks camelCase is "
        "already lowercase and quoting it to preserve case fails: write driverId or "
        'driverid, never "driverId".\n'
        f"Address tables with all three parts: catalog.schema.table. {catalog_rule} A "
        "two-part schema.table reference resolves to the same rows but is not always "
        "index-accelerated, so write the full form. "
        f"{discovery}." + (f"\n{row_cap}" if row_cap else "")
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

    ``metadata.client_warning`` carries anything this package noticed about a call that
    succeeded without doing what it said: rows capped at ``max_rows``, or a date/time
    format pattern the engine will not interpret. The engine's own
    ``metadata.warning`` is separate and passed through untouched.

    A format pattern that makes the query *fail* is reported the same way, in the
    exception's message and ahead of the engine's own. That is the case where the
    package's read matters most: applying a PostgreSQL template to a column, rather than
    to a literal, was measured returning nothing more specific than "An internal server
    error occurred", so the model has only this to work from.
    """
    warnings = format_pattern_warnings(sql)
    try:
        result = client.execute_sql(sql, database=query_scope(database))
    except Exception as exc:
        if not warnings:
            raise
        raise HotdataToolError(
            " ".join([*warnings, f"The engine reported: {engine_error_message(exc)}"])
        ) from exc
    return result_json(result, max_rows=max_rows, warnings=warnings)


def make_hotdata_tools(
    client: HotdataClient,
    *,
    max_rows: int = 100,
    database_id: str | ManagedDatabase | None = None,
    search_table: str | None = None,
    search_column: str | None = None,
    search_columns: Sequence[str] | None = None,
    search_key_column: str | None = DEFAULT_KEY_COLUMN,
    search_k: int = DEFAULT_SEARCH_LIMIT,
    search_tool_name: str | None = None,
    search_strategy: SearchStrategy = "auto",
    search_embedding: Embeddings | None = None,
    describe_tables: bool = True,
    describe_column_stats: bool = True,
    management_tools: bool = True,
    handle_errors: bool = False,
    allow_private_hosts: bool = False,
    catalog: str | None = None,
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
    ``describe_column_stats`` (on by default) has that tool also report how many rows
    hold a value in each column, so an empty column is visible as such rather than as an
    ordinary typed one; it costs one aggregate query per table described.

    ``management_tools`` (on by default) adds the three tools that work on managed
    databases themselves — listing, creating and loading. Turn it off for an agent that
    reads one fixed database, where they are surface the model can only misuse. The flag
    is not called ``read_only``: listing databases is itself a read, so the set it removes
    is the managed-database workflow rather than everything that writes.

    ``handle_errors`` returns each tool's failures as ``{"error": "<engine message>"}``
    instead of raising. An exception out of a tool aborts the whole agent run, so one
    invalid query ends the conversation rather than costing a turn; with this on, the
    model reads the engine's message and retries. Off by default, because these tools are
    thin pass-throughs outside an agent loop and swallowing an exception there would hide
    a real failure. See :func:`hotdata_langchain.errors.with_error_feedback` to apply the
    same wrapping to tools built elsewhere.

    ``allow_private_hosts`` lets the load tool fetch a URL resolving to a private address.
    Off by default: the URL is chosen by the model, and the model's inputs include text it
    retrieved, so the default must not let a planted link reach a service only this process
    can see. Turn it on when the data genuinely sits on an internal host. See
    :func:`hotdata_langchain.databases.reject_unroutable_url`.

    Passing both ``search_table`` and ``search_column`` appends a search tool bound to
    that column, which requires a search index on it. Which kind of search it does is read
    off that column's indexes here rather than chosen: a BM25 index gives
    ``hotdata_search_text``, a vector index ``hotdata_search_semantic``. ``search_strategy``
    forces one; ``"semantic"`` raises if no vector index covers the column, while ``"text"``
    falls through to the engine's own error, because index introspection fails open.
    ``search_embedding`` is a LangChain
    ``Embeddings`` and is required only for a *plain* vector index, where the engine cannot
    embed the query itself; it must be the same model the column was written with.

    ``search_columns`` selects the columns each hit returns; left unset, a hit carries the
    searched column plus ``search_key_column`` when the table has one, so the hit can be
    joined back to the table it came from. Pass ``search_key_column=None`` for the searched
    column alone. A search over a vector column never returns that column, so there the key
    carries the hit on its own unless ``search_columns`` names more.

    Supplying only one of ``search_table``/``search_column`` raises ``ValueError``.

    For more than one searchable corpus, call
    :func:`hotdata_langchain.search.make_hotdata_search_tool` directly per corpus and
    extend this list.

    ``catalog`` is the catalog name the SQL tool tells the model to address tables with.
    When it is omitted and the tools are scoped to a database, the catalogs are read from
    that database's ``information_schema`` once, here — a managed database answers to
    ``default`` and an attached source answers to its attachment alias, so there is no
    correct constant to fall back on. Pass it to skip that lookup.
    """
    if (search_table is None) != (search_column is None):
        raise ValueError("search_table and search_column must be provided together")

    database = resolve_database_by_id(client, database_id) if database_id is not None else None
    if catalog is not None:
        catalogs = [catalog]
    elif database is not None:
        catalogs = query_catalogs(client, database)
    else:
        catalogs = []

    def hotdata_execute_sql(sql: str) -> str:
        """Run SQL against the Hotdata workspace and return JSON rows.

        Args:
            sql: one read-only DataFusion SQL statement. Rows are capped, so read
                metadata.row_count for the total the query matched before that cap.
        """
        return execute_sql_json(client, sql, max_rows=max_rows, database=database)

    def hotdata_list_managed_databases() -> str:
        """List Hotdata-managed databases in the workspace."""
        return list_managed_databases_json(client)

    def hotdata_create_managed_database(
        name: str,
        schema_name: str = DEFAULT_SCHEMA,
        tables: str = "",
    ) -> str:
        """Create a managed database and optionally declare tables.

        Args:
            name: display label for the database; it is not an identifier, and the
                response carries the id every other tool needs.
            schema_name: schema the declared tables live in.
            tables: table names to declare up front, comma- or newline-separated.
        """
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
        """Load a parquet file, local or at a URL, into a declared managed table.

        Args:
            database_id: id of the target database, as returned by listing or creating
                one; a database name is rejected.
            table: name of a table already declared on that database. The load replaces
                whatever it holds.
            file: a local filesystem path, or an http:// or https:// URL, to a parquet
                file. Only parquet is accepted.
            schema_name: schema the table was declared in.
        """
        loaded = load_managed_table(
            client,
            database_id=database_id,
            table=table,
            file=file,
            schema=schema_name or DEFAULT_SCHEMA,
            allow_private_hosts=allow_private_hosts,
        )
        return json.dumps(load_result_summary(loaded), indent=2)

    # Saves the model a turn it would otherwise spend discovering the rule from an error.
    url_rule = (
        ""
        if allow_private_hosts
        else " (a URL must be on the public internet, not an internal address)"
    )

    has_search = search_table is not None and search_column is not None
    # Resolved once, before either description is built: the SQL tool and the search tool
    # both describe this column to the same model in the same prompt, and resolving twice
    # is how the two would come to disagree.
    search_route = (
        resolve_search_route(
            client,
            table=search_table,
            column=search_column,
            database=database,
            strategy=search_strategy,
            has_embedding=search_embedding is not None,
        )
        if has_search and search_table is not None and search_column is not None
        else None
    )
    # The name reaches the model too, so it follows the route: a semantic search called
    # "search_text" would tell the model it matches wording, which is what it does not do.
    resolved_search_name = search_tool_name or (
        DEFAULT_SEMANTIC_TOOL_NAME
        if search_route is not None and search_route.semantic
        else DEFAULT_SEARCH_TOOL_NAME
    )
    tools = [
        StructuredTool.from_function(
            func=hotdata_execute_sql,
            name=DEFAULT_SQL_TOOL_NAME,
            description=sql_tool_description(
                resolved_search_name if has_search else None,
                DEFAULT_DESCRIBE_TOOL_NAME if describe_tables else None,
                search_table=search_table if has_search else None,
                search_column=search_column if has_search else None,
                search_route=search_route,
                catalogs=catalogs,
                max_rows=max_rows,
            ),
            parse_docstring=True,
        ),
    ]

    management = [
        StructuredTool.from_function(
            func=hotdata_list_managed_databases,
            name=DEFAULT_LIST_DATABASES_TOOL_NAME,
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
            name=DEFAULT_CREATE_DATABASE_TOOL_NAME,
            description=(
                "Create a managed database to hold tables you load. 'name' is a display "
                "label only and is not an identifier; the response carries the 'id', which "
                "is what every other tool needs — keep it. Declare the tables you intend "
                "to load up front as a comma- or newline-separated list, so data loads "
                "straight into them."
            ),
            parse_docstring=True,
        ),
        StructuredTool.from_function(
            func=hotdata_load_managed_table,
            name=DEFAULT_LOAD_TABLE_TOOL_NAME,
            description=(
                "Load a parquet file into a table that was declared on a managed "
                "database, replacing whatever the table held. 'file' is either a path on "
                "the local filesystem or an http:// or https:// URL, which is downloaded "
                f"and uploaded for you{url_rule}. 'database_id' must be a database id returned by "
                "hotdata_list_managed_databases or hotdata_create_managed_database — call "
                "one of those first if you do not have an id. A database name is rejected: "
                "names are not unique, and this load overwrites the table, so the wrong "
                "target would destroy data. Only parquet is accepted, not CSV or JSON."
            ),
            parse_docstring=True,
        ),
    ]

    if management_tools:
        tools.extend(management)

    if describe_tables:
        tools.append(
            make_hotdata_describe_tables_tool(
                client,
                database_id=database,
                column_stats=describe_column_stats,
            )
        )

    if has_search:
        assert search_table is not None and search_column is not None
        tools.append(
            make_hotdata_search_tool(
                client,
                table=search_table,
                column=search_column,
                columns=search_columns,
                key_column=search_key_column,
                k=search_k,
                name=resolved_search_name,
                max_rows=max_rows,
                database_id=database,
                embedding=search_embedding,
                route=search_route,
            )
        )

    return with_error_feedback(tools) if handle_errors else tools
