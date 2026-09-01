"""LangChain tools built on hotdata-framework."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from hotdata_framework import DEFAULT_SCHEMA, HotdataClient, ManagedDatabase, QueryResult
from langchain_core.embeddings import Embeddings
from langchain_core.tools import StructuredTool

from hotdata_langchain._sql import format_pattern_warnings
from hotdata_langchain.databases import (
    LoadMode,
    create_managed_database,
    database_label,
    list_managed_databases_json,
    load_managed_table,
    load_result_summary,
    managed_database_summary,
    query_catalogs,
    query_scope,
    resolve_database_by_id,
    scoped_description,
)
from hotdata_langchain.errors import HotdataToolError, engine_error_message, with_error_feedback
from hotdata_langchain.indexes import (
    SEMANTIC,
    TEXT,
    SearchableColumn,
    verify_searchable_columns,
)
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

#: What a tool name may contain, and how long it may be. Tool-calling APIs validate the
#: name, and 64 is the shortest limit among the providers this package is used with.
TOOL_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
MAX_TOOL_NAME_LENGTH = 64

DEFAULT_SQL_TOOL_NAME = "hotdata_execute_sql"
DEFAULT_LIST_DATABASES_TOOL_NAME = "hotdata_list_managed_databases"
DEFAULT_CREATE_DATABASE_TOOL_NAME = "hotdata_create_managed_database"
DEFAULT_LOAD_TABLE_TOOL_NAME = "hotdata_load_managed_table"

#: Tools that can destroy data a caller already has, for wiring approval around.
#: Pass straight to ``HumanInTheLoopMiddleware(interrupt_on=...)``, which is keyed by
#: tool name. These are the default names: a tool set built with ``tool_name_suffix``
#: carries different ones, so read ``tool.metadata["destructive"]`` off the built tools
#: instead of matching against this set.
#:
#: Creating a database is not here. It makes something new rather than overwriting
#: something existing, and gating it would put an approval in front of the one call an
#: agent has to make before it can do anything at all.
DESTRUCTIVE_TOOL_NAMES: frozenset[str] = frozenset({DEFAULT_LOAD_TABLE_TOOL_NAME})

logger = logging.getLogger(__name__)


def _search_examples(
    function: str,
    *,
    singular: str,
    generic: str,
    search_table: str | None,
    search_column: str | None,
    also_searchable: Sequence[SearchableColumn] = (),
) -> str:
    """Return the worked calls naming every column this function can search.

    A model was measured writing whichever call the description shows it and ignoring
    columns it was merely told about: naming a second indexed column moved the table it
    searched in 2 runs of 12, while giving that column its own worked call moved it in 8.
    So each column gets a call rather than a mention, and the order is the caller's,
    because the leading one is what a model reaches for most.
    """
    named: list[tuple[str, str]] = [(one.table, one.column) for one in also_searchable]
    if search_table and search_column and (search_table, search_column) not in named:
        named.append((search_table, search_column))

    def call(table: str, column: str) -> str:
        return f"{function}('{table}', '{column}', '<query text>', <k>)"

    if not named:
        return generic
    if len(named) == 1:
        table, column = named[0]
        return f"Here '{column}' on {table} {singular}, so the call is {call(table, column)}."
    ranked = [f"{call(table, column)} ranks {table}" for table, column in named]
    return (
        f"More than one column can be searched here, with a call for each: "
        f"{', '.join(ranked[:-1])}, and {ranked[-1]}. Search the one whose table your "
        f"answer is about."
    )


def sql_tool_description(
    search_tool_name: str | None = None,
    describe_tool_name: str | None = DEFAULT_DESCRIBE_TOOL_NAME,
    *,
    search_table: str | None = None,
    search_column: str | None = None,
    search_route: SearchRoute | None = None,
    also_searchable: Sequence[SearchableColumn] | None = None,
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
    is told which column is actually indexed rather than guessing one. They name the
    column the search tool was registered over, which is not necessarily the only indexed
    one, so the sentence says that column has an index rather than that it is *the*
    indexed column. The stronger claim was measured being followed in preference to what
    ``hotdata_describe_tables`` reports, which makes it a wrong answer the model cannot
    recover from rather than a wording preference.

    ``also_searchable`` names the other columns a caller has had confirmed, each with its
    own worked call. Which one a model picks tracks which call the description shows it,
    so the order is the caller's and the first is the one it reaches for most. This
    narrows the failure rather than removing it: the table a model chose still followed
    the leading example, and a description is written once while the right table depends
    on the question. Declaring the table an aggregate question was about moved the
    right-table rate from 0 runs of 12 to 7 (p = 0.005); the rest of what was observed —
    more composing, less substring matching — did not reach significance at that sample
    size and should not be quoted as an effect.

    ``search_route`` says which function that column is reachable through, and the
    paragraph is rewritten around it. The two descriptions arrive in one prompt, so a SQL
    description naming ``bm25_search`` beside a tool that ranks by meaning would tell the
    model to call a function that has no index on the column it was just given. The
    semantic wording also carries the sort, because ``vector_search`` returns its rows
    unsorted and a trailing ``LIMIT`` without ``ORDER BY _distance`` was measured
    returning arbitrary rows rather than the nearest ones. A plain vector index gets no
    composable call of its own: writing that search needs a query vector, which an agent
    writing SQL cannot produce. That is a fact about the registered column and not about
    the database, so it is stated of that column by name, and any ``also_searchable``
    column that does compose is still offered beside it.

    Table references are asked for in full. A two-part `schema.table` reference resolves
    and returns correct rows, but the engine's index-lookup rewrite matches on the
    reference as written, so the short form can silently forfeit an index and fall back
    to a scan (datafusion-vector-search-ext#32). The wording states the preference rather
    than the current defect, so it stays accurate once that is fixed.

    ``catalogs`` names the catalogs the tools are scoped to, so the description can state
    the catalog outright. There is no universal answer to state instead: an instant
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
    declared = list(also_searchable or ())
    text_columns = [one for one in declared if one.kind == TEXT]
    semantic_columns = [one for one in declared if one.kind == SEMANTIC and one.composable]
    for one in declared:
        if one.kind == SEMANTIC and not one.composable:
            logger.debug(
                "%r on %s is reached only with a query vector, which SQL cannot express; "
                "not naming it as composable",
                one.column,
                one.table,
            )
    # A route that cannot be composed still leaves the *other* declared columns callable,
    # so which function leads is the registered route's and neither is dropped for it.
    registered_composes = search_route is None or search_route.composable

    def text_paragraph() -> str:
        example = _search_examples(
            "bm25_search",
            singular="has a BM25 index",
            generic=(
                "The call is bm25_search('catalog.schema.table', '<column>', "
                "'<query text>', <k>), over a column that has a BM25 index."
            ),
            search_table=None if semantic else search_table,
            search_column=None if semantic else search_column,
            also_searchable=text_columns,
        )
        return (
            f"To rank rows by how well their text matches a phrase, call bm25_search "
            f"inside SQL: it is a table-valued function returning the matched rows' "
            f"columns plus a `score`, so it joins, groups and nests in subqueries like "
            f"any other table. {example} {prefer}"
        )

    def semantic_paragraph() -> str:
        named = semantic and registered_composes
        example = _search_examples(
            "vector_search",
            singular="is searchable by meaning",
            generic=(
                "The call is vector_search('catalog.schema.table', '<column>', "
                "'<query text>', <k>), over a column that is searchable by meaning."
            ),
            search_table=search_table if named else None,
            search_column=search_column if named else None,
            also_searchable=semantic_columns,
        )
        return (
            f"To rank rows by how close their meaning is to a phrase, call vector_search "
            f"inside SQL: it is a table-valued function returning the matched rows' "
            f"columns plus a `_distance` where smaller is nearer, so it joins, groups and "
            f"nests in subqueries like any other table. {example} Its rows come back "
            f"unsorted, so add ORDER BY _distance ASC — a trailing LIMIT without that "
            f"sort returns arbitrary rows rather than the nearest ones. {prefer}"
        )

    parts: list[str] = []
    if semantic:
        if registered_composes or semantic_columns:
            parts.append(semantic_paragraph())
        if not registered_composes:
            # Scoped to the registered column. "not available in SQL here" would be a
            # claim about the database made from one tool's registration, which is the
            # defect the rest of this function was corrected for.
            reach = (
                f"Ranking '{search_column}' on {search_table} by meaning needs the query "
                f"as a vector, which SQL cannot express"
            )
            parts.append(
                f"{reach}, so use the {search_tool_name} tool for it and aggregate over "
                f"what it returns."
                if search_tool_name
                else f"{reach}."
            )
        if text_columns:
            parts.append(text_paragraph())
    else:
        parts.append(text_paragraph())
        if semantic_columns:
            parts.append(semantic_paragraph())
    composable = " ".join(parts)
    # On a fused route the search tool does *not* do the same ranking: it also ranks by
    # meaning, and only the text half of that is expressible in SQL. Saying "the same
    # ranking" there would understate the tool in the one prompt that also carries the
    # tool's own description, leaving the model with two accounts that disagree.
    same_ranking = (
        "combines this ranking with one by meaning and returns the rows directly"
        if search_route is not None and search_route.hybrid
        else "does the same ranking and returns them directly"
    )
    text_guidance = (
        f"{composable} To simply list the most relevant rows, the "
        f"{search_tool_name} tool {same_ranking}. "
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
            "catalog name: an instant database answers to 'default', an attached source "
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


def suffixed_tool_name(name: str, suffix: str | None) -> str:
    """Return ``name`` with ``suffix`` appended after an underscore.

    Registering two tool sets over different databases puts two tools called
    ``hotdata_execute_sql`` in one prompt, and a model cannot address either. The suffix
    is what separates them, so it is validated here rather than at the point a provider
    rejects the call: it must be a bare token of letters, digits, underscores or hyphens,
    and the result must fit the 64-character limit.

    Returns ``name`` unchanged when ``suffix`` is ``None``.
    """
    if suffix is None:
        return name
    if not TOOL_NAME_PATTERN.fullmatch(suffix):
        raise ValueError(
            f"tool name suffix must be letters, digits, underscores or hyphens, got {suffix!r}"
        )
    combined = f"{name}_{suffix}"
    if len(combined) > MAX_TOOL_NAME_LENGTH:
        raise ValueError(
            f"{combined!r} is {len(combined)} characters; tool names are capped at "
            f"{MAX_TOOL_NAME_LENGTH}, so {suffix!r} is too long a suffix"
        )
    return combined


def result_rows_for_llm(result: QueryResult, *, max_rows: int = 20) -> list[dict[str, Any]]:
    return result.to_records(max_rows=max_rows)


def execute_sql_json(
    client: HotdataClient,
    sql: str,
    *,
    max_rows: int = 100,
    database_id: str | ManagedDatabase | None = None,
) -> str:
    """Run SQL scoped to an instant database and return JSON.

    ``database_id`` takes a database id or an already-resolved ``ManagedDatabase``, and
    is the same parameter :func:`make_hotdata_tools` takes. A name is not accepted: names
    are display labels and are not unique, so resolution is by id only. An id costs a
    lookup per call, which passing the resolved record skips — `make_hotdata_tools`
    resolves once at construction for exactly that reason.

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
    scope = resolve_database_by_id(client, database_id) if database_id is not None else None
    try:
        result = client.execute_sql(sql, database=query_scope(scope))
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
    search_semantic_column: str | None = None,
    searchable_columns: Sequence[tuple[str, str]] | None = None,
    describe_tables: bool = True,
    describe_column_stats: bool = True,
    describe_search_capabilities: bool = True,
    management_tools: bool = True,
    handle_errors: bool = False,
    allow_private_hosts: bool = False,
    catalog: str | None = None,
    tool_name_suffix: str | None = None,
    label: str | None = None,
) -> list[StructuredTool]:
    """Return LangChain tools for SQL and instant database workflows.

    ``database_id`` scopes every query these tools run to one instant database. It is a
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
    ``describe_search_capabilities`` (on by default) has it also report what each column
    can be searched by, which is the one fact about a table that SQL cannot answer; it
    costs one control-plane call per table described.

    ``tool_name_suffix`` appends a token to every tool name in the set, so two sets built
    over different databases do not both register ``hotdata_execute_sql`` — a name a model
    cannot address twice. An explicit ``search_tool_name`` is used exactly as given, since
    naming that tool is already the caller's own decision.

    ``label`` names the database at the front of every description that is scoped to it —
    the SQL, schema and search tools, but not the instant-database tools, which act on the
    workspace rather than on one database. It defaults to the resolved database's name,
    and a database with no name gets no sentence rather than one naming its id.

    ``management_tools`` (on by default) adds the three tools that work on instant
    databases themselves — listing, creating and loading. Turn it off for an agent that
    reads one fixed database, where they are surface the model can only misuse. The flag
    is not called ``read_only``: listing databases is itself a read, so the set it removes
    is the instant-database workflow rather than everything that writes. The load tool
    carries ``metadata={"destructive": True}``; :data:`DESTRUCTIVE_TOOL_NAMES` holds the
    same set under the default names.

    The create tool takes ``keys`` and ``expires_at`` but not ``partition_by`` or
    ``sorted_by``, which :func:`~hotdata_langchain.databases.create_managed_database`
    accepts. Layout is permanent — the API has no ALTER path, and undoing a choice means
    deleting the table and reloading it, which burns the table name in that database —
    so it is set by the caller building the tools, not chosen per call by a model.

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
    forces one; ``"semantic"`` raises if no vector index covers the column, ``"hybrid"``
    raises if the two cannot be fused, while ``"text"``
    falls through to the engine's own error, because index introspection fails open.
    ``search_embedding`` is a LangChain
    ``Embeddings`` and is required for a *plain* vector index, where the engine cannot
    embed the query itself; it must be the same model the column was written with.

    Supplying ``search_embedding`` beside a BM25 column *fuses* the two: the text tool
    keeps its name and its contract and ranks by both wording and meaning, combining them
    with reciprocal rank fusion in a single query. The two miss different things — BM25
    misses paraphrase, vector search misses rare exact tokens — so doing both beats making
    the model choose, which is why this is not offered as a second tool. It applies only
    where the engine allows both indexes on one table, which rules out a provider-backed
    vector index. ``search_semantic_column`` names the vector column to pair with, and is
    needed only when the table carries more than one; ``search_strategy="text"`` opts back
    out.

    ``search_columns`` selects the columns each hit returns; left unset, a hit carries the
    searched column plus ``search_key_column`` when the table has one, so the hit can be
    joined back to the table it came from. Pass ``search_key_column=None`` for the searched
    column alone. A search over a vector column never returns that column, so there the key
    carries the hit on its own unless ``search_columns`` names more.

    ``searchable_columns`` names other indexed columns as ``(table, column)`` pairs, each
    written ``catalog.schema.table``. The search tool ranks one corpus, so ``search_table``
    describes that one and nothing else; a database usually has more, and the SQL tool's
    description is the only place a model learns they exist. Each pair is confirmed
    against the control plane before it is named, and one no ready index covers is dropped
    with a warning rather than offered. Order carries: the first is the one a model
    reaches for most, so lead with the table most questions are about. A malformed pair
    raises ``ValueError`` here, and without ``database_id`` there is no scope to confirm
    against, so the argument is ignored entirely. A declared column of the kind the
    registered route does not use is still named, through its own function — a BM25
    column beside a semantic search tool composes perfectly well, and dropping it would
    repeat the defect this parameter exists to fix. Only a plain vector column is left
    out, because writing that search needs a query vector.

    Supplying only one of ``search_table``/``search_column`` raises ``ValueError``.

    For more than one searchable corpus, call
    :func:`hotdata_langchain.search.make_hotdata_search_tool` directly per corpus and
    extend this list.

    ``catalog`` is the catalog name the SQL tool tells the model to address tables with.
    When it is omitted and the tools are scoped to a database, the catalogs are read from
    that database's ``information_schema`` once, here — an instant database answers to
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
        return execute_sql_json(client, sql, max_rows=max_rows, database_id=database)

    def hotdata_list_managed_databases() -> str:
        """List Hotdata instant databases in the workspace."""
        return list_managed_databases_json(client)

    def hotdata_create_managed_database(
        name: str,
        schema_name: str = DEFAULT_SCHEMA,
        tables: str = "",
        keys: dict[str, list[str]] | None = None,
        expires_at: str = "",
    ) -> str:
        """Create an instant database and optionally declare tables.

        Args:
            name: display label for the database; it is not an identifier, and the
                response carries the id every other tool needs.
            schema_name: schema the declared tables live in.
            tables: table names to declare up front, comma- or newline-separated.
            keys: each table's natural key, as a table name mapped to its key columns.
                A key can only be set here. A table declared without one can never be
                loaded with upsert, update or delete.
            expires_at: when to reap the database, as an RFC 3339 timestamp or a
                relative window such as '24h' or '7d'. Left empty it lives until
                something deletes it.
        """
        table_names = [t.strip() for t in tables.replace(",", "\n").splitlines() if t.strip()]
        undeclared = sorted(set(keys or {}) - set(table_names))
        if undeclared:
            raise ValueError(
                f"keys names {undeclared}, which is not among the declared tables "
                f"{table_names}. A key can only be set when its table is declared, so a "
                "key on a table that is not created here can never take effect."
            )
        db = create_managed_database(
            client,
            name=name,
            schema=schema_name or DEFAULT_SCHEMA,
            tables=table_names or None,
            keys=keys or None,
            expires_at=expires_at or None,
        )
        return json.dumps(managed_database_summary(db), indent=2)

    def hotdata_load_managed_table(
        database_id: str,
        table: str,
        file: str,
        schema_name: str = DEFAULT_SCHEMA,
        mode: LoadMode = "replace",
        key: list[str] | None = None,
    ) -> str:
        """Load a parquet file, local or at a URL, into a declared managed table.

        Args:
            database_id: id of the target database, as returned by listing or creating
                one; a database name is rejected.
            table: name of a table already declared on that database.
            file: a local filesystem path, or an http:// or https:// URL, to a parquet
                file. Only parquet is accepted.
            schema_name: schema the table was declared in.
            mode: what happens to rows already in the table. 'replace' discards them and
                'append' keeps them. The other three match an incoming row to an existing
                one by the table's key. 'upsert' replaces a matched row and inserts one
                that matches nothing, 'update' replaces a matched row only, and 'delete'
                removes a matched row and inserts nothing.
            key: the key columns to match on, required by upsert, update and delete.
                They must be the columns the table was declared with.
        """
        loaded = load_managed_table(
            client,
            database_id=database_id,
            table=table,
            file=file,
            schema=schema_name or DEFAULT_SCHEMA,
            mode=mode,
            key=key or None,
            allow_private_hosts=allow_private_hosts,
        )
        return json.dumps(load_result_summary(loaded), indent=2)

    # Saves the model a turn it would otherwise spend discovering the rule from an error.
    url_rule = (
        ""
        if allow_private_hosts
        else " (a URL must be on the public internet, not an internal address)"
    )

    # Confirmed before it is named. A column reported searchable that no index covers
    # sends the model to a function with no fallback, and the error arrives after it has
    # committed to the route.
    confirmed = verify_searchable_columns(
        client, columns=list(searchable_columns or ()), database=database
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
            semantic_column=search_semantic_column,
            key_column=search_key_column,
        )
        if has_search and search_table is not None and search_column is not None
        else None
    )
    # The name reaches the model too, so it follows the route: a semantic search called
    # "search_text" would tell the model it matches wording, which is what it does not do.
    resolved_search_name = search_tool_name or suffixed_tool_name(
        DEFAULT_SEMANTIC_TOOL_NAME
        if search_route is not None and search_route.semantic
        else DEFAULT_SEARCH_TOOL_NAME,
        tool_name_suffix,
    )
    sql_name = suffixed_tool_name(DEFAULT_SQL_TOOL_NAME, tool_name_suffix)
    describe_name = suffixed_tool_name(DEFAULT_DESCRIBE_TOOL_NAME, tool_name_suffix)
    scope_label = label if label is not None else database_label(database)
    tools = [
        StructuredTool.from_function(
            func=hotdata_execute_sql,
            name=sql_name,
            description=scoped_description(
                sql_tool_description(
                    resolved_search_name if has_search else None,
                    describe_name if describe_tables else None,
                    search_table=search_table if has_search else None,
                    search_column=search_column if has_search else None,
                    search_route=search_route,
                    also_searchable=confirmed,
                    catalogs=catalogs,
                    max_rows=max_rows,
                ),
                scope_label,
            ),
            parse_docstring=True,
        ),
    ]

    if management_tools:
        # Names resolved here rather than above, so a suffix too long for one of these
        # is reported against a tool the caller actually asked for.
        list_name = suffixed_tool_name(DEFAULT_LIST_DATABASES_TOOL_NAME, tool_name_suffix)
        create_name = suffixed_tool_name(DEFAULT_CREATE_DATABASE_TOOL_NAME, tool_name_suffix)
        load_name = suffixed_tool_name(DEFAULT_LOAD_TABLE_TOOL_NAME, tool_name_suffix)
        tools.extend(
            [
                StructuredTool.from_function(
                    func=hotdata_list_managed_databases,
                    name=list_name,
                    description=(
                        "List the instant databases in this workspace. Returns each database's "
                        "'id' and its human-readable 'name'. Names are display labels and are "
                        "not unique — pass the 'id' to other tools, never the name. "
                        "An id cannot be guessed or built from a name; it only comes from here or "
                        "from creating a database."
                    ),
                ),
                StructuredTool.from_function(
                    func=hotdata_create_managed_database,
                    name=create_name,
                    description=(
                        "Create an instant database to hold tables you load. 'name' is a display "
                        "label only and is not an identifier; the response carries the 'id', which "
                        "is what every other tool needs — keep it. Declare the tables you intend "
                        "to load up front as a comma- or newline-separated list, so data loads "
                        "straight into them. Declare 'keys' at the same time for any table you "
                        "will load more than once: a key can only be set here, and a table "
                        "created without one can never be loaded with upsert, update or delete. "
                        "Set 'expires_at' when the data is temporary, so the database is reaped "
                        "rather than left behind."
                    ),
                    parse_docstring=True,
                ),
                StructuredTool.from_function(
                    func=hotdata_load_managed_table,
                    name=load_name,
                    description=(
                        "Load a parquet file into a table that was declared on an instant "
                        "database. By default this replaces whatever the table held; pass 'mode' "
                        "to keep it. 'append' adds rows blindly. The other three match an "
                        "incoming row to an existing one by key, so they need 'key' and work "
                        "only on a table that was declared with one: 'upsert' replaces a matched "
                        "row and inserts one that matches nothing, 'update' replaces a matched "
                        "row only, and 'delete' REMOVES a matched row and inserts nothing — the "
                        "rows you upload choose what is deleted, they are not added. "
                        "'file' is either a path on "
                        "the local filesystem or an http:// or https:// URL, which is downloaded "
                        f"and uploaded for you{url_rule}. 'database_id' must be a database id "
                        f"returned by {list_name} or {create_name} — call one of those first if "
                        "you do not have an id. A database name is rejected: "
                        "names are not unique, and this load overwrites the table, so the wrong "
                        "target would destroy data. Only parquet is accepted, not CSV or JSON. "
                        "If a load fails without saying whether it landed, do not repeat it with "
                        "mode='append': the retry adds the rows a second time. Re-run 'replace', "
                        "or use 'upsert' with a key — both land the same rows however many "
                        "times they run."
                    ),
                    parse_docstring=True,
                    metadata={"destructive": True},
                ),
            ]
        )

    if describe_tables:
        tools.append(
            make_hotdata_describe_tables_tool(
                client,
                database_id=database,
                name=describe_name,
                column_stats=describe_column_stats,
                search_capabilities=describe_search_capabilities,
                catalogs=catalogs,
                label=scope_label,
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
                label=scope_label,
            )
        )

    return with_error_feedback(tools) if handle_errors else tools
