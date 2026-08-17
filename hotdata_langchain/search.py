"""Full-text (BM25) search helpers and tools for LangChain agents."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from hotdata_framework import HotdataClient, ManagedDatabase
from langchain_core.tools import StructuredTool

from hotdata_langchain._sql import quote_literal, validate_identifier
from hotdata_langchain.databases import query_scope, resolve_database_by_id
from hotdata_langchain.results import SEARCH_REMEDY, result_json

logger = logging.getLogger(__name__)

#: Column the engine appends to every ``bm25_search`` result, holding the BM25 relevance score.
SCORE_COLUMN = "score"

#: Default number of ranked hits requested when a caller does not specify one.
DEFAULT_SEARCH_LIMIT = 5

#: Column added to the default projection when the searched table has one, so a hit
#: carries the value that joins it back to the rest of the table.
DEFAULT_KEY_COLUMN = "id"

DEFAULT_SEARCH_TOOL_NAME = "hotdata_search_text"

_TABLE_REF_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*"
)


def _validate_table_ref(table: str) -> str:
    """Return ``table`` if it is a bare ``catalog.schema.table`` reference, else raise."""
    if not _TABLE_REF_RE.fullmatch(table):
        raise ValueError(
            "table must be a fully qualified 'catalog.schema.table' reference "
            f"of bare identifiers, got {table!r}"
        )
    return table


def _projection(column: str, columns: Sequence[str] | None) -> list[str]:
    selected = list(columns) if columns is not None else [column]
    if not selected:
        raise ValueError("columns must not be empty")
    for name in selected:
        validate_identifier(name, label="column")
    return [*(name for name in selected if name != SCORE_COLUMN), SCORE_COLUMN]


def bm25_search_sql(
    *,
    table: str,
    column: str,
    query: str,
    k: int = DEFAULT_SEARCH_LIMIT,
    columns: Sequence[str] | None = None,
) -> str:
    """Build the SQL for a ranked BM25 top-k search over an indexed text column.

    ``table`` is a fully qualified ``catalog.schema.table`` reference. Inside a managed
    database the built-in catalog is always ``default``, so a managed table reads as
    ``default.public.listings`` when the query is scoped to that database.

    ``column`` must be a column carrying a BM25 index; the engine has no brute-force
    fallback and errors when no index exists. ``columns`` selects which table columns
    come back (defaulting to the searched column alone); ``score`` is always appended
    last and never duplicated.

    ``k`` is emitted twice, and both are load-bearing. It is passed as the ``bm25_search``
    fourth argument, which bounds the search tantivy runs, and again as a trailing
    ``LIMIT``. The explicit argument is what actually caps the scan: ``ORDER BY`` blocks
    limit pushdown, so a query relying on the trailing ``LIMIT`` alone falls back to the
    engine's much larger default bound.

    Raises ``ValueError`` for identifiers that are not bare SQL identifiers, for a
    non-positive ``k``, and for search text containing null bytes.
    """
    _validate_table_ref(table)
    validate_identifier(column, label="column")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    projection = _projection(column, columns)
    return (
        f"SELECT {', '.join(projection)} "
        f"FROM bm25_search("
        f"{quote_literal(table)}, {quote_literal(column)}, {quote_literal(query)}, {k}) "
        f"ORDER BY {SCORE_COLUMN} DESC "
        f"LIMIT {k}"
    )


def bm25_search_json(
    client: HotdataClient,
    *,
    table: str,
    column: str,
    query: str,
    k: int = DEFAULT_SEARCH_LIMIT,
    columns: Sequence[str] | None = None,
    max_rows: int = 100,
    database: ManagedDatabase | None = None,
    warnings: Sequence[str] = (),
) -> str:
    """Run a BM25 search and return ``{"metadata": ..., "rows": [...]}`` as JSON.

    Mirrors the envelope :func:`hotdata_langchain.tools.execute_sql_json` returns, so an
    agent sees one result shape across every Hotdata tool. Rows arrive ranked by
    ``score`` descending.

    ``database`` is a resolved ``ManagedDatabase``, not an id or a name — resolve one
    with :func:`hotdata_langchain.databases.resolve_database_by_id`.

    ``warnings`` are client-side notes to carry in ``metadata.client_warning`` alongside
    the one this adds when the result is capped at ``max_rows``. That one is phrased for
    a caller who supplies a search string rather than SQL, since paging or rewriting the
    query is not something this tool's caller can do.
    """
    sql = bm25_search_sql(table=table, column=column, query=query, k=k, columns=columns)
    result = client.execute_sql(sql, database=query_scope(database))
    return result_json(result, max_rows=max_rows, warnings=warnings, remedy=SEARCH_REMEDY)


def clamp_warning(*, requested: int, ceiling: int) -> str | None:
    """Return the warning for a model-supplied ``k`` cut to ``ceiling``, or ``None``.

    The clamp happens before the query runs, so the engine only ever ranks ``ceiling``
    rows and ``row_count`` honestly reports them. Nothing in the result distinguishes
    that from a corpus with only that many matches, which is the whole defect: an agent
    was measured asking for 200, receiving 100, and reporting a cohort it believed was
    200.
    """
    if requested <= ceiling:
        return None
    return (
        f"Asked for k={requested}, but this tool ranks at most {ceiling} rows, so k was "
        f"reduced to {ceiling} before searching. These are the top {ceiling} matches, "
        f"not a sample of {requested}, and rows beyond {ceiling} were never ranked. To "
        f"reason over a wider cohort, call bm25_search inside SQL and aggregate there "
        f"rather than raising k here."
    )


def default_search_description(
    table: str,
    column: str,
    *,
    columns: Sequence[str] | None = None,
    max_k: int | None = None,
) -> str:
    """Return the agent-facing tool description used when no override is given.

    Describes the capability ("find rows whose text is relevant") rather than the index
    behind it, so the contract the model is given survives the retrieval strategy
    changing underneath it.

    This tool is registered alongside the SQL tool, so both descriptions reach the model
    in one prompt and must agree. Two earlier sentences here contradicted it: that SQL
    cannot rank rows by textual relevance, which is false, and an instruction to carry the
    returned values into SQL, which is the measured failure — an agent pasted 100 literal
    ids into `WHERE id IN (...)`, capping the cohort at this tool's row limit rather than
    at intent. Ranking inside SQL is named as the route for an aggregate; this tool is
    described as the route for listing and inspecting matches.

    The `LIKE`/`ILIKE` guard the removed sentence carried is kept, because stating only
    that `LIKE` "works" was observed to pull models into `ILIKE '%word%'` instead of
    searching.

    ``columns`` names what a hit carries, so the model can see whether a result can be
    joined back to the table rather than discovering it from the rows. ``max_k`` states
    the ceiling on ``k``, which is otherwise invisible: the tool clamps a larger ``k``
    before the query runs, so "ask for more" was an invitation the tool did not honour.
    """
    if not columns:
        returns = "Returns the best-matching rows ordered by a 'score' column, highest first"
    else:
        named = list(columns)
        listed = named[0] if len(named) == 1 else f"{', '.join(named[:-1])} and {named[-1]}"
        returns = f"Each hit carries {listed}, ordered by a 'score' column, highest first"
    ceiling = "" if max_k is None else f", up to a maximum of {max_k}"
    return (
        f"Find rows of {table} whose '{column}' text is relevant to a natural-language "
        "query, ranked by relevance. LIKE and ILIKE only test for a literal substring you "
        "already know, so they are a filter, not a way to find relevant rows.\n"
        f"{returns}; "
        f"scores are comparable within one result set but not across queries. Ask for "
        f"more with 'k' when you need a wider net{ceiling}.\n"
        "Use this to list or inspect the matches themselves. When the answer aggregates "
        "over the matches rather than listing them, rank inside SQL instead — that keeps "
        "the whole cohort in the query, where carrying values back as literals caps it at "
        "this tool's row limit."
    )


def make_hotdata_search_tool(
    client: HotdataClient,
    *,
    table: str,
    column: str,
    columns: Sequence[str] | None = None,
    key_column: str | None = DEFAULT_KEY_COLUMN,
    k: int = DEFAULT_SEARCH_LIMIT,
    name: str = DEFAULT_SEARCH_TOOL_NAME,
    description: str | None = None,
    max_rows: int = 100,
    database_id: str | ManagedDatabase | None = None,
) -> StructuredTool:
    """Return a LangChain tool that full-text searches one indexed column.

    The corpus is pinned here rather than chosen by the model: nothing in the tool
    surface lets an agent discover which columns carry a BM25 index, and the engine
    errors outright when one is missing. The agent supplies only ``query`` and an
    optional ``k``.

    ``database_id`` scopes the search to one managed database, by id and never by name;
    it is resolved once here. Pass an already-resolved ``ManagedDatabase`` to skip the
    lookup.

    Register the factory more than once, with distinct ``name`` and ``description``
    values, to expose several searchable corpora; the agent then routes on the
    descriptions.

    A ``k`` the model supplies is clamped to ``max_rows``, since anything above it would
    have the engine rank and ship rows that are then discarded before the model sees
    them. The caller's own ``k`` is trusted and left alone. A clamped call says so in
    ``metadata.client_warning``: the clamp runs before the query, so the result carries
    no other trace of it.

    ``key_column`` is added to the default projection when the table has such a column,
    so a hit carries the value that joins it back to the rest of the table. Returning the
    searched column alone quietly disables that join, which is this integration's central
    claim — that a retrieved row is an ordinary SQL value. The column is looked up once
    here rather than assumed, and dropped when the table does not have it. It is ignored
    when ``columns`` is given: a caller naming the projection has already chosen.
    """
    _validate_table_ref(table)
    validate_identifier(column, label="column")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows}")
    if key_column is not None:
        validate_identifier(key_column, label="key_column")
    if columns is not None:
        _projection(column, columns)
    default_k = k
    database = resolve_database_by_id(client, database_id) if database_id is not None else None
    if columns is not None:
        projection = list(columns)
    else:
        projection = _default_columns(
            client,
            table=table,
            column=column,
            key_column=key_column,
            database=database,
        )

    def hotdata_search_text(query: str, k: int | None = None) -> str:
        """Search indexed text by relevance and return ranked rows as JSON.

        Args:
            query: what to look for, in natural language; whole phrases work better
                than single keywords.
            k: how many ranked rows to return. Values above this tool's row limit are
                reduced to it before the search runs, so a larger k does not widen the
                cohort.
        """
        requested = None if k is None else max(1, k)
        clamped = clamp_warning(requested=requested, ceiling=max_rows) if requested else None
        return bm25_search_json(
            client,
            table=table,
            column=column,
            query=query,
            k=default_k if requested is None else min(requested, max_rows),
            columns=projection,
            max_rows=max_rows,
            database=database,
            warnings=[clamped] if clamped else (),
        )

    return StructuredTool.from_function(
        func=hotdata_search_text,
        name=name,
        description=description
        or default_search_description(table, column, columns=projection, max_k=max_rows),
        parse_docstring=True,
    )


def _default_columns(
    client: HotdataClient,
    *,
    table: str,
    column: str,
    key_column: str | None,
    database: ManagedDatabase | None,
) -> list[str]:
    """Return the projection to use when the caller named none.

    The searched column, preceded by ``key_column`` when the table actually has one.
    Falls back to the searched column alone when the lookup fails, so a schema query is
    never the reason tool construction fails.
    """
    if key_column is None or key_column == column:
        return [column]
    catalog, schema, name = table.split(".")
    sql = (
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_catalog = {quote_literal(catalog)} "
        f"AND table_schema = {quote_literal(schema)} "
        f"AND table_name = {quote_literal(name)} "
        f"AND column_name = {quote_literal(key_column)}"
    )
    try:
        found = bool(client.execute_sql(sql, database=query_scope(database)).rows)
    except Exception:
        logger.warning(
            "could not check %s for a %r column; searching %s alone, so hits will carry "
            "no join key",
            table,
            key_column,
            column,
            exc_info=True,
        )
        return [column]
    if not found:
        logger.debug("%s has no %r column; hits will carry %s alone", table, key_column, column)
        return [column]
    return [key_column, column]
