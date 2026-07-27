"""Full-text (BM25) search helpers and tools for LangChain agents."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from hotdata_framework import HotdataClient
from langchain_core.tools import StructuredTool

#: Column the engine appends to every ``bm25_search`` result, holding the BM25 relevance score.
SCORE_COLUMN = "score"

#: Default number of ranked hits requested when a caller does not specify one.
DEFAULT_SEARCH_LIMIT = 5

DEFAULT_SEARCH_TOOL_NAME = "hotdata_search_text"

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
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


def _validate_identifier(value: str, *, label: str) -> str:
    """Return ``value`` if it is a bare SQL identifier, else raise."""
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{label} must be a bare SQL identifier, got {value!r}")
    return value


def _quote_literal(value: str) -> str:
    """Return ``value`` as a single-quoted SQL string literal with quotes doubled."""
    if "\x00" in value:
        raise ValueError("search text may not contain null bytes")
    return "'" + value.replace("'", "''") + "'"


def _projection(column: str, columns: Sequence[str] | None) -> list[str]:
    selected = list(columns) if columns is not None else [column]
    if not selected:
        raise ValueError("columns must not be empty")
    for name in selected:
        _validate_identifier(name, label="column")
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
    _validate_identifier(column, label="column")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    projection = _projection(column, columns)
    return (
        f"SELECT {', '.join(projection)} "
        f"FROM bm25_search("
        f"{_quote_literal(table)}, {_quote_literal(column)}, {_quote_literal(query)}, {k}) "
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
    database: str | None = None,
) -> str:
    """Run a BM25 search and return ``{"metadata": ..., "rows": [...]}`` as JSON.

    Mirrors the envelope :func:`hotdata_langchain.tools.execute_sql_json` returns, so an
    agent sees one result shape across every Hotdata tool. Rows arrive ranked by
    ``score`` descending.
    """
    sql = bm25_search_sql(table=table, column=column, query=query, k=k, columns=columns)
    result = client.execute_sql(sql, database=database)
    payload = {
        "metadata": result.metadata_dict(),
        "rows": result.to_records(max_rows=max_rows),
    }
    return json.dumps(payload, indent=2)


def default_search_description(table: str, column: str) -> str:
    """Return the agent-facing tool description used when no override is given.

    Describes the capability ("find rows whose text is relevant") rather than the index
    behind it, so the contract the model is given survives the retrieval strategy
    changing underneath it.
    """
    return (
        f"Find rows of {table} whose '{column}' text is relevant to a natural-language "
        "query. This is the only way to match on what the text says — SQL cannot rank "
        "rows by textual relevance.\n"
        "Returns the best-matching rows ordered by a 'score' column, highest first; "
        "scores are comparable within one result set but not across queries. Ask for "
        "more with 'k' when you need a wider net.\n"
        "Use it to identify which rows are relevant, then take the values you need from "
        "the results into the SQL tool for filters, joins and aggregates."
    )


def make_hotdata_search_tool(
    client: HotdataClient,
    *,
    table: str,
    column: str,
    columns: Sequence[str] | None = None,
    k: int = DEFAULT_SEARCH_LIMIT,
    name: str = DEFAULT_SEARCH_TOOL_NAME,
    description: str | None = None,
    max_rows: int = 100,
    database: str | None = None,
) -> StructuredTool:
    """Return a LangChain tool that full-text searches one indexed column.

    The corpus is pinned here rather than chosen by the model: nothing in the tool
    surface lets an agent discover which columns carry a BM25 index, and the engine
    errors outright when one is missing. The agent supplies only ``query`` and an
    optional ``k``.

    Register the factory more than once, with distinct ``name`` and ``description``
    values, to expose several searchable corpora; the agent then routes on the
    descriptions.
    """
    _validate_table_ref(table)
    _validate_identifier(column, label="column")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if columns is not None:
        _projection(column, columns)
    default_k = k

    def hotdata_search_text(query: str, k: int | None = None) -> str:
        """Search indexed text by relevance and return ranked rows as JSON."""
        return bm25_search_json(
            client,
            table=table,
            column=column,
            query=query,
            k=default_k if k is None else k,
            columns=columns,
            max_rows=max_rows,
            database=database,
        )

    return StructuredTool.from_function(
        func=hotdata_search_text,
        name=name,
        description=description or default_search_description(table, column),
    )
