"""Schema introspection so an agent can learn what it is allowed to query."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from hotdata_framework import HotdataClient, ManagedDatabase
from langchain_core.tools import StructuredTool

from hotdata_langchain._sql import quote_identifier, validate_identifier
from hotdata_langchain.databases import query_scope, resolve_database_by_id
from hotdata_langchain.indexes import (
    SearchIndex,
    generated_vector_columns,
    list_search_indexes,
    search_nouns_by_column,
)

logger = logging.getLogger(__name__)

DEFAULT_DESCRIBE_TOOL_NAME = "hotdata_describe_tables"

#: Catalog a managed table answers to. An attached source's tables answer to the
#: attachment alias instead, so a reference naming any other catalog cannot be a managed
#: table on this database — which is what makes the declared-but-empty check answerable.
MANAGED_CATALOG = "default"

#: Cap on columns returned for a single table, so one wide table cannot flood the context.
DEFAULT_MAX_COLUMNS = 200


def _split_table(table: str) -> tuple[str | None, str | None, str]:
    """Split a table reference into ``(catalog, schema, name)``, validating each part.

    All three forms are accepted — ``table``, ``schema.table`` and
    ``catalog.schema.table`` — because the SQL tool's description tells the model to
    address tables with all three parts, and both descriptions reach it in one prompt.
    Rejecting the form the other tool teaches made the model's first correct instinct an
    error it had to recover from, which was measured costing a call on a real run.

    The database is already scoped, so a catalog is redundant here rather than wrong; it
    is used as a filter when given, which is what makes it meaningful on a database
    exposing more than one.

    Parts come back lowercased, because that is the form every caller needs. The engine
    lowercases identifiers when it stores them, so ``information_schema`` holds the lower
    form and an exact filter on ``Public`` matches nothing — while ``FROM
    DEFAULT.public.listings`` resolves fine, a bare reference in SQL being
    case-insensitive. Without this, describing a table was the one place where the case
    the model typed decided whether it got an answer.
    """
    parts = table.split(".")
    if len(parts) > 3:
        raise ValueError(
            f"table must be 'table', 'schema.table' or 'catalog.schema.table', got {table!r}"
        )
    for part in parts:
        validate_identifier(part, label="table")
    lowered = [part.lower() for part in parts]
    if len(lowered) == 3:
        return (lowered[0], lowered[1], lowered[2])
    if len(lowered) == 2:
        return (None, lowered[0], lowered[1])
    return (None, None, lowered[0])


def table_overview_sql() -> str:
    """Return SQL listing every table in the scoped database with its column count."""
    return (
        "SELECT table_schema, table_name, COUNT(column_name) AS column_count "
        "FROM information_schema.columns "
        "GROUP BY table_schema, table_name "
        "ORDER BY table_schema, table_name"
    )


def table_columns_sql(table: str, *, limit: int = DEFAULT_MAX_COLUMNS) -> str:
    """Return SQL listing one table's columns and types, in declaration order.

    ``table`` may name a catalog, a schema, both or neither; each part given narrows the
    lookup. A catalog matters on a database exposing more than one — an attached source's
    tables answer to its alias rather than to ``default`` — where the same schema and
    table name can exist under both.
    """
    catalog, schema, name = _split_table(table)
    where = f"WHERE table_name = '{name}'"
    if schema is not None:
        where += f" AND table_schema = '{schema}'"
    if catalog is not None:
        where += f" AND table_catalog = '{catalog}'"
    return (
        f"SELECT table_schema, table_name, column_name, data_type "
        f"FROM information_schema.columns {where} "
        f"ORDER BY table_schema, table_name, ordinal_position "
        f"LIMIT {limit}"
    )


def column_stats_sql(table: str, columns: Sequence[str]) -> str:
    """Return SQL counting the table's rows and each column's non-NULL values.

    One aggregate query over every column, so learning which columns hold data costs a
    single extra round trip rather than one per column. ``COUNT(*)`` is safe here even
    on the tables that reject a bare ``COUNT(*)``: the projection names real columns
    alongside it, which is what those tables require.

    Column names are quoted, since they come from the table rather than from a caller.
    Most words a caller might worry about need no quoting here — ``order``, ``group``,
    ``table``, ``end`` and ``start`` all parse unquoted (verified 2026-08-17) — but
    ``all`` does not, and one such column would otherwise fail the whole aggregate and
    cost every column its count.
    """
    aggregates = ", ".join(
        f"COUNT({quote_identifier(name)}) AS n{index}" for index, name in enumerate(columns)
    )
    return f"SELECT COUNT(*) AS row_count, {aggregates} FROM {table}"


def _quotable(name: str) -> bool:
    """Return whether ``name`` can be written into SQL as a quoted identifier."""
    try:
        quote_identifier(name)
    except ValueError:
        return False
    return True


def _table_indexes(
    client: HotdataClient,
    *,
    schema: str,
    name: str,
    database: ManagedDatabase | None,
) -> list[SearchIndex]:
    """Return the search indexes on one table, or none when the scope cannot answer.

    Indexes belong to a connection, so an unscoped call has nothing to ask. Schema and
    table come from the rows ``information_schema`` returned rather than from re-parsing
    the reference, so a name this package would not have parsed still resolves.
    """
    if database is None:
        logger.debug("no database scope for %s.%s; not reporting searchable columns", schema, name)
        return []
    return list_search_indexes(client, table=name, schema=schema, database=database)


def _column_stats(
    client: HotdataClient,
    *,
    table: str,
    columns: Sequence[str],
    database: ManagedDatabase | None,
) -> tuple[int, dict[str, int]] | None:
    """Return ``(row_count, non_null_by_column)``, or ``None`` when the query fails.

    Fails open in both directions: a table whose stats cannot be read is still described
    by its schema, which is what the tool did before the counts existed, and a column
    whose name cannot be quoted — one carrying a double quote or a null byte — is skipped
    rather than taking the whole description down with it. Names come from the table, not
    from the caller, so an unusual one is a data property rather than a mistake to raise
    on.
    """
    countable = [name for name in columns if _quotable(name)]
    if len(countable) != len(columns):
        logger.debug("%s has columns that cannot be counted by name", table)
    if not countable:
        return None
    try:
        result = client.execute_sql(
            column_stats_sql(table, countable), database=query_scope(database)
        )
        values = result.rows[0]
        if len(values) != len(countable) + 1:
            raise ValueError(f"expected {len(countable) + 1} counts, got {len(values)}")
        return int(values[0]), {
            name: int(value) for name, value in zip(countable, values[1:], strict=True)
        }
    except Exception:
        logger.warning("could not count populated columns for %s", table, exc_info=True)
        return None


def _declared_but_empty(
    client: HotdataClient,
    *,
    table: str,
    database: ManagedDatabase | None,
) -> bool:
    """Return whether ``table`` is declared on ``database`` but holds no data.

    A declared table that has never been loaded reports zero rows in
    ``information_schema.columns``, so its schema lookup is indistinguishable from a
    missing table. The managed-table listing is what tells the two apart.

    The schema is filtered by the listing call rather than by reading it back off each
    record, matching how ``HotdataVectorStore`` asks the same question.

    A catalog other than :data:`MANAGED_CATALOG` answers ``False`` outright. The listing
    takes no catalog filter, so dropping the part would let ``wrong.public.listings`` match
    the managed ``public.listings`` and report a table that does not exist here as one
    awaiting a load — a false claim whose remedy points at work that cannot be done. The
    column lookup does filter on the catalog, so without this the two checks disagree.
    """
    if database is None:
        return False
    # `_split_table` lowercases, so this compares the stored form rather than what the
    # model happened to type.
    catalog, schema, name = _split_table(table)
    if catalog is not None and catalog != MANAGED_CATALOG:
        return False
    try:
        return any(
            entry.table == name for entry in client.list_managed_tables(database, schema=schema)
        )
    except Exception:
        logger.debug("could not list managed tables for %s", database.id, exc_info=True)
        return False


def describe_tables_json(
    client: HotdataClient,
    *,
    table: str | None = None,
    database_id: str | ManagedDatabase | None = None,
    max_columns: int = DEFAULT_MAX_COLUMNS,
    column_stats: bool = True,
    search_capabilities: bool = True,
) -> str:
    """Describe the scoped database's tables, or one table's columns, as JSON.

    Without ``table`` this returns every table with its column count — a cheap map of
    what exists. With ``table`` it returns that table's columns and types in
    declaration order, capped at ``max_columns`` so a wide table cannot flood the
    model's context; the payload says so when the cap truncated the list.

    ``column_stats`` adds the table's ``row_count`` and each column's ``non_null``
    count, from one extra aggregate query. Types alone say a column exists, not whether
    it holds anything: an agent was measured recommending an analysis of a column that
    is NULL on all 7,535 rows, and a table of 63 columns where most are populated only
    by the instrumentation that writes them presents every one as equally available.
    Turn it off to describe a table without scanning it.

    ``search_capabilities`` annotates each column with ``searchable_by`` — what it can be
    ranked by, as ``text relevance`` or ``meaning`` — at the cost of one control-plane
    call per described table. Indexes are invisible to SQL, so this is the only way an
    agent can find out which column to search rather than guess. It also drops the vector
    columns a provider-backed index generated: those are real columns that no caller
    wrote, and describing a 1536-wide float list as ordinary data invites queries against
    it. Only indexes the engine reports as ready are named, because a search against one
    still building fails after the model has committed to the route.

    A table that is declared on the database but has never been loaded reports no
    columns at all, which reads as a missing table. It is reported as declared and
    empty instead.

    ``database_id`` takes a database id or an already-resolved ``ManagedDatabase``. A
    name is not accepted: names are display labels and are not unique, so resolution is by
    id only. Passing a resolved record skips the per-call lookup an id costs.

    Raises ``ValueError`` for a non-positive ``max_columns``.
    """
    if max_columns < 1:
        raise ValueError(f"max_columns must be >= 1, got {max_columns}")
    database = resolve_database_by_id(client, database_id) if database_id is not None else None
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
        if _declared_but_empty(client, table=table, database=database):
            return json.dumps(
                {
                    "table": table,
                    "columns": [],
                    "row_count": 0,
                    "note": (
                        f"{table} is declared on this database but has no data yet, so it "
                        f"has no columns and every query against it fails. Load data into "
                        f"it before querying."
                    ),
                },
                indent=2,
            )
        return json.dumps(
            {"table": table, "columns": [], "error": f"no table named {table!r} in this database"},
            indent=2,
        )
    truncated = len(records) > max_columns
    records = records[:max_columns]
    qualified = f"{records[0]['table_schema']}.{records[0]['table_name']}"
    indexes = (
        _table_indexes(
            client,
            schema=str(records[0]["table_schema"]),
            name=str(records[0]["table_name"]),
            database=database,
        )
        if search_capabilities
        else []
    )
    generated = set(generated_vector_columns(indexes))
    records = [r for r in records if str(r["column_name"]) not in generated]
    names = [str(r["column_name"]) for r in records]
    columns: list[dict[str, object]] = [
        {"name": name, "type": r["data_type"]} for name, r in zip(names, records, strict=True)
    ]
    searchable = search_nouns_by_column(indexes)
    annotated = False
    for entry in columns:
        nouns = searchable.get(str(entry["name"]))
        if nouns:
            entry["searchable_by"] = nouns
            annotated = True
    payload: dict[str, object] = {"table": qualified, "columns": columns}
    if annotated:
        payload["search"] = (
            "searchable_by names what a column can be ranked by: 'text relevance' matches "
            "the words a value uses, 'meaning' matches what it is about. A column without "
            "it can still be filtered in SQL, but a substring filter is not a search — it "
            "matches only the literal characters given."
        )
    stats = (
        _column_stats(client, table=qualified, columns=names, database=database)
        if column_stats
        else None
    )
    if stats is not None:
        row_count, non_null = stats
        payload["row_count"] = row_count
        for entry in columns:
            count = non_null.get(str(entry["name"]))
            if count is not None:
                entry["non_null"] = count
        payload["column_stats"] = (
            "non_null is how many of the table's rows hold a value in that column; a "
            "column at 0 is empty and nothing can be computed from it."
        )
    if truncated:
        payload["truncated_at"] = max_columns
    return json.dumps(payload, indent=2)


def default_describe_description(
    *,
    column_stats: bool = True,
    search_capabilities: bool = True,
    catalogs: Sequence[str] | None = None,
) -> str:
    """Return the agent-facing description for the schema tool.

    ``column_stats`` adds the sentence about ``non_null``. It is stated in the
    description as well as in the payload because the point of the counts is to change
    what the model plans before it reads any rows, and a column it never asked about is
    a column whose emptiness it never sees.

    ``search_capabilities`` adds the sentence about ``searchable_by``, for the same reason
    ``column_stats`` earns one: which column is searchable cannot be read in SQL, so a
    model that does not know this tool reports it has no way to find out, and picks a
    column to search by guessing.

    ``catalogs`` names the catalogs the tools are scoped to, and the worked example uses
    one only when there is exactly one to use. There is no correct constant to fall back
    on: an instant database answers to ``default`` and an attached source answers to its
    attachment alias, so a hardcoded ``default`` would put this description at odds with
    the SQL tool's — which resolves the catalog per database — in the one prompt they both
    reach. It would also send a model following the example to a reference the catalog
    filter then finds nothing for, reported as a missing table rather than as the loud
    format error that used to name the accepted forms.
    """
    populated = (
        "Each column also reports 'non_null', how many rows actually hold a value: a "
        "column can exist, be correctly typed, and still be empty on every row, so "
        "check it before building an analysis on that column.\n"
        if column_stats
        else ""
    )
    searchable = (
        "A column that can be searched also reports 'searchable_by': 'text relevance' "
        "matches the words a value uses, 'meaning' matches what it is about. Check it "
        "before searching — which column is searchable cannot be worked out from SQL.\n"
        if search_capabilities
        else ""
    )
    known = list(catalogs or ())
    full = f"'{known[0]}.public.listings'" if len(known) == 1 else "the full 'catalog.schema.table'"
    return (
        "Discover what data is available before writing a query. Called with no "
        "arguments it lists every table with how many columns it has; called with a "
        f"table name ('listings', 'public.listings' or {full}) it "
        "returns that table's columns and their types.\n"
        f"{populated}"
        f"{searchable}"
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
    column_stats: bool = True,
    search_capabilities: bool = True,
    catalogs: Sequence[str] | None = None,
) -> StructuredTool:
    """Return a LangChain tool that reports the scoped database's tables and columns.

    ``database_id`` scopes the introspection to one instant database, by id and never by
    name; it is resolved once here. Pass an already-resolved ``ManagedDatabase`` to skip
    the lookup.

    ``column_stats`` (on by default) reports each column's non-NULL count alongside its
    type, at the cost of one aggregate query per described table. Turn it off where
    describing a table must not scan it.

    ``search_capabilities`` (on by default) reports what each column can be searched by,
    at the cost of one control-plane call per described table. Turn it off to describe a
    table without asking what is indexed on it.

    Fails fast on a non-positive ``max_columns`` rather than at first invocation.
    """
    if max_columns < 1:
        raise ValueError(f"max_columns must be >= 1, got {max_columns}")
    database = resolve_database_by_id(client, database_id) if database_id is not None else None

    def hotdata_describe_tables(table: str | None = None) -> str:
        """List the tables in the database, or one table's columns and types.

        Args:
            table: the table to describe, as 'listings', 'public.listings' or the full
                'catalog.schema.table'. Omit it to list every table in the database
                instead.
        """
        return describe_tables_json(
            client,
            table=table,
            database_id=database,
            max_columns=max_columns,
            column_stats=column_stats,
            search_capabilities=search_capabilities,
        )

    return StructuredTool.from_function(
        func=hotdata_describe_tables,
        name=name,
        description=description
        or default_describe_description(
            column_stats=column_stats,
            search_capabilities=search_capabilities,
            catalogs=catalogs,
        ),
        parse_docstring=True,
    )
