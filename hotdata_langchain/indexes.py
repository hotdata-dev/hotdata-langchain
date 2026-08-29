"""Which columns of a table are searchable, and by what kind of search.

Indexes are invisible to SQL — there is no ``pg_indexes`` and no
``information_schema.indexes`` — so the control plane is the only place that can answer
this. Everything here is read-only introspection over ``IndexesApi.list_indexes``.

The vocabulary is the capability, not the mechanism: a column is searchable *by text
relevance* or *by meaning*, never "has a BM25 index". Callers that need the mechanism can
read it off :class:`SearchIndex`, but the words that reach a model come from the capability.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from hotdata.api.indexes_api import IndexesApi
from hotdata_framework import HotdataClient, ManagedDatabase

logger = logging.getLogger(__name__)

SearchKind = Literal["text", "semantic"]

#: Search kinds a column can support, named for the capability.
TEXT: Final[SearchKind] = "text"
SEMANTIC: Final[SearchKind] = "semantic"

#: What each kind is called where the capability is named on its own — a payload field,
#: a list of what a column supports.
SEARCH_NOUNS: dict[str, str] = {
    TEXT: "text relevance",
    SEMANTIC: "meaning",
}

#: How each kind reads in a sentence written for a model.
CAPABILITY_PHRASES: dict[str, str] = {
    kind: f"searchable by {noun}" for kind, noun in SEARCH_NOUNS.items()
}

_READY = "ready"


def _wire_value(value: Any) -> str:
    """Render an API field that may be a ``str``-mixin enum as its wire string."""
    return str(getattr(value, "value", value))


@dataclass(frozen=True)
class SearchIndex:
    """One index, described by what it lets a caller search rather than how.

    ``column`` is the column a query names, which is not always the column the index is
    built over. A provider-backed vector index embeds a *text* column into a generated
    vector column and is queried through the text one; the vector column it produced is
    ``vector_column``, and it appears in ``information_schema`` as a real column.

    ``embeds_query`` is the load-bearing field. When it is true the engine embeds the
    query string itself, so a semantic query is written in SQL with a text literal and no
    embedding model is needed on this side. When it is false the caller has to supply the
    query as a vector, which means holding an embedding model and using the same one the
    column was written with.
    """

    column: str
    kind: SearchKind
    index_type: str
    ready: bool
    metric: str | None = None
    vector_column: str | None = None
    embeds_query: bool = False
    index_name: str | None = None

    @property
    def capability(self) -> str:
        """Return the phrase describing this index for a model-facing sentence."""
        return CAPABILITY_PHRASES[self.kind]

    @property
    def search_noun(self) -> str:
        """Return what this index makes the column searchable by, named on its own."""
        return SEARCH_NOUNS[self.kind]


def _search_index(index: Any) -> SearchIndex | None:
    """Return ``index`` as a :class:`SearchIndex`, or ``None`` if it serves no search.

    A sorted index is skipped: it is reached by the planner substituting sorted parquet
    for a matching filter, never by a caller naming it, so it is not a search capability
    an agent can be told about.
    """
    # Read defensively: this package supports a range of SDK versions, and a field that
    # an older one does not send should cost the caller that index rather than the whole
    # listing. `source_column` in particular is what separates the two kinds of vector
    # index, and its absence means "not provider-backed", which is the older behaviour.
    index_type = _wire_value(getattr(index, "index_type", "")).lower()
    columns = list(getattr(index, "columns", None) or [])
    source = getattr(index, "source_column", None)
    metric = getattr(index, "metric", None)
    ready = _wire_value(getattr(index, "status", "")).lower() == _READY
    name = getattr(index, "index_name", None)

    if index_type == "bm25":
        if not columns:
            return None
        return SearchIndex(
            column=columns[0],
            kind=TEXT,
            index_type=index_type,
            ready=ready,
            index_name=name,
        )
    if index_type == "vector":
        # A provider-backed index reports the text column it embeds as `source_column`
        # and the generated vector column in `columns`; a plain one reports only the
        # vector column it was built over, and has no source.
        if source:
            return SearchIndex(
                column=source,
                kind=SEMANTIC,
                index_type=index_type,
                ready=ready,
                metric=_wire_value(metric).lower() if metric else None,
                vector_column=columns[0] if columns else None,
                embeds_query=True,
                index_name=name,
            )
        if not columns:
            return None
        return SearchIndex(
            column=columns[0],
            kind=SEMANTIC,
            index_type=index_type,
            ready=ready,
            metric=_wire_value(metric).lower() if metric else None,
            vector_column=columns[0],
            embeds_query=False,
            index_name=name,
        )
    return None


def list_search_indexes(
    client: HotdataClient,
    *,
    table: str,
    schema: str,
    database: ManagedDatabase,
    ready_only: bool = True,
) -> list[SearchIndex]:
    """Return the search indexes on one table, newest kinds and all.

    ``ready_only`` drops indexes the server has accepted but not finished building.
    Reporting one as usable is worse than not reporting it: a search against a pending
    index fails, and it fails at the point where the model has already committed to the
    route.

    Fails open. A workspace whose control plane is unreachable, or a caller whose token
    cannot list indexes, gets an empty list and a logged warning rather than an exception,
    because every caller here is deciding how to *describe* a table and none of them
    should fail to build a tool over it.
    """
    try:
        listed = (
            IndexesApi(client.api)
            .list_indexes(database.default_connection_id, schema, table)
            .indexes
        )
    except Exception:
        logger.warning(
            "could not list indexes for %s.%s; treating it as having no searchable "
            "columns, so searches over it will not be offered",
            schema,
            table,
            exc_info=True,
        )
        return []
    found: list[SearchIndex] = []
    for index in listed or []:
        described = _search_index(index)
        if described is None:
            continue
        if ready_only and not described.ready:
            logger.debug(
                "index %s on %s.%s is not ready; not offering it",
                described.index_name,
                schema,
                table,
            )
            continue
        found.append(described)
    return found


def indexes_for_column(indexes: Sequence[SearchIndex], column: str) -> list[SearchIndex]:
    """Return the indexes among ``indexes`` that make ``column`` searchable."""
    return [index for index in indexes if index.column == column]


def _by_column(
    indexes: Sequence[SearchIndex], describe: Callable[[SearchIndex], str]
) -> dict[str, list[str]]:
    by_column: dict[str, list[str]] = {}
    for index in indexes:
        entries = by_column.setdefault(index.column, [])
        described = describe(index)
        if described not in entries:
            entries.append(described)
    return by_column


def capabilities_by_column(indexes: Sequence[SearchIndex]) -> dict[str, list[str]]:
    """Return each column's search capabilities, as phrases, keyed by column name.

    A column carries one in every configuration the engine allows today, and the list is
    a list because that is a property of the engine rather than of this function: BM25
    indexes a text column and a plain vector index a vector column, so the two land on
    different columns of the same table, and a provider-backed index cannot share a table
    with any other index at all.
    """
    return _by_column(indexes, lambda index: index.capability)


def search_nouns_by_column(indexes: Sequence[SearchIndex]) -> dict[str, list[str]]:
    """Return what each column is searchable by, keyed by column name.

    The same grouping as :func:`capabilities_by_column`, naming the capability on its own
    rather than as a sentence fragment, for a payload field that already says the column
    is searchable.
    """
    return _by_column(indexes, lambda index: index.search_noun)


def generated_vector_columns(indexes: Sequence[SearchIndex]) -> Iterator[str]:
    """Yield the vector columns that provider-backed indexes generated.

    These are real columns in ``information_schema`` that no caller wrote: building a
    provider-backed index over ``content`` materialises ``content_embedding`` beside it.
    Describing them to a model as ordinary data invites queries against a 1536-wide float
    list, so callers that report schemas filter them out.
    """
    for index in indexes:
        if index.embeds_query and index.vector_column:
            yield index.vector_column


@dataclass(frozen=True)
class SearchableColumn:
    """One column a ready index covers, carrying the table reference to name it by.

    :class:`SearchIndex` describes an index within a table already known to the caller.
    This pairs one with the three-part reference a query has to write, which is what a
    tool description needs and what the index record does not carry.
    """

    table: str
    index: SearchIndex

    @property
    def column(self) -> str:
        return self.index.column

    @property
    def kind(self) -> SearchKind:
        return self.index.kind

    @property
    def function(self) -> str:
        """Return the table function this column is searched through."""
        return "vector_search" if self.index.kind == SEMANTIC else "bm25_search"

    @property
    def composable(self) -> bool:
        """Report whether a query against this column can be written in SQL.

        False for a plain vector index, whose query has to arrive as a vector.
        """
        return self.index.kind == TEXT or self.index.embeds_query


def verify_searchable_columns(
    client: HotdataClient,
    *,
    columns: Sequence[tuple[str, str]],
    database: ManagedDatabase | None,
) -> list[SearchableColumn]:
    """Return the declared ``(table, column)`` pairs a ready index actually covers.

    Declared rather than discovered, and then confirmed rather than trusted. Naming a
    column a model can search is a claim about the database, and this package states one
    only after reading it back — the same stance :func:`query_catalogs` takes towards the
    catalog name. A pair no index covers is dropped with a warning rather than named,
    because BM25 has no brute-force fallback and a search against an unindexed column is
    a hard error at the point the model has already committed to the route.

    One control-plane call per distinct table, not per declared column. Order is the
    caller's, and it is preserved: a description that names several columns leads with
    the first, which is the one a model was measured reaching for most.

    Returns an empty list without ``database``, which is the scope every index listing
    needs. Duplicated pairs are named once.
    """
    if database is None:
        return []
    listed: dict[str, list[SearchIndex]] = {}
    found: list[SearchableColumn] = []
    seen: set[tuple[str, str]] = set()
    for table, column in columns:
        if (table, column) in seen:
            continue
        seen.add((table, column))
        parts = table.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"a searchable column's table must be written catalog.schema.table, got {table!r}"
            )
        if table not in listed:
            listed[table] = list_search_indexes(
                client, table=parts[2], schema=parts[1], database=database
            )
        covering = indexes_for_column(listed[table], column)
        if not covering:
            logger.warning(
                "no ready search index was found covering %r on %s; not naming it as searchable",
                column,
                table,
            )
            continue
        found.append(SearchableColumn(table, covering[0]))
    return found


def fusable_vector_indexes(indexes: Sequence[SearchIndex]) -> list[SearchIndex]:
    """Return the plain vector indexes among ``indexes``, in listing order.

    Plain because those are the only ones a text search can be fused with. A
    provider-backed index cannot coexist with the BM25 index the other half of a fusion
    needs, so a table carrying one has nothing to fuse.

    The engine records no link between a plain index's vector column and the text it was
    derived from — ``source_column`` is ``None`` — so pairing one with a text column is
    the caller's statement, not something that can be read back. This returns the
    candidates so a caller can pair when there is exactly one and ask when there are more.
    """
    return [index for index in indexes if index.kind == SEMANTIC and not index.embeds_query]
