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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from hotdata.api.indexes_api import IndexesApi
from hotdata_framework import HotdataClient, ManagedDatabase

logger = logging.getLogger(__name__)

SearchKind = Literal["text", "semantic"]

#: Search kinds a column can support, named for the capability.
TEXT: Final[SearchKind] = "text"
SEMANTIC: Final[SearchKind] = "semantic"

#: How each kind reads in a sentence written for a model.
CAPABILITY_PHRASES: dict[str, str] = {
    TEXT: "searchable by text relevance",
    SEMANTIC: "searchable by meaning",
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


def capabilities_by_column(indexes: Sequence[SearchIndex]) -> dict[str, list[str]]:
    """Return each column's search capabilities, as phrases, keyed by column name.

    A column carries one in every configuration the engine allows today, and the list is
    a list because that is a property of the engine rather than of this function: BM25
    indexes a text column and a plain vector index a vector column, so the two land on
    different columns of the same table, and a provider-backed index cannot share a table
    with any other index at all.
    """
    by_column: dict[str, list[str]] = {}
    for index in indexes:
        phrases = by_column.setdefault(index.column, [])
        if index.capability not in phrases:
            phrases.append(index.capability)
    return by_column


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
