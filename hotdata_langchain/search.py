"""Search helpers and tools for LangChain agents, by text relevance and by meaning."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from hotdata_framework import HotdataClient, ManagedDatabase
from langchain_core.embeddings import Embeddings
from langchain_core.tools import StructuredTool

from hotdata_langchain._sql import (
    DISTANCE_FUNCTIONS,
    DistanceMetric,
    quote_literal,
    validate_identifier,
    vector_literal,
)
from hotdata_langchain.databases import query_scope, resolve_database_by_id
from hotdata_langchain.indexes import (
    SEMANTIC,
    TEXT,
    SearchIndex,
    SearchKind,
    indexes_for_column,
    list_search_indexes,
)
from hotdata_langchain.results import SEARCH_REMEDY, result_json, search_remedy

logger = logging.getLogger(__name__)

#: Column the engine appends to every ``bm25_search`` result, holding the BM25 relevance score.
SCORE_COLUMN = "score"

#: Column ``vector_search`` appends, holding the distance from the query. Lower is nearer,
#: the opposite direction to ``score``. The leading underscore is the engine's, not ours, and
#: results keep it: an earlier version renamed it to ``distance`` because the underscore reads
#: as private, which left one value with two names across the two descriptions the model reads
#: — and only the engine's name works in SQL the model writes itself.
DISTANCE_COLUMN = "_distance"

#: Which retrieval route a search tool takes. ``auto`` reads it off the column's indexes.
SearchStrategy = Literal["auto", "text", "semantic"]

#: Default number of ranked hits requested when a caller does not specify one.
DEFAULT_SEARCH_LIMIT = 5

#: Column added to the default projection when the searched table has one, so a hit
#: carries the value that joins it back to the rest of the table.
DEFAULT_KEY_COLUMN = "id"

DEFAULT_SEARCH_TOOL_NAME = "hotdata_search_text"

#: Default name when the route ranks by meaning. The names differ because the name reaches
#: the model: calling a semantic search "search_text" tells it the tool matches wording,
#: which is the one thing this route does not do.
DEFAULT_SEMANTIC_TOOL_NAME = "hotdata_search_semantic"

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


def _semantic_projection(
    column: str,
    columns: Sequence[str] | None,
    *,
    include_column: bool,
) -> list[str]:
    """Return the projection for a semantic search, with the distance column last.

    ``include_column`` is false when the searched column holds the vectors themselves.
    Selecting it would ship a 1536-wide float list per row to the model, and it also
    forfeits the index rewrite, so a vector column is never projected even when the
    caller lists it.
    """
    if columns is not None:
        selected = [name for name in columns if include_column or name != column]
    else:
        selected = [column] if include_column else []
    for name in selected:
        validate_identifier(name, label="column")
    if not selected:
        raise ValueError(
            "columns must name at least one column to return; a semantic search over a "
            f"vector column cannot fall back to returning {column!r}, since a projected "
            "vector column both floods the result and forfeits the index"
        )
    return [*(name for name in selected if name != DISTANCE_COLUMN), DISTANCE_COLUMN]


def vector_search_sql(
    *,
    table: str,
    column: str,
    query: str,
    k: int = DEFAULT_SEARCH_LIMIT,
    columns: Sequence[str] | None = None,
) -> str:
    """Build the SQL for a ranked semantic top-k search, with the engine embedding ``query``.

    ``column`` is the *text* column a provider-backed vector index was built over — the
    ``source_column`` such an index reports. The engine embeds both the column and the
    query with the index's own provider, so no embedding model is needed on this side.

    ``ORDER BY`` is not optional here, and its absence is silent rather than loud.
    ``vector_search`` returns its rows in rowid order with the distances unsorted, so a
    query that omits the sort looks ranked and is not. Worse, a trailing ``LIMIT`` applied
    without a sort takes the first rows by rowid: ``vector_search(..., 20) LIMIT 3``
    returned the three lowest ids, not the three nearest matches. The fourth argument is
    therefore the only bound emitted, and the sort is always emitted with it.

    Raises ``ValueError`` for identifiers that are not bare SQL identifiers, for a
    non-positive ``k``, and for search text containing null bytes.
    """
    _validate_table_ref(table)
    validate_identifier(column, label="column")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    projection = _semantic_projection(column, columns, include_column=True)
    return (
        f"SELECT {', '.join(projection)} "
        f"FROM vector_search("
        f"{quote_literal(table)}, {quote_literal(column)}, {quote_literal(query)}, {k}) "
        f"ORDER BY {DISTANCE_COLUMN} ASC"
    )


def vector_distance_sql(
    *,
    table: str,
    column: str,
    vector: Sequence[float],
    k: int = DEFAULT_SEARCH_LIMIT,
    columns: Sequence[str] | None = None,
    metric: DistanceMetric = "cosine",
) -> str:
    """Build the SQL for a ranked semantic top-k search from a caller-supplied vector.

    For a plain vector index — one built over a column that already holds vectors. The
    engine does not know how those vectors were produced and cannot embed a query to
    match them, so ``vector`` must come from the same embedding model the column was
    written with. A vector from a different model returns rows in a confident order that
    means nothing.

    ``metric`` selects the distance function and must match the metric the index was
    built with. A mismatch is not an error: the query still returns correct rows, by full
    scan, having quietly declined the index.

    Unlike :func:`vector_search_sql` this shape needs a trailing ``LIMIT``, because the
    distance is computed per row and the sort is what selects the top k. Omitting it also
    forfeits the index rewrite.
    """
    _validate_table_ref(table)
    validate_identifier(column, label="column")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(vector) == 0:
        raise ValueError("query vector must not be empty")
    function = DISTANCE_FUNCTIONS.get(metric)
    if function is None:
        known = ", ".join(sorted(DISTANCE_FUNCTIONS))
        raise ValueError(f"metric must be one of {known}, got {metric!r}")

    projection = _semantic_projection(column, columns, include_column=False)
    literal = vector_literal(vector)
    selected = ", ".join(
        f"{function}({column}, {literal}) AS {DISTANCE_COLUMN}" if name == DISTANCE_COLUMN else name
        for name in projection
    )
    return f"SELECT {selected} FROM {table} ORDER BY {DISTANCE_COLUMN} ASC LIMIT {k}"


def semantic_search_json(
    client: HotdataClient,
    *,
    table: str,
    column: str,
    query: str | None = None,
    vector: Sequence[float] | None = None,
    k: int = DEFAULT_SEARCH_LIMIT,
    columns: Sequence[str] | None = None,
    metric: DistanceMetric = "cosine",
    max_rows: int = 100,
    database: ManagedDatabase | None = None,
    warnings: Sequence[str] = (),
) -> str:
    """Run a semantic search and return ``{"metadata": ..., "rows": [...]}`` as JSON.

    Takes exactly one of ``query`` (the engine embeds it, for a provider-backed index) or
    ``vector`` (already embedded, for a plain one). Rows arrive ranked by ``_distance``
    ascending — nearest first, which is the reverse of what ``score`` means in a
    :func:`bm25_search_json` result.

    Mirrors the envelope every other tool here returns, so an agent sees one result shape
    across all of them.
    """
    if (query is None) == (vector is None):
        raise ValueError(
            "pass exactly one of query (the engine embeds it, for a provider-backed "
            "index) or vector (already embedded, for a plain one)"
        )
    if query is not None:
        sql = vector_search_sql(table=table, column=column, query=query, k=k, columns=columns)
        remedy = search_remedy("vector_search")
    else:
        assert vector is not None
        sql = vector_distance_sql(
            table=table, column=column, vector=vector, k=k, columns=columns, metric=metric
        )
        # A plain index has no text-taking form, so there is no table function to name.
        remedy = (
            "these are the nearest matches rather than the whole set, so to reason over a "
            "wider cohort, rank by distance inside SQL and aggregate there"
        )
    result = client.execute_sql(sql, database=query_scope(database))
    return result_json(result, max_rows=max_rows, warnings=warnings, remedy=remedy)


def clamp_warning(
    *,
    requested: int,
    ceiling: int,
    function: str = "bm25_search",
) -> str | None:
    """Return the warning for a model-supplied ``k`` cut to ``ceiling``, or ``None``.

    The clamp happens before the query runs, so the engine only ever ranks ``ceiling``
    rows and ``row_count`` honestly reports them. Nothing in the result distinguishes
    that from a corpus with only that many matches, which is the whole defect: an agent
    was measured asking for 200, receiving 100, and reporting a cohort it believed was
    200.

    ``function`` names the table function to compose with instead, and differs per
    retrieval route — a semantic tool pointing at ``bm25_search`` would send the model to
    a function with no index on the column it was searching.
    """
    if requested <= ceiling:
        return None
    return (
        f"Asked for k={requested}, but this tool ranks at most {ceiling} rows, so k was "
        f"reduced to {ceiling} before searching. These are the top {ceiling} matches, "
        f"not a sample of {requested}, and rows beyond {ceiling} were never ranked. To "
        f"reason over a wider cohort, call {function} inside SQL and aggregate there "
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


def default_semantic_description(
    table: str,
    column: str,
    *,
    columns: Sequence[str] | None = None,
    max_k: int | None = None,
    composable: bool = True,
) -> str:
    """Return the agent-facing description for a search that ranks by meaning.

    Deliberately parallel to :func:`default_search_description`, and deliberately never
    says vector, embedding or HNSW: the capability is "closest in meaning", and the
    contract has to survive the retrieval strategy changing underneath it.

    Two things differ from the text description and both are load-bearing. The ranking
    column is a *distance*, so nearest is smallest — the opposite direction to ``score``,
    and a model that assumes "higher is better" reads the ranking backwards. And the
    function to compose with for an aggregate is ``vector_search``, not ``bm25_search``.

    ``composable`` says whether this route can be written in SQL at all, and the closing
    advice turns on it. A plain vector index cannot: composing one needs a query vector,
    which SQL cannot express. Sending a model to "rank inside SQL" there contradicts the
    SQL tool's own description, which says on that route that ranking by meaning is
    unavailable — and both descriptions arrive in the same prompt. The route object exists
    to stop those two disagreeing, so it decides this sentence too. The cohort is capped at
    this tool's row limit either way, which is worth saying on both routes: where the
    composed form exists it is the way around the cap, and where it does not, raising ``k``
    is.

    The ``LIKE``/``ILIKE`` guard is carried over unchanged. It exists because stating only
    that ``LIKE`` works was measured pulling models into ``ILIKE '%word%'`` instead of
    searching, and that regression is no less likely on this route.
    """
    if not columns:
        returns = f"Returns the closest-matching rows ordered by a '{DISTANCE_COLUMN}' column"
    else:
        named = [name for name in columns if name != DISTANCE_COLUMN]
        listed = named[0] if len(named) == 1 else f"{', '.join(named[:-1])} and {named[-1]}"
        returns = f"Each hit carries {listed}, ordered by a '{DISTANCE_COLUMN}' column"
    ceiling = "" if max_k is None else f", up to a maximum of {max_k}"
    aggregating = (
        "Use this to list or inspect the matches themselves. When the answer aggregates "
        "over the matches rather than listing them, rank inside SQL instead — that keeps "
        "the whole cohort in the query, where carrying values back as literals caps it at "
        "this tool's row limit."
        if composable
        else "Use this to find the matches, then aggregate over them in SQL using the ids "
        "it returns — ranking by meaning is not available in SQL here, so raise 'k' to "
        "widen the cohort rather than trying to rank there. The cohort is whatever this "
        "returns, so it is capped at this tool's row limit."
    )
    return (
        f"Find rows of {table} whose '{column}' is closest in meaning to a "
        "natural-language query, ranked by closeness. Matches rows that express the same "
        "idea in different words, so it finds paraphrases that share no vocabulary with "
        "the query. LIKE and ILIKE only test for a literal substring you already know, so "
        "they are a filter, not a way to find relevant rows.\n"
        f"{returns}, smallest first — it is a distance, so nearer matches have lower "
        f"values, and 0 would be identical. Distances are comparable within one result "
        f"set but not across queries. Ask for more with 'k' when you need a wider "
        f"net{ceiling}.\n"
        f"{aggregating}"
    )


@dataclass(frozen=True)
class SearchRoute:
    """Which retrieval route a column takes, and the index serving it.

    Resolved once from the control plane and then passed wherever the answer is needed,
    so the search tool and the SQL tool's description cannot disagree about what the
    column supports. They reach the model in the same prompt, and a SQL description
    naming ``bm25_search`` beside a tool that ranks by meaning is a contradiction the
    model has no way to resolve.
    """

    kind: SearchKind
    index: SearchIndex | None = None

    @property
    def semantic(self) -> bool:
        return self.kind == SEMANTIC

    @property
    def composable(self) -> bool:
        """Report whether an agent can write this search itself, in SQL.

        False for a plain vector index: composing one needs a query vector, and an agent
        writing SQL has no way to produce one.
        """
        if not self.semantic:
            return True
        return bool(self.index and self.index.embeds_query)

    @property
    def function(self) -> str:
        """Return the table function to compose with, for the routes that have one."""
        return "vector_search" if self.semantic else "bm25_search"


def resolve_search_route(
    client: HotdataClient,
    *,
    table: str,
    column: str,
    database: ManagedDatabase | None,
    strategy: SearchStrategy = "auto",
    has_embedding: bool = False,
) -> SearchRoute:
    """Return the retrieval route for ``column``, and the index serving it.

    Indexes are invisible to SQL, so the route is read from the control plane once here
    rather than guessed or left to a constructor flag the caller has to keep in step with
    the data.

    Text and meaning are mutually exclusive per column in every configuration the engine
    permits, so ``auto`` needs no preference rule: a BM25 index sits on a text column, a
    plain vector index on a vector column, and a provider-backed vector index cannot
    coexist with any other index on its table. Finding both on one column would mean the
    engine had grown a combination that does not exist today, and the semantic route wins
    only because it is the more specific claim.

    Falls back to text when nothing is known — no ``database_id`` to introspect with, or
    an introspection that failed — which is what this tool did before it could ask. The
    engine's own "No BM25 index found" is then the error, at the same point it was before.
    """
    _, schema, name = table.split(".")
    indexes = (
        list_search_indexes(client, table=name, schema=schema, database=database)
        if database is not None
        else []
    )
    matching = indexes_for_column(indexes, column)
    semantic = next((index for index in matching if index.kind == SEMANTIC), None)
    text = next((index for index in matching if index.kind == TEXT), None)

    if strategy == "text":
        return SearchRoute(TEXT, text)
    if strategy == "semantic":
        if semantic is None:
            named = {index.column for index in indexes if index.kind == SEMANTIC}
            searchable = ", ".join(sorted(named))
            offer = f" Columns searchable by meaning here: {searchable}." if searchable else ""
            raise ValueError(
                f"strategy='semantic' needs a vector index covering {column!r} on {table}, "
                f"and none was found.{offer}"
            )
        _require_embedding(semantic, column=column, has_embedding=has_embedding)
        _require_metric(semantic, column=column)
        return SearchRoute(SEMANTIC, semantic)
    if semantic is not None:
        _require_embedding(semantic, column=column, has_embedding=has_embedding)
        _require_metric(semantic, column=column)
        return SearchRoute(SEMANTIC, semantic)
    if text is None and database is not None:
        logger.debug(
            "no search index covers %s.%s; searching it as text, which the engine will "
            "reject if no BM25 index exists",
            table,
            column,
        )
    return SearchRoute(TEXT, text)


def _require_embedding(index: SearchIndex, *, column: str, has_embedding: bool) -> None:
    """Raise unless ``index`` can be queried, given whether an embedding model was passed.

    A plain vector index is the only route with a hard client-side requirement. The engine
    has no record of how its vectors were produced, so it cannot embed a query to match
    them, and there is no text fallback either — a vector column carries no BM25 index.
    Raising here beats failing at query time, where the model would receive a type error
    about an array argument and have no way to act on it.
    """
    if index.embeds_query or has_embedding:
        return
    raise ValueError(
        f"the vector index on {column!r} was built over a column that already holds "
        "vectors, so the engine cannot embed a query for it. Pass the model the column "
        "was written with — search_embedding= on make_hotdata_tools, embedding= on "
        "make_hotdata_search_tool — or build a provider-backed index over the source "
        "text column instead, which lets the engine embed both sides."
    )


def _require_metric(index: SearchIndex, *, column: str) -> DistanceMetric:
    """Return the metric to emit a distance function for, or raise.

    Only a plain vector index needs one. A provider-backed index resolves the function
    from itself, so its metric is not this side's business and ``cosine`` is returned
    unused.

    Neither an absent nor an unrecognised metric is safe to guess past. Assuming
    ``cosine`` for an index built on ``l2`` is not an error the engine reports — the query
    returns rows, by full scan, ranked by a function the vectors were never indexed for,
    which is a wrong answer that looks like a right one. An unrecognised string is
    survivable but would raise from inside the tool on every invocation, which is the
    failure shape :func:`_require_embedding` exists to avoid. ``HotdataVectorStore``
    refuses to guess in the same situation, for the same reason.
    """
    if index.embeds_query:
        return "cosine"
    where = f"the vector index on {column!r}"
    if index.index_name:
        where = f"{index.index_name}, the vector index on {column!r},"
    if index.metric is None:
        raise ValueError(
            f"{where} reports no metric, so which distance function serves it cannot be "
            "determined. Guessing would rank by a function the vectors were not indexed "
            "for, which returns rows in a confident order that means nothing. Pass "
            "strategy='text' to search this column another way, or rebuild the index with "
            "an explicit metric."
        )
    if index.metric not in DISTANCE_FUNCTIONS:
        known = ", ".join(sorted(DISTANCE_FUNCTIONS))
        raise ValueError(
            f"{where} reports metric {index.metric!r}, which this package has no distance "
            f"function for; it knows {known}. Searching it would fail on every call."
        )
    return cast("DistanceMetric", index.metric)


def make_hotdata_search_tool(
    client: HotdataClient,
    *,
    table: str,
    column: str,
    columns: Sequence[str] | None = None,
    key_column: str | None = DEFAULT_KEY_COLUMN,
    k: int = DEFAULT_SEARCH_LIMIT,
    name: str | None = None,
    description: str | None = None,
    max_rows: int = 100,
    database_id: str | ManagedDatabase | None = None,
    strategy: SearchStrategy = "auto",
    embedding: Embeddings | None = None,
    route: SearchRoute | None = None,
) -> StructuredTool:
    """Return a LangChain tool that searches one indexed column, by text or by meaning.

    Which of the two it does is read off the column's indexes at construction time, not
    chosen by the caller: a BM25 index means text relevance, a vector index means
    closeness in meaning. Both produce one tool with one agent-facing contract — "find the
    rows relevant to this query" — because a model asked to choose between two search
    tools is being handed an implementation detail as a decision.

    ``strategy`` overrides that when a caller wants a specific route. ``"semantic"`` raises
    if no vector index covers the column, since without one this cannot know whether the
    engine embeds the query or what metric to use. ``"text"`` does not raise: introspection
    fails open, so a listing that failed would otherwise turn a working BM25 column into a
    build error. ``embedding`` is a LangChain
    ``Embeddings`` and is required only for a plain vector index, where the engine has no
    way to embed the query itself; it must be the same model the column was written with.
    See :func:`resolve_search_route` for why ``auto`` needs no preference rule.

    The corpus is pinned here rather than chosen by the model: nothing in the tool
    surface lets an agent discover which columns are searchable, and the engine errors
    outright when the index is missing. The agent supplies only ``query`` and an
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
    default_k = k
    database = resolve_database_by_id(client, database_id) if database_id is not None else None
    route = route or resolve_search_route(
        client,
        table=table,
        column=column,
        database=database,
        strategy=strategy,
        has_embedding=embedding is not None,
    )
    index = route.index
    projects_column = not route.semantic or bool(index and index.embeds_query)
    if columns is not None:
        if not route.semantic:
            _projection(column, columns)
            projection = list(columns)
        else:
            # Take the filtered result, not the caller's list: a semantic search over a
            # vector column drops that column from the SELECT, and passing the unfiltered
            # list on would have the description promise a column no hit carries.
            selected = _semantic_projection(column, columns, include_column=projects_column)
            projection = [name for name in selected if name != DISTANCE_COLUMN]
    else:
        projection = _default_columns(
            client,
            table=table,
            column=column,
            key_column=key_column,
            database=database,
            include_column=projects_column,
        )

    if not route.semantic:

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
            name=name or DEFAULT_SEARCH_TOOL_NAME,
            description=description
            or default_search_description(table, column, columns=projection, max_k=max_rows),
            parse_docstring=True,
        )

    embeds_query = bool(index and index.embeds_query)
    metric: DistanceMetric = "cosine" if index is None else _require_metric(index, column=column)
    # Bound outside the closure and left untyped: _require_embedding has already made
    # this non-None on the branch that reaches it.
    embedder: Any = embedding

    def hotdata_search_semantic(query: str, k: int | None = None) -> str:
        """Search by meaning and return the closest rows as JSON.

        Args:
            query: what to look for, in natural language. Describe the idea rather than
                guessing the words the rows use; this matches on meaning, so a phrase
                sharing no vocabulary with a row can still be its closest match.
            k: how many rows to return, nearest first. Values above this tool's row limit
                are reduced to it before the search runs, so a larger k does not widen the
                cohort.
        """
        requested = None if k is None else max(1, k)
        clamped = (
            clamp_warning(requested=requested, ceiling=max_rows, function="vector_search")
            if requested
            else None
        )
        wanted = default_k if requested is None else min(requested, max_rows)
        return semantic_search_json(
            client,
            table=table,
            column=column,
            query=query if embeds_query else None,
            vector=None if embeds_query else embedder.embed_query(query),
            k=wanted,
            columns=projection,
            metric=metric,
            max_rows=max_rows,
            database=database,
            warnings=[clamped] if clamped else (),
        )

    return StructuredTool.from_function(
        func=hotdata_search_semantic,
        name=name or DEFAULT_SEMANTIC_TOOL_NAME,
        description=description
        or default_semantic_description(
            table, column, columns=projection, max_k=max_rows, composable=route.composable
        ),
        parse_docstring=True,
    )


def _default_columns(
    client: HotdataClient,
    *,
    table: str,
    column: str,
    key_column: str | None,
    database: ManagedDatabase | None,
    include_column: bool = True,
) -> list[str]:
    """Return the projection to use when the caller named none.

    The searched column, preceded by ``key_column`` when the table actually has one.
    Falls back to the searched column alone when the lookup fails, so a schema query is
    never the reason tool construction fails.

    ``include_column`` is false when the searched column holds vectors, which leaves the
    key column carrying the result on its own. That is a thin hit, and deliberately so:
    the alternative is shipping a 1536-wide float list per row. A caller searching a
    vector column should name ``columns`` to say which text belongs beside the key.
    """
    if not include_column:
        if key_column is None:
            raise ValueError(
                f"searching the vector column {column!r} returns no readable column on "
                "its own; pass columns=[...] naming what a hit should carry, or "
                "key_column= for the column that joins it back to the table"
            )
        if not _has_column(client, table=table, column=key_column, database=database, default=True):
            raise ValueError(
                f"{table} has no {key_column!r} column, so searching the vector column "
                f"{column!r} would return nothing readable; pass columns=[...] naming "
                "what a hit should carry"
            )
        return [key_column]
    if key_column is None or key_column == column:
        return [column]
    if not _has_column(client, table=table, column=key_column, database=database, default=False):
        return [column]
    return [key_column, column]


def _has_column(
    client: HotdataClient,
    *,
    table: str,
    column: str,
    database: ManagedDatabase | None,
    default: bool,
) -> bool:
    """Report whether ``table`` has ``column``, returning ``default`` if the check fails.

    ``default`` is what an unanswerable question means to the caller. Building a text
    projection, a failed check means dropping the join key and carrying on, so it is
    false. Building a vector-column projection, the key is the only readable thing a hit
    would carry, and dropping it on a failed lookup would leave a tool that returns
    nothing but distances — so there it is true, and a table genuinely lacking the column
    fails later against a message naming it.
    """
    catalog, schema, name = table.split(".")
    sql = (
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_catalog = {quote_literal(catalog)} "
        f"AND table_schema = {quote_literal(schema)} "
        f"AND table_name = {quote_literal(name)} "
        f"AND column_name = {quote_literal(column)}"
    )
    try:
        found = bool(client.execute_sql(sql, database=query_scope(database)).rows)
    except Exception:
        logger.warning(
            "could not check %s for a %r column; assuming %s",
            table,
            column,
            "it exists" if default else "it does not",
            exc_info=True,
        )
        return default
    if not found:
        logger.debug("%s has no %r column", table, column)
    return found
