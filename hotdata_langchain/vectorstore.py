"""LangChain ``VectorStore`` backed by a Hotdata managed table."""

from __future__ import annotations

import json
import logging
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq
from hotdata_framework import DEFAULT_SCHEMA, HotdataClient, ManagedDatabase
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from hotdata_langchain._sql import quote_literal, validate_identifier
from hotdata_langchain.databases import resolve_database_by_id

logger = logging.getLogger(__name__)

DistanceMetric = Literal["cosine", "l2", "dot"]
MetadataColumnType = Literal["string", "int", "float", "bool"]

#: Engine scalar UDF backing each metric. All three return lower-is-closer distances.
DISTANCE_FUNCTIONS: dict[str, str] = {
    "cosine": "cosine_distance",
    "l2": "l2_distance",
    "dot": "negative_dot_product",
}

ID_COLUMN = "id"
CONTENT_COLUMN = "content"
METADATA_COLUMN = "metadata_json"
EMBEDDING_COLUMN = "embedding"
DISTANCE_ALIAS = "dist"

RESERVED_COLUMNS = frozenset(
    {ID_COLUMN, CONTENT_COLUMN, METADATA_COLUMN, EMBEDDING_COLUMN, DISTANCE_ALIAS}
)

_ARROW_TYPES: dict[str, pa.DataType] = {
    "string": pa.string(),
    "int": pa.int64(),
    "float": pa.float64(),
    "bool": pa.bool_(),
}

_PYTHON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
}


def _matches_type(value: Any, column_type: str) -> bool:
    """Report whether ``value`` may be stored in a column declared ``column_type``.

    ``bool`` is a subclass of ``int`` in Python, so a plain ``isinstance`` check would
    accept ``True`` for an ``int`` column and store it as ``1`` while ``metadata_json``
    kept ``true`` — the same key disagreeing with itself across the two
    representations. Booleans are therefore only ever accepted by a ``bool`` column.
    """
    if isinstance(value, bool):
        return column_type == "bool"
    return isinstance(value, _PYTHON_TYPES[column_type])


class HotdataVectorStore(VectorStore):
    """Vector store over one managed table in a Hotdata managed database.

    Rows are stored as ``id`` / ``content`` / ``metadata_json`` / ``embedding``, keyed on
    ``id`` so writes upsert rather than duplicate. Searches run as a single SQL query
    using the engine's scalar distance UDFs::

        SELECT id, content, metadata_json,
               cosine_distance(embedding, ARRAY[...]) AS dist
        FROM "default"."public"."vectors"
        ORDER BY dist ASC
        LIMIT k

    That shape is correct with no index at all: it brute-forces the table, which is what
    every search does today. It is also written to match the shape the engine's
    optimizer rewrites into an HNSW index lookup, so the same query should get faster
    once a matching-metric index exists on the embedding column, with nothing here
    changing. That rewrite is not yet confirmed for these queries: its conditions come
    from reading the engine's optimizer rule, and observing it needs an index this
    package cannot create. The raw ``embedding`` column is never projected, because a
    vector column in the output declines the rewrite.

    ``database_id`` addresses the database by id and is resolved once here; every read
    and write afterwards addresses the resolved record. The store never creates a
    database, and never resolves one by name — a Hotdata database name is a display
    label and is not unique.

    The table is declared here, keyed on ``id``. Let the store declare it, or declare it
    yourself with that key — a managed table declared without a key accepts writes as
    appends, so re-adding a document would duplicate it rather than replace it, and the
    key of an existing table cannot be read back to warn about it.

    ``metadata_columns`` declares which metadata keys are promoted to real typed columns
    so they can be filtered on. Full metadata always round-trips through
    ``metadata_json`` regardless; promotion only buys filterability. It has to match the
    table: an upsert must carry every column the table has, so pointing a store at an
    existing table with different promoted columns fails the first write with ``upload is
    missing column '<name>'``.

    ``distance`` defaults to ``"cosine"``, whose relevance score (``1 - distance``) is
    exact and assumes nothing about embedding scale. ``"l2"`` maps to the engine's
    ``l2_distance``, which is *squared* L2, while LangChain's Euclidean relevance score
    expects true Euclidean distance over unit-normalised vectors — relevance scores
    under ``"l2"`` are therefore on the wrong scale, though the ranking itself is
    correct. ``"dot"`` maps to ``negative_dot_product``.
    """

    def __init__(
        self,
        client: HotdataClient,
        embedding: Embeddings,
        *,
        database_id: str | ManagedDatabase,
        table: str = "vectors",
        schema: str = DEFAULT_SCHEMA,
        distance: DistanceMetric = "cosine",
        metadata_columns: Mapping[str, MetadataColumnType] | None = None,
    ) -> None:
        if distance not in DISTANCE_FUNCTIONS:
            raise ValueError(
                f"distance must be one of {sorted(DISTANCE_FUNCTIONS)}, got {distance!r}"
            )
        validate_identifier(table, label="table")
        validate_identifier(schema, label="schema")

        promoted = dict(metadata_columns or {})
        for name, column_type in promoted.items():
            validate_identifier(name, label="metadata column")
            if name in RESERVED_COLUMNS:
                raise ValueError(f"metadata column {name!r} collides with a reserved column")
            if column_type not in _ARROW_TYPES:
                raise ValueError(
                    f"metadata column {name!r} has unsupported type {column_type!r}; "
                    f"expected one of {sorted(_ARROW_TYPES)}"
                )

        self._client = client
        self._embedding = embedding
        self._table = table
        self._schema = schema
        self._distance: DistanceMetric = distance
        self._metadata_columns = promoted
        self._database = resolve_database_by_id(client, database_id)
        self._declare_table()

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def database(self) -> ManagedDatabase:
        """The resolved managed database every query and load is addressed to."""
        return self._database

    @property
    def table_ref(self) -> str:
        """Fully qualified table reference used in generated SQL."""
        return f'"default"."{self._schema}"."{self._table}"'

    def _table_exists(self) -> bool:
        return any(
            managed.table == self._table
            for managed in self._client.list_managed_tables(self._database, schema=self._schema)
        )

    def _declare_table(self) -> None:
        """Declare the table keyed on ``id`` unless it already exists.

        The key is what makes upsert and delete loads address existing rows; a keyless
        table takes writes as appends instead. Existence is checked first rather than
        declaring and swallowing whatever comes back, because the client reports a
        permission failure, an outage and an already-declared table as the same
        ``RuntimeError`` — swallowing them all would construct a store that looks
        correctly keyed and duplicates rows on every write instead.

        A declaration that loses a race with another process is the one tolerated
        failure, and only once the table is confirmed to exist.
        """
        if self._table_exists():
            return
        try:
            self._client.add_managed_table(
                self._database,
                self._table,
                schema=self._schema,
                key=[ID_COLUMN],
            )
        except RuntimeError:
            if not self._table_exists():
                raise
            logger.debug("table %s was declared concurrently", self.table_ref)

    # ------------------------------------------------------------------ writes

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed ``texts`` and upsert them, returning their ids in input order.

        Ids are generated per row where one is not supplied, so the returned list is
        always fully populated. Re-adding a row with an existing id replaces it.
        """
        texts = list(texts)
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError(
                f"got {len(metadatas)} metadatas for {len(texts)} texts; they must match"
            )
        if ids is not None and len(ids) != len(texts):
            raise ValueError(f"got {len(ids)} ids for {len(texts)} texts; they must match")
        if not texts:
            return []

        supplied: list[str | None] = list(ids) if ids is not None else [None] * len(texts)
        resolved_ids = [self._row_id(row_id) for row_id in supplied]
        row_metadatas = metadatas if metadatas is not None else [{} for _ in texts]
        vectors = self._embedding.embed_documents(texts)

        arrays: list[pa.Array] = [
            pa.array(resolved_ids, pa.string()),
            pa.array(texts, pa.string()),
            pa.array([json.dumps(metadata) for metadata in row_metadatas], pa.string()),
            pa.array(vectors, pa.list_(pa.float32())),
        ]
        names = [ID_COLUMN, CONTENT_COLUMN, METADATA_COLUMN, EMBEDDING_COLUMN]
        for name, column_type in self._metadata_columns.items():
            arrays.append(self._promoted_array(name, column_type, row_metadatas))
            names.append(name)

        self._load(pa.table(arrays, names=names), mode="upsert")
        return resolved_ids

    @staticmethod
    def _row_id(supplied: str | None) -> str:
        """Return the id to store, generating one only when none was supplied.

        ``add_documents`` passes ``None`` for a document without an id, which is the
        only case that gets a generated one. An empty string is a caller mistake rather
        than an absent id, so it raises instead of being quietly replaced.
        """
        if supplied is None:
            return uuid.uuid4().hex
        if not supplied:
            raise ValueError("document ids must not be empty")
        quote_literal(supplied)  # rejects ids that could not be looked up later
        return supplied

    @staticmethod
    def _promoted_array(
        name: str,
        column_type: MetadataColumnType,
        metadatas: Sequence[Mapping[str, Any]],
    ) -> pa.Array:
        values = [metadata.get(name) for metadata in metadatas]
        for value in values:
            if value is not None and not _matches_type(value, column_type):
                raise ValueError(
                    f"metadata key {name!r} is declared {column_type!r} but got "
                    f"{type(value).__name__} ({value!r})"
                )
        return pa.array(values, _ARROW_TYPES[column_type])

    def _load(self, table: pa.Table, *, mode: Literal["upsert", "delete"]) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{self._table}.parquet"
            pq.write_table(table, path)
            self._client.load_managed_table(
                self._database,
                self._table,
                schema=self._schema,
                file=str(path),
                mode=mode,
                key=[ID_COLUMN],
            )

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        """Delete rows by id.

        ``ids`` is required: an omitted-means-everything delete is too easy to trigger
        by accident to expose. Ids that are not present are ignored. Backend failures
        raise rather than returning ``False``, since a delete that silently reports
        success is worse than one that fails loudly.
        """
        if ids is None:
            raise ValueError("delete requires ids; deleting the whole store is not supported")
        if not ids:
            return True
        for row_id in ids:
            quote_literal(row_id)
        self._load(pa.table([pa.array(ids, pa.string())], names=[ID_COLUMN]), mode="delete")
        return True

    # ------------------------------------------------------------------- reads

    def _where(self, filter: Mapping[str, Any] | None) -> str:
        """Build a ``WHERE`` clause of equality predicates over promoted columns.

        The clause goes into the search query itself rather than wrapping its result:
        filtering after a top-k selection can only shrink the result, never re-fill it
        back to ``k``.
        """
        if not filter:
            return ""
        predicates = []
        for key, value in filter.items():
            if key.startswith("$") or isinstance(value, Mapping):
                raise ValueError(
                    f"filter operators are not supported yet; {key!r} must be a plain "
                    "equality of the form {'key': value}"
                )
            column_type = self._metadata_columns.get(key)
            if column_type is None:
                raise ValueError(
                    f"cannot filter on metadata key {key!r}: only keys declared in "
                    f"metadata_columns are filterable, which are "
                    f"{sorted(self._metadata_columns) or '(none)'}"
                )
            predicates.append(f"{key} = {self._literal(key, column_type, value)}")
        return " WHERE " + " AND ".join(predicates)

    @staticmethod
    def _literal(key: str, column_type: MetadataColumnType, value: Any) -> str:
        if not _matches_type(value, column_type):
            raise ValueError(
                f"filter on {key!r} expects {column_type}, got {type(value).__name__} ({value!r})"
            )
        if column_type == "string":
            return quote_literal(cast("str", value))
        if column_type == "bool":
            return "true" if value else "false"
        return repr(value)

    def _search_sql(
        self,
        embedding: Sequence[float],
        k: int,
        filter: Mapping[str, Any] | None,
    ) -> str:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if len(embedding) == 0:
            raise ValueError("query embedding must not be empty")
        vector = "ARRAY[" + ", ".join(repr(float(value)) for value in embedding) + "]"
        return (
            f"SELECT {ID_COLUMN}, {CONTENT_COLUMN}, {METADATA_COLUMN}, "
            f"{DISTANCE_FUNCTIONS[self._distance]}({EMBEDDING_COLUMN}, {vector}) "
            f"AS {DISTANCE_ALIAS} "
            f"FROM {self.table_ref}"
            f"{self._where(filter)} "
            f"ORDER BY {DISTANCE_ALIAS} ASC "
            f"LIMIT {k}"
        )

    def _rows(self, sql: str) -> list[dict[str, Any]]:
        return self._client.execute_sql(sql, database=self._database).to_records()

    @staticmethod
    def _document(row: Mapping[str, Any]) -> Document:
        raw = row.get(METADATA_COLUMN)
        return Document(
            id=row[ID_COLUMN],
            page_content=row[CONTENT_COLUMN],
            metadata=json.loads(raw) if raw else {},
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: Sequence[float],
        k: int = 4,
        *,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return the ``k`` nearest rows with their raw distances, nearest first.

        Scores are distances (lower is closer), not relevance scores; use
        ``similarity_search_with_relevance_scores`` for a ``[0, 1]`` relevance score.
        """
        rows = self._rows(self._search_sql(embedding, k, filter))
        return [(self._document(row), float(row[DISTANCE_ALIAS])) for row in rows]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return self.similarity_search_with_score_by_vector(
            self._embedding.embed_query(query), k, filter=filter, **kwargs
        )

    def similarity_search_by_vector(
        self,
        embedding: Sequence[float],
        k: int = 4,
        *,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        scored = self.similarity_search_with_score_by_vector(embedding, k, filter=filter, **kwargs)
        return [document for document, _ in scored]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        scored = self.similarity_search_with_score(query, k, filter=filter, **kwargs)
        return [document for document, _ in scored]

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Return the rows with these ids, skipping any that are absent."""
        if not ids:
            return []
        wanted = ", ".join(quote_literal(row_id) for row_id in ids)
        rows = self._rows(
            f"SELECT {ID_COLUMN}, {CONTENT_COLUMN}, {METADATA_COLUMN} "
            f"FROM {self.table_ref} "
            f"WHERE {ID_COLUMN} IN ({wanted})"
        )
        return [self._document(row) for row in rows]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self._distance == "cosine":
            return self._cosine_relevance_score_fn
        if self._distance == "dot":
            return self._max_inner_product_relevance_score_fn
        return self._euclidean_relevance_score_fn

    # ----------------------------------------------------------- construction

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> HotdataVectorStore:
        """Build a store and write ``texts`` into it.

        ``client`` and ``database_id`` are passed through ``kwargs`` alongside any other
        constructor argument, the extension point the base signature leaves open.
        """
        client = kwargs.pop("client", None)
        if client is None:
            raise ValueError("from_texts requires client=<HotdataClient>")
        store = cls(client, embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store
