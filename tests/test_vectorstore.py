"""Behaviour the LangChain conformance suite does not reach.

The suite in ``test_vectorstore_standard.py`` certifies the id/CRUD contract. It never
looks at the SQL, the stored column types, filtering, scores, or ``from_texts`` — which
is where a Hotdata-specific implementation can be wrong while still conforming.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
from hotdata.exceptions import ApiException
from hotdata_framework import ManagedDatabase, ManagedTable
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import VectorStore

from hotdata_langchain.vectorstore import HotdataVectorStore
from tests.conftest import vector_index
from tests.fake_hotdata import FakeHotdataClient

EMBEDDING_SIZE = 6


@pytest.fixture
def fake_client() -> FakeHotdataClient:
    return FakeHotdataClient()


@pytest.fixture
def store(
    fake_client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
) -> HotdataVectorStore:
    return HotdataVectorStore(
        fake_client,  # type: ignore[arg-type]
        DeterministicFakeEmbedding(size=EMBEDDING_SIZE),
        database_id=managed_db.id,
        metadata_columns={"city": "string", "beds": "int", "rating": "float", "live": "bool"},
    )


def _build_store(
    client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    **kwargs: Any,
) -> HotdataVectorStore:
    return HotdataVectorStore(
        client,  # type: ignore[arg-type]
        DeterministicFakeEmbedding(size=EMBEDDING_SIZE),
        database_id=managed_db.id,
        **kwargs,
    )


def _search(store: HotdataVectorStore, **kwargs: Any) -> str:
    """Run one search and return the SQL it emitted."""
    store.similarity_search("anything", **kwargs)
    client: FakeHotdataClient = store._client  # type: ignore[assignment]
    return client.queries[-1]


# ------------------------------------------------------------------- addressing


def test_constructor_resolves_by_id_exactly_once(
    fake_client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
) -> None:
    """One id lookup at construction is the only lookup the store is allowed."""
    store = HotdataVectorStore(
        fake_client,  # type: ignore[arg-type]
        DeterministicFakeEmbedding(size=EMBEDDING_SIZE),
        database_id=managed_db.id,
    )
    store.add_texts(["a"])
    store.similarity_search("a")

    databases_api.return_value.get_database.assert_called_once_with(managed_db.id)
    databases_api.return_value.list_databases.assert_not_called()


def test_every_call_addresses_the_resolved_record(store: HotdataVectorStore) -> None:
    client: FakeHotdataClient = store._client  # type: ignore[assignment]
    store.add_texts(["a"])

    assert store.database is store._database
    assert client.declared[0]["database"] is store.database


def test_unknown_metric_is_rejected(
    fake_client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
) -> None:
    with pytest.raises(ValueError, match="distance must be one of"):
        HotdataVectorStore(
            fake_client,  # type: ignore[arg-type]
            DeterministicFakeEmbedding(size=EMBEDDING_SIZE),
            database_id=managed_db.id,
            distance="manhattan",  # type: ignore[arg-type]
        )


def test_reserved_metadata_column_is_rejected(
    fake_client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
) -> None:
    with pytest.raises(ValueError, match="collides with a reserved column"):
        HotdataVectorStore(
            fake_client,  # type: ignore[arg-type]
            DeterministicFakeEmbedding(size=EMBEDDING_SIZE),
            database_id=managed_db.id,
            metadata_columns={"content": "string"},
        )


def test_table_name_must_be_an_identifier(
    fake_client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
) -> None:
    with pytest.raises(ValueError, match="table must be a bare SQL identifier"):
        HotdataVectorStore(
            fake_client,  # type: ignore[arg-type]
            DeterministicFakeEmbedding(size=EMBEDDING_SIZE),
            database_id=managed_db.id,
            table='vectors"; drop table x',
        )


# ------------------------------------------------------------------------ writes


def test_table_is_declared_keyed_on_id(store: HotdataVectorStore) -> None:
    """Without the key, upsert and delete loads silently degrade to append-only."""
    client: FakeHotdataClient = store._client  # type: ignore[assignment]
    assert client.declared == [
        {"database": store.database, "table": "vectors", "schema": "public", "key": ["id"]}
    ]


def test_existing_table_is_not_redeclared(
    fake_client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
) -> None:
    _build_store(fake_client, managed_db)
    _build_store(fake_client, managed_db)
    assert len(fake_client.declared) == 1


def test_declaration_failure_is_not_swallowed(
    fake_client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
) -> None:
    """A store that cannot declare its keyed table would append instead of upsert.

    The client reports a permission failure and an already-declared table as the same
    RuntimeError, so a blanket catch here would construct a store that looks correctly
    keyed and duplicates every row it writes.
    """
    fake_client.add_managed_table = MagicMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("Forbidden")
    )
    with pytest.raises(RuntimeError, match="Forbidden"):
        _build_store(fake_client, managed_db)


def test_losing_a_declaration_race_is_tolerated(
    fake_client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
) -> None:
    """The table is absent when checked, then present once the failed declare returns."""
    declared = ManagedTable(
        full_name=f"{managed_db.id}.public.vectors",
        schema="public",
        table="vectors",
        synced=False,
        last_sync=None,
    )
    fake_client.list_managed_tables = MagicMock(side_effect=[[], [declared]])  # type: ignore[method-assign]
    fake_client.add_managed_table = MagicMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("already exists")
    )
    _build_store(fake_client, managed_db)


def test_supplied_ids_are_preserved_and_missing_ones_generated(
    store: HotdataVectorStore,
) -> None:
    written = store.add_texts(["a", "b"], ids=["mine", None])  # type: ignore[list-item]
    assert written[0] == "mine"
    assert written[1] and written[1] != "mine"


def test_empty_id_is_rejected_rather_than_replaced(store: HotdataVectorStore) -> None:
    """An empty id is a caller mistake, not an absent id to fill in."""
    with pytest.raises(ValueError, match="document ids must not be empty"):
        store.add_texts(["a"], ids=[""])


def test_writes_are_upserts(store: HotdataVectorStore) -> None:
    store.add_texts(["a"])
    client: FakeHotdataClient = store._client  # type: ignore[assignment]
    assert client.loads == ["upsert"]


def test_written_column_types(store: HotdataVectorStore) -> None:
    store.add_texts(["a"], [{"city": "sf", "beds": 2, "rating": 4.5, "live": True}])
    client: FakeHotdataClient = store._client  # type: ignore[assignment]
    schema = client.schemas[0]

    assert schema.field("id").type == pa.string()
    assert schema.field("content").type == pa.string()
    assert schema.field("metadata_json").type == pa.string()
    assert schema.field("embedding").type == pa.list_(pa.float32())
    assert schema.field("city").type == pa.string()
    assert schema.field("beds").type == pa.int64()
    assert schema.field("rating").type == pa.float64()
    assert schema.field("live").type == pa.bool_()


def test_embedding_round_trips_at_full_width(store: HotdataVectorStore) -> None:
    store.add_texts(["a"])
    client: FakeHotdataClient = store._client  # type: ignore[assignment]
    stored = next(iter(client.rows.values()))
    assert len(stored["embedding"]) == EMBEDDING_SIZE


def test_metadata_round_trips_including_non_string_values(store: HotdataVectorStore) -> None:
    metadata = {"city": "sf", "beds": 2, "nested": {"tags": ["a", "b"]}, "live": True}
    store.add_texts(["a"], [metadata], ids=["one"])
    assert store.get_by_ids(["one"])[0].metadata == metadata


def test_undeclared_metadata_keys_are_stored_but_not_promoted(store: HotdataVectorStore) -> None:
    store.add_texts(["a"], [{"whatever": "kept"}], ids=["one"])
    client: FakeHotdataClient = store._client  # type: ignore[assignment]

    assert "whatever" not in client.schemas[0].names
    assert store.get_by_ids(["one"])[0].metadata == {"whatever": "kept"}


def test_promoted_value_of_the_wrong_type_is_rejected(store: HotdataVectorStore) -> None:
    with pytest.raises(ValueError, match="declared 'int' but got str"):
        store.add_texts(["a"], [{"beds": "two"}])


def test_a_bool_is_not_accepted_by_an_int_column(store: HotdataVectorStore) -> None:
    """bool subclasses int, so a plain isinstance check would store True as 1 here
    while metadata_json kept true — one key disagreeing with itself."""
    with pytest.raises(ValueError, match="declared 'int' but got bool"):
        store.add_texts(["a"], [{"beds": True}])


def test_a_bool_is_not_accepted_by_a_float_column(store: HotdataVectorStore) -> None:
    with pytest.raises(ValueError, match="declared 'float' but got bool"):
        store.add_texts(["a"], [{"rating": True}])


def test_a_bool_column_still_accepts_bools(store: HotdataVectorStore) -> None:
    store.add_texts(["a"], [{"live": True}], ids=["one"])
    assert store.get_by_ids(["one"])[0].metadata == {"live": True}


def test_mismatched_metadatas_length_is_rejected(store: HotdataVectorStore) -> None:
    with pytest.raises(ValueError, match="2 metadatas for 1 texts"):
        store.add_texts(["a"], [{}, {}])


def test_adding_nothing_writes_nothing(store: HotdataVectorStore) -> None:
    client: FakeHotdataClient = store._client  # type: ignore[assignment]
    assert store.add_texts([]) == []
    assert client.loads == []


# ----------------------------------------------------------------------- deletes


def test_delete_requires_ids(store: HotdataVectorStore) -> None:
    """An omitted-means-everything delete is too easy to trigger by accident."""
    with pytest.raises(ValueError, match="delete requires ids"):
        store.delete()


def test_delete_uses_a_delete_load(store: HotdataVectorStore) -> None:
    store.add_texts(["a"], ids=["one"])
    store.delete(["one"])
    client: FakeHotdataClient = store._client  # type: ignore[assignment]

    assert client.loads == ["upsert", "delete"]
    assert client.rows == {}


# --------------------------------------------------------------------- read SQL


def test_search_sql_shape(store: HotdataVectorStore) -> None:
    sql = _search(store, k=3)
    assert sql.startswith("SELECT id, content, metadata_json, cosine_distance(embedding, ARRAY[")
    assert sql.endswith('FROM "default"."public"."vectors" ORDER BY dist ASC LIMIT 3')


def test_search_never_projects_the_vector_column(store: HotdataVectorStore) -> None:
    """A vector column in the output declines the engine's HNSW fast-path rewrite."""
    sql = _search(store)
    projection = sql[len("SELECT ") : sql.index(" FROM ")]
    plain_columns, _, distance_expression = projection.partition("cosine_distance(")

    assert [column.strip() for column in plain_columns.split(",") if column.strip()] == [
        "id",
        "content",
        "metadata_json",
    ]
    assert distance_expression.startswith("embedding, ARRAY[")


@pytest.mark.parametrize(
    ("distance", "function"),
    [("cosine", "cosine_distance"), ("l2", "l2_distance"), ("dot", "negative_dot_product")],
)
def test_metric_selects_its_distance_function(
    fake_client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
    distance: Any,
    function: str,
) -> None:
    store = HotdataVectorStore(
        fake_client,  # type: ignore[arg-type]
        DeterministicFakeEmbedding(size=EMBEDDING_SIZE),
        database_id=managed_db.id,
        distance=distance,
    )
    assert f"{function}(embedding, ARRAY[" in _search(store)


@pytest.mark.parametrize(
    ("distance", "expected"),
    [("cosine", 0.75), ("dot", 0.75), ("l2", 1.0 - 0.25 / 2**0.5)],
)
def test_relevance_score_per_metric(
    fake_client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
    distance: Any,
    expected: float,
) -> None:
    store = HotdataVectorStore(
        fake_client,  # type: ignore[arg-type]
        DeterministicFakeEmbedding(size=EMBEDDING_SIZE),
        database_id=managed_db.id,
        distance=distance,
    )
    assert store._select_relevance_score_fn()(0.25) == pytest.approx(expected)


def test_scores_are_distances_nearest_first(store: HotdataVectorStore) -> None:
    store.add_texts(["alpha", "beta", "gamma"])
    scored = store.similarity_search_with_score("alpha", k=3)

    assert [score for _, score in scored] == sorted(score for _, score in scored)
    assert scored[0][0].page_content == "alpha"
    assert scored[0][1] == pytest.approx(0.0)


def test_k_must_be_positive(store: HotdataVectorStore) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        store.similarity_search("a", k=0)


def test_get_by_ids_without_ids_does_not_query(store: HotdataVectorStore) -> None:
    client: FakeHotdataClient = store._client  # type: ignore[assignment]
    assert store.get_by_ids([]) == []
    assert client.queries == []


# ---------------------------------------------------------------------- filtering


def test_filter_predicate_is_inside_the_search_query(store: HotdataVectorStore) -> None:
    """Filtering after a top-k selection can only shrink the result, never re-fill it."""
    sql = _search(store, k=2, filter={"city": "sf"})
    assert " WHERE city = 'sf' ORDER BY dist ASC LIMIT 2" in sql
    assert sql.index("WHERE") < sql.index("ORDER BY")


def test_filter_combines_predicates_with_and(store: HotdataVectorStore) -> None:
    sql = _search(store, filter={"city": "sf", "beds": 2, "live": True})
    assert "WHERE city = 'sf' AND beds = 2 AND live = true" in sql


def test_filter_narrows_results(store: HotdataVectorStore) -> None:
    store.add_texts(["a", "b"], [{"city": "sf"}, {"city": "nyc"}])
    found = store.similarity_search("a", k=5, filter={"city": "nyc"})
    assert [document.page_content for document in found] == ["b"]


def test_filter_literals_are_quoted(store: HotdataVectorStore) -> None:
    sql = _search(store, filter={"city": "o'brien"})
    assert "WHERE city = 'o''brien'" in sql


def test_filter_on_an_undeclared_key_is_rejected(store: HotdataVectorStore) -> None:
    """Fail at call time rather than returning a silently unfiltered result."""
    with pytest.raises(ValueError, match="only keys declared in metadata_columns"):
        store.similarity_search("a", filter={"country": "us"})


def test_filter_of_the_wrong_type_is_rejected(store: HotdataVectorStore) -> None:
    with pytest.raises(ValueError, match="filter on 'beds' expects int"):
        store.similarity_search("a", filter={"beds": "two"})


def test_filtering_an_int_column_by_a_bool_is_rejected(store: HotdataVectorStore) -> None:
    """Otherwise `beds=True` would silently become `beds = 1`."""
    with pytest.raises(ValueError, match="filter on 'beds' expects int"):
        store.similarity_search("a", filter={"beds": True})


def test_filter_operators_are_rejected_explicitly(store: HotdataVectorStore) -> None:
    with pytest.raises(ValueError, match="filter operators are not supported yet"):
        store.similarity_search("a", filter={"beds": {"$gte": 2}})


# -------------------------------------------------------------------- from_texts


def test_from_texts_round_trip(
    fake_client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
) -> None:
    store = HotdataVectorStore.from_texts(
        ["alpha", "beta"],
        DeterministicFakeEmbedding(size=EMBEDDING_SIZE),
        [{"city": "sf"}, {"city": "nyc"}],
        ids=["one", "two"],
        client=fake_client,
        database_id=managed_db.id,
        metadata_columns={"city": "string"},
    )
    found = store.similarity_search("alpha", k=1)

    assert isinstance(store, VectorStore)
    assert found[0].id == "one"
    assert found[0].metadata == {"city": "sf"}


def test_from_texts_requires_a_client(managed_db: ManagedDatabase) -> None:
    with pytest.raises(ValueError, match="from_texts requires client="):
        HotdataVectorStore.from_texts(
            ["a"],
            DeterministicFakeEmbedding(size=EMBEDDING_SIZE),
            database_id=managed_db.id,
        )


def test_as_retriever_works_off_the_base_class(store: HotdataVectorStore) -> None:
    store.add_texts(["alpha", "beta"], ids=["one", "two"])
    retriever = store.as_retriever(search_kwargs={"k": 1})

    assert [document.id for document in retriever.invoke("alpha")] == ["one"]


# ---------------------------------------------------------------------- indexing


@pytest.mark.parametrize("distance", ["cosine", "l2", "dot"])
def test_create_index_builds_for_the_stores_own_metric(
    fake_client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
    indexes_api: MagicMock,
    distance: str,
) -> None:
    """The metric is what earns the index lookup; the server would otherwise default to l2."""
    store = _build_store(fake_client, managed_db, distance=distance)
    store.add_texts(["alpha"], ids=["one"])
    store.create_index()

    assert fake_client.indexes[0]["metric"] == distance


def test_create_index_requests_one_vector_index_over_the_embedding_column(
    store: HotdataVectorStore,
    fake_client: FakeHotdataClient,
    indexes_api: MagicMock,
) -> None:
    """No dimensions: the engine reads the width off stored data, and ignores a supplied one."""
    store.add_texts(["alpha"], ids=["one"])
    store.create_index()

    request = fake_client.indexes[0]
    assert request["index_type"] == "vector"
    assert request["columns"] == ["embedding"]
    assert "dimensions" not in request
    assert request["timeout_s"] > 300.0


def test_create_index_is_a_no_op_when_a_matching_index_exists(
    store: HotdataVectorStore,
    fake_client: FakeHotdataClient,
    indexes_api: MagicMock,
) -> None:
    indexes_api.return_value.list_indexes.return_value = SimpleNamespace(
        indexes=[vector_index(metric="cosine")]
    )

    assert store.create_index() is None
    assert fake_client.indexes == []


def test_create_index_rejects_an_existing_index_on_another_metric(
    store: HotdataVectorStore,
    fake_client: FakeHotdataClient,
    indexes_api: MagicMock,
) -> None:
    """A mismatched index is not an error at query time; it just silently full-scans."""
    indexes_api.return_value.list_indexes.return_value = SimpleNamespace(
        indexes=[vector_index(metric="l2")]
    )

    with pytest.raises(ValueError, match="silently falls back to a full scan"):
        store.create_index()
    assert fake_client.indexes == []


def test_create_index_ignores_an_index_on_another_column(
    store: HotdataVectorStore,
    fake_client: FakeHotdataClient,
    indexes_api: MagicMock,
) -> None:
    bm25 = SimpleNamespace(
        index_name="content_bm25",
        index_type="bm25",
        columns=["content"],
        metric=None,
        status="ready",
    )
    indexes_api.return_value.list_indexes.return_value = SimpleNamespace(indexes=[bm25])
    store.add_texts(["alpha"], ids=["one"])
    store.create_index()

    assert fake_client.indexes[0]["columns"] == ["embedding"]


def test_from_texts_does_not_index_unless_asked(
    fake_client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
    indexes_api: MagicMock,
) -> None:
    HotdataVectorStore.from_texts(
        ["alpha"],
        DeterministicFakeEmbedding(size=EMBEDDING_SIZE),
        client=fake_client,
        database_id=managed_db.id,
    )

    assert fake_client.indexes == []


def test_from_texts_indexes_after_writing(
    fake_client: FakeHotdataClient,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
    indexes_api: MagicMock,
) -> None:
    """An index built before the first write has no stored data to read a width from."""
    HotdataVectorStore.from_texts(
        ["alpha", "beta"],
        DeterministicFakeEmbedding(size=EMBEDDING_SIZE),
        ids=["one", "two"],
        create_index=True,
        client=fake_client,
        database_id=managed_db.id,
    )

    assert fake_client.indexes[0]["rows_at_build"] == 2


def test_create_index_tolerates_losing_a_build_race(
    store: HotdataVectorStore,
    fake_client: FakeHotdataClient,
    indexes_api: MagicMock,
) -> None:
    """Absent when checked, present once another process's build returns."""
    indexes_api.return_value.list_indexes.side_effect = [
        SimpleNamespace(indexes=[]),
        SimpleNamespace(indexes=[vector_index(metric="cosine")]),
    ]
    fake_client.create_index = MagicMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("index already exists")
    )

    assert store.create_index() is None


def test_create_index_still_raises_when_the_build_really_failed(
    store: HotdataVectorStore,
    fake_client: FakeHotdataClient,
    indexes_api: MagicMock,
) -> None:
    """The engine's dimension-detection failure must not be read as a lost race."""
    fake_client.create_index = MagicMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("could not detect dimension for 'embedding'")
    )

    with pytest.raises(RuntimeError, match="could not detect dimension"):
        store.create_index()


def test_create_index_does_not_swallow_a_failure_that_left_a_pending_index(
    store: HotdataVectorStore,
    fake_client: FakeHotdataClient,
    indexes_api: MagicMock,
) -> None:
    """A listed-but-unbuilt index cannot be told from a lost race, so the error wins."""
    indexes_api.return_value.list_indexes.side_effect = [
        SimpleNamespace(indexes=[]),
        SimpleNamespace(indexes=[vector_index(status="pending")]),
    ]
    fake_client.create_index = MagicMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("build rejected")
    )

    with pytest.raises(RuntimeError, match="build rejected"):
        store.create_index()


def test_create_index_matches_the_metric_case_insensitively(
    store: HotdataVectorStore,
    fake_client: FakeHotdataClient,
    indexes_api: MagicMock,
) -> None:
    """The reported metric is the server's rendering; only `cosine` is verified verbatim."""
    indexes_api.return_value.list_indexes.return_value = SimpleNamespace(
        indexes=[vector_index(metric="COSINE")]
    )

    assert store.create_index() is None


def test_create_index_reports_an_index_whose_metric_is_unknown(
    store: HotdataVectorStore,
    indexes_api: MagicMock,
) -> None:
    indexes_api.return_value.list_indexes.return_value = SimpleNamespace(
        indexes=[vector_index(metric=None)]
    )

    with pytest.raises(ValueError, match="reports no metric"):
        store.create_index()


def test_index_lookup_translates_api_errors(
    store: HotdataVectorStore,
    indexes_api: MagicMock,
) -> None:
    """Every other failure on this class is a RuntimeError; this one was leaking raw."""
    indexes_api.return_value.list_indexes.side_effect = ApiException(status=403)

    with pytest.raises(RuntimeError):
        store.create_index()


def test_create_index_leaves_an_in_flight_build_alone(
    store: HotdataVectorStore,
    fake_client: FakeHotdataClient,
    indexes_api: MagicMock,
) -> None:
    """Start-up must not fight a build another process already started."""
    indexes_api.return_value.list_indexes.return_value = SimpleNamespace(
        indexes=[vector_index(status="pending")]
    )

    assert store.create_index() is None
    assert fake_client.indexes == []
