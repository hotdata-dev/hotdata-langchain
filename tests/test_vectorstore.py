"""Behaviour the LangChain conformance suite does not reach.

The suite in ``test_vectorstore_standard.py`` certifies the id/CRUD contract. It never
looks at the SQL, the stored column types, filtering, scores, or ``from_texts`` — which
is where a Hotdata-specific implementation can be wrong while still conforming.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
from hotdata_framework import ManagedDatabase
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import VectorStore

from hotdata_langchain.vectorstore import HotdataVectorStore
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
