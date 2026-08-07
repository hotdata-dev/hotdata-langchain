from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hotdata_framework import ManagedDatabase, QueryResult


@pytest.fixture
def managed_db() -> ManagedDatabase:
    """A resolved managed database, as ``resolve_database_by_id`` returns one.

    Query scopes are resolved records rather than id strings, so tests pass this where
    an id would previously have been threaded through.
    """
    return ManagedDatabase(
        id="dbidsf000000000000000000000001",
        description="sf_airbnb",
        default_connection_id="connsf00000000000000000000001",
    )


@pytest.fixture
def databases_api(managed_db: ManagedDatabase) -> Iterator[MagicMock]:
    """Patch the raw databases API so an id lookup resolves to ``managed_db``.

    ``GET /databases/{id}`` is the only lookup the package is allowed to make, so tests
    stub it here and assert against the calls it received.
    """
    detail = SimpleNamespace(
        id=managed_db.id,
        name=managed_db.description,
        default_connection_id=managed_db.default_connection_id,
    )
    with patch("hotdata_langchain.databases.DatabasesApi") as api:
        api.return_value.get_database.return_value = detail
        yield api


def vector_index(
    name: str = "vectors_embedding_vector",
    metric: str | None = "cosine",
    status: str = "ready",
) -> SimpleNamespace:
    """An entry as ``IndexesApi.list_indexes`` reports one."""
    return SimpleNamespace(
        index_name=name,
        index_type="vector",
        columns=["embedding"],
        metric=metric,
        status=status,
    )


@pytest.fixture
def indexes_api() -> Iterator[MagicMock]:
    """Patch the raw indexes API, reporting no index until a test says otherwise.

    Index existence is not readable in SQL, so the store asks the control plane before
    building one. Tests set ``list_indexes.return_value`` to change what it finds.
    """
    with patch("hotdata_langchain.vectorstore.IndexesApi") as api:
        api.return_value.list_indexes.return_value = SimpleNamespace(indexes=[])
        yield api


@pytest.fixture
def sample_result() -> QueryResult:
    return QueryResult(
        columns=["n"],
        rows=[[1], [2]],
        row_count=2,
        result_id="res_1",
        query_run_id="run_1",
        execution_time_ms=12,
        warning=None,
        error_message=None,
    )


@pytest.fixture
def search_result() -> QueryResult:
    return QueryResult(
        columns=["description", "score"],
        rows=[["Cozy apartment with a view", 8.5], ["Cozy studio, great light", 4.25]],
        row_count=2,
        result_id="res_bm25",
        query_run_id="run_bm25",
        execution_time_ms=8,
        warning=None,
        error_message=None,
    )


@pytest.fixture
def mock_client(sample_result: QueryResult):
    client = MagicMock()
    client.workspace_id = "ws_test"
    client.execute_sql = MagicMock(return_value=sample_result)
    client.list_managed_databases = MagicMock(return_value=[])
    return client
