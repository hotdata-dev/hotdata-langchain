from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.request import Request

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hotdata_framework import ManagedDatabase, QueryResult


@pytest.fixture
def parquet_file(tmp_path: Path) -> Path:
    """A real parquet file on disk, so a load reads real magic bytes rather than a stub."""
    path = tmp_path / "orders.parquet"
    table = pa.table({"id": [1, 2, 3], "label": ["a", "b", "c"]})
    pq.write_table(table, path)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    return path


class FakeResponse(io.BytesIO):
    """What an opener returns, as far as a streaming download is concerned."""

    def __init__(self, payload: bytes, headers: dict[str, str]) -> None:
        super().__init__(payload)
        self.headers = headers


class FakeOpener:
    def __init__(self, payload: bytes, headers: dict[str, str]) -> None:
        self._payload = payload
        self._headers = headers
        self.requests: list[Request] = []

    def open(self, request: Request, timeout: float | None = None) -> FakeResponse:
        self.requests.append(request)
        return FakeResponse(self._payload, self._headers)


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every host to a public address.

    The download tests are about what happens to the bytes, not about DNS, and a test
    that really resolved a name would depend on the network. Tests covering the address
    check itself set their own resolution.
    """
    monkeypatch.setattr(
        "hotdata_langchain.databases.socket.getaddrinfo",
        lambda host, port: [(0, 0, 0, "", ("93.184.216.34", 0))],
    )


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch, public_dns: None) -> Callable[..., FakeOpener]:
    """Answer the next fetch with this payload, and hand back the opener that saw it."""

    def _serve(payload: bytes, headers: dict[str, str] | None = None) -> FakeOpener:
        opener = FakeOpener(payload, headers or {})
        monkeypatch.setattr("hotdata_langchain.databases.build_opener", lambda *handlers: opener)
        return opener

    return _serve


@pytest.fixture
def managed_db() -> ManagedDatabase:
    """A resolved instant database, as ``resolve_database_by_id`` returns one.

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
    columns: list[str] | None = None,
    source_column: str | None = None,
) -> SimpleNamespace:
    """An entry as ``IndexesApi.list_indexes`` reports one.

    ``source_column`` set is a provider-backed index: the engine embedded that text
    column into ``columns[0]`` and will embed a query to match. Left unset it is a plain
    index over a column that already held vectors, which the caller must query with a
    vector of its own.
    """
    return SimpleNamespace(
        index_name=name,
        index_type="vector",
        columns=columns if columns is not None else ["embedding"],
        metric=metric,
        status=status,
        source_column=source_column,
    )


def bm25_index(
    name: str = "listings_description_bm25",
    column: str = "description",
    status: str = "ready",
) -> SimpleNamespace:
    """A BM25 entry as ``IndexesApi.list_indexes`` reports one.

    Carries ``metric=None`` and ``source_column=None`` because that is what the server
    sends for a text index, and the routing reads both.
    """
    return SimpleNamespace(
        index_name=name,
        index_type="bm25",
        columns=[column],
        metric=None,
        status=status,
        source_column=None,
    )


@pytest.fixture
def search_indexes() -> Iterator[MagicMock]:
    """Patch the indexes API the search routing reads, reporting none by default.

    Separate from ``indexes_api`` because the two patch different modules: the vector
    store imports ``IndexesApi`` into its own namespace and this reads it through
    ``hotdata_langchain.indexes``. A test that patched only one would leave the other
    live against a MagicMock client.
    """
    with patch("hotdata_langchain.indexes.IndexesApi") as api:
        api.return_value.list_indexes.return_value = SimpleNamespace(indexes=[])
        yield api


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
def mock_client(sample_result: QueryResult) -> MagicMock:
    client = MagicMock()
    client.workspace_id = "ws_test"
    client.execute_sql = MagicMock(return_value=sample_result)
    client.list_managed_databases = MagicMock(return_value=[])
    return client
