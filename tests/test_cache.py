from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import MagicMock

import pyarrow.parquet as pq
import pytest
from hotdata_framework import ManagedDatabase, QueryResult

from hotdata_langchain.cache import MISS, HotdataToolCache, cached
from hotdata_langchain.tools import make_hotdata_tools


@pytest.fixture
def fake_client():
    """A HotdataClient double backed by an in-memory dict, exercising real
    serialization (pyarrow parquet write/read) on every set()/get() round trip."""
    store: dict[str, dict[str, object]] = {}

    client = MagicMock()
    client.resolve_managed_database.side_effect = KeyError("not found")
    client.create_managed_database.return_value = ManagedDatabase(
        id="db_1", description="langchain_tool_cache", default_connection_id="conn_1"
    )
    client.add_managed_table.return_value = None

    def _load_managed_table(database, table, *, schema, file, mode, key):
        row = pq.read_table(file).to_pylist()[0]
        store[row["cache_key"]] = row

    def _execute_sql(sql, *, database=None):
        cache_key = sql.split("cache_key = '")[1].split("'")[0]
        row = store.get(cache_key)
        if row is None:
            return QueryResult(
                columns=["result_json", "created_at"],
                rows=[],
                row_count=0,
                result_id=None,
                query_run_id=None,
                execution_time_ms=None,
            )
        return QueryResult(
            columns=["result_json", "created_at"],
            rows=[[row["result_json"], row["created_at"]]],
            row_count=1,
            result_id=None,
            query_run_id=None,
            execution_time_ms=None,
        )

    client.load_managed_table.side_effect = _load_managed_table
    client.execute_sql.side_effect = _execute_sql
    client.store = store
    return client


@pytest.fixture
def cache(fake_client):
    return HotdataToolCache(fake_client)


def test_make_key_deterministic_regardless_of_dict_order(cache):
    key_a = cache.make_key("t", {"a": 1, "b": 2})
    key_b = cache.make_key("t", {"b": 2, "a": 1})
    assert key_a == key_b


def test_make_key_differs_on_args_or_tool(cache):
    base = cache.make_key("t", {"a": 1})
    assert base != cache.make_key("t", {"a": 2})
    assert base != cache.make_key("other", {"a": 1})


def test_get_miss_when_absent(cache):
    key = cache.make_key("t", {"a": 1})
    assert cache.get(key) is MISS


@pytest.mark.parametrize("value", ["a plain string", {"n": 1}, 42, [1, 2, 3], None])
def test_set_then_get_round_trips_exact_type(cache, value):
    key = cache.make_key("t", {"a": 1})
    cache.set(key, tool_name="t", args={"a": 1}, result=value)
    got = cache.get(key)
    assert got == value
    assert type(got) is type(value)


def test_ttl_expiry_treated_as_miss(cache):
    key = cache.make_key("t", {"a": 1})
    cache.set(key, tool_name="t", args={"a": 1}, result="cached-value")
    assert cache.get(key, ttl=timedelta(seconds=-1)) is MISS


def test_declared_key_used_for_upsert(cache, fake_client):
    key = cache.make_key("t", {"a": 1})
    cache.set(key, tool_name="t", args={"a": 1}, result="v")
    _, kwargs = fake_client.load_managed_table.call_args
    assert kwargs["mode"] == "upsert"
    assert kwargs["key"] == ["cache_key"]


def test_cached_calls_underlying_fn_once_then_serves_hits(cache):
    calls = {"n": 0}

    def fn(x: int) -> str:
        calls["n"] += 1
        return f"value-{x}"

    wrapped = cached(fn, cache=cache, tool_name="fn")
    assert wrapped(1) == "value-1"
    assert wrapped(1) == "value-1"
    assert wrapped(x=1) == "value-1"
    assert calls["n"] == 1

    assert wrapped(2) == "value-2"
    assert calls["n"] == 2


def test_cached_async_variant(cache):
    calls = {"n": 0}

    async def fn(x: int) -> str:
        calls["n"] += 1
        return f"value-{x}"

    wrapped = cached(fn, cache=cache, tool_name="fn")

    async def run() -> None:
        assert await wrapped(1) == "value-1"
        assert await wrapped(1) == "value-1"

    asyncio.run(run())
    assert calls["n"] == 1


def test_cached_fails_open_on_backend_errors():
    broken_cache = MagicMock()
    broken_cache.make_key.return_value = "k"
    broken_cache.get.side_effect = RuntimeError("network blip")
    broken_cache.set.side_effect = RuntimeError("network blip")

    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        return "real result"

    wrapped = cached(fn, cache=broken_cache, tool_name="fn")
    assert wrapped() == "real result"
    assert calls["n"] == 1


def test_make_hotdata_tools_only_wraps_read_tools(mock_client):
    cache_mock = MagicMock()
    cache_mock.make_key.return_value = "k"
    cache_mock.get.return_value = MISS

    tools = {t.name: t for t in make_hotdata_tools(mock_client, cache=cache_mock)}

    tools["hotdata_execute_sql"].invoke({"sql": "select 1"})
    tools["hotdata_list_managed_databases"].invoke({})
    assert cache_mock.make_key.call_count == 2
    assert mock_client.execute_sql.call_count == 1
    assert mock_client.list_managed_databases.call_count == 1

    mock_client.create_managed_database.return_value = ManagedDatabase(
        id="c1", description="sales", default_connection_id="conn_c1"
    )
    tools["hotdata_create_managed_database"].invoke({"name": "sales", "tables": "orders"})
    # Mutating tools never consult the cache.
    assert cache_mock.make_key.call_count == 2
    assert mock_client.create_managed_database.call_count == 1


def test_make_hotdata_tools_without_cache_is_unaffected(mock_client):
    tools = {t.name: t for t in make_hotdata_tools(mock_client)}
    tools["hotdata_execute_sql"].invoke({"sql": "select 1"})
    tools["hotdata_execute_sql"].invoke({"sql": "select 1"})
    assert mock_client.execute_sql.call_count == 2
