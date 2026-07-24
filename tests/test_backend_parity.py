"""Both cache backends must be interchangeable, or the benchmark compares two things.

Every test here runs against HotdataToolCache (over the same parquet-round-tripping fake
client tests/test_cache.py uses) and SqliteToolCache (over a real temp file), asserting
identical observable behaviour. Needs no credentials and no network.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import MagicMock

import pyarrow.parquet as pq
import pytest
from hotdata_framework import ManagedDatabase, QueryResult

from benchmarks.backends import (
    LayeredToolCache,
    SqliteToolCache,
    ToolCache,
    make_cache_key,
)
from hotdata_langchain.cache import MISS, HotdataToolCache, cached
from hotdata_langchain.tools import make_hotdata_tools

VERSION = "parity-v1"


@pytest.fixture
def fake_client():
    """A HotdataClient double that round-trips through real parquet writes and reads."""
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
        rows = [] if row is None else [[row["result_json"], row["created_at"]]]
        return QueryResult(
            columns=["result_json", "created_at"],
            rows=rows,
            row_count=len(rows),
            result_id=None,
            query_run_id=None,
            execution_time_ms=None,
        )

    client.load_managed_table.side_effect = _load_managed_table
    client.execute_sql.side_effect = _execute_sql
    client.store = store
    return client


@pytest.fixture(params=["hotdata", "sqlite"])
def backend(request, fake_client, tmp_path):
    """Each test body runs once per backend."""
    if request.param == "hotdata":
        return HotdataToolCache(fake_client, version=VERSION)
    return SqliteToolCache(tmp_path / "cache.db", version=VERSION)


# --- the contract itself -------------------------------------------------------------


def test_backend_satisfies_the_protocol(backend):
    assert isinstance(backend, ToolCache)


def test_keys_are_identical_across_backends(fake_client, tmp_path):
    hot = HotdataToolCache(fake_client, version=VERSION)
    lite = SqliteToolCache(tmp_path / "c.db", version=VERSION)
    args = {"sql": "SELECT 1", "limit": 10, "flag": None}
    assert hot.make_key("t", args) == lite.make_key("t", args)
    assert hot.make_key("t", args) == make_cache_key(VERSION, "t", args)


def test_key_is_stable_across_dict_ordering(backend):
    a = backend.make_key("t", {"x": 1, "y": 2})
    b = backend.make_key("t", {"y": 2, "x": 1})
    assert a == b


def test_key_changes_with_tool_args_and_version(fake_client, tmp_path):
    lite = SqliteToolCache(tmp_path / "c.db", version=VERSION)
    other = SqliteToolCache(tmp_path / "d.db", version="different")
    base = lite.make_key("t", {"x": 1})
    assert base != lite.make_key("other_tool", {"x": 1})
    assert base != lite.make_key("t", {"x": 2})
    assert base != other.make_key("t", {"x": 1})


# --- read/write behaviour -----------------------------------------------------------


def test_miss_returns_the_miss_sentinel(backend):
    assert backend.get(backend.make_key("t", {"a": 1})) is MISS


@pytest.mark.parametrize("value", ["a plain string", {"n": 1}, 42, [1, 2, 3], None, True, 1.5])
def test_round_trip_preserves_value_and_type(backend, value):
    key = backend.make_key("t", {"v": repr(value)})
    backend.set(key, tool_name="t", args={"v": repr(value)}, result=value)
    got = backend.get(key)
    assert got == value
    assert type(got) is type(value)


def test_none_is_a_hit_not_a_miss(backend):
    """The reason MISS exists: None is a legitimate cached value."""
    key = backend.make_key("t", {"a": "none-case"})
    backend.set(key, tool_name="t", args={"a": "none-case"}, result=None)
    assert backend.get(key) is None


def test_set_twice_overwrites(backend):
    key = backend.make_key("t", {"a": 1})
    backend.set(key, tool_name="t", args={"a": 1}, result="first")
    backend.set(key, tool_name="t", args={"a": 1}, result="second")
    assert backend.get(key) == "second"


def test_expired_entry_reads_as_a_miss(backend):
    key = backend.make_key("t", {"a": 1})
    backend.set(key, tool_name="t", args={"a": 1}, result="v")
    assert backend.get(key, ttl=timedelta(seconds=-1)) is MISS
    assert backend.get(key, ttl=timedelta(hours=1)) == "v"


def test_per_call_ttl_overrides_the_instance_default(fake_client, tmp_path):
    for cache in (
        HotdataToolCache(fake_client, version=VERSION, ttl=timedelta(seconds=-1)),
        SqliteToolCache(tmp_path / "ttl.db", version=VERSION, ttl=timedelta(seconds=-1)),
    ):
        key = cache.make_key("t", {"a": 1})
        cache.set(key, tool_name="t", args={"a": 1}, result="v")
        assert cache.get(key) is MISS
        assert cache.get(key, ttl=timedelta(hours=1)) == "v"


def test_non_json_args_do_not_break_key_derivation(backend):
    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    key = backend.make_key("t", {"obj": Opaque()})
    assert len(key) == 64
    backend.set(key, tool_name="t", args={"obj": Opaque()}, result="ok")
    assert backend.get(key) == "ok"


# --- behaviour through cached() ------------------------------------------------------


def test_cached_serves_the_second_call_without_rerunning(backend):
    calls = {"n": 0}

    def tool(q: str) -> str:
        calls["n"] += 1
        return f"result for {q}"

    wrapped = cached(tool, cache=backend, tool_name="tool")
    assert wrapped("x") == "result for x"
    assert wrapped("x") == "result for x"
    assert calls["n"] == 1


def test_cached_normalises_positional_and_keyword_calls(backend):
    calls = {"n": 0}

    def tool(a: int, b: int = 2) -> int:
        calls["n"] += 1
        return a + b

    wrapped = cached(tool, cache=backend, tool_name="tool")
    assert wrapped(1) == 3
    assert wrapped(1, 2) == 3
    assert wrapped(a=1, b=2) == 3
    assert calls["n"] == 1


def test_cached_distinguishes_different_args(backend):
    calls = {"n": 0}

    def tool(q: str) -> str:
        calls["n"] += 1
        return q.upper()

    wrapped = cached(tool, cache=backend, tool_name="tool")
    wrapped("a")
    wrapped("b")
    assert calls["n"] == 2


def test_cached_works_on_async_tools(backend):
    calls = {"n": 0}

    async def tool(q: str) -> str:
        calls["n"] += 1
        return f"async {q}"

    wrapped = cached(tool, cache=backend, tool_name="tool")

    async def run() -> None:
        assert await wrapped("x") == "async x"
        assert await wrapped("x") == "async x"

    asyncio.run(run())
    assert calls["n"] == 1


def test_cached_fails_open_when_the_backend_raises(backend, monkeypatch):
    """A broken cache must degrade to uncached, never to a broken tool call."""
    monkeypatch.setattr(
        backend, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("backend down"))
    )
    monkeypatch.setattr(
        backend, "set", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("backend down"))
    )
    calls = {"n": 0}

    def tool(q: str) -> str:
        calls["n"] += 1
        return "real result"

    wrapped = cached(tool, cache=backend, tool_name="tool")
    assert wrapped("x") == "real result"
    assert wrapped("x") == "real result"
    assert calls["n"] == 2  # never served from cache, never broken


def test_make_hotdata_tools_accepts_either_backend(backend):
    """Structural typing means a non-Hotdata backend already works here at runtime."""
    client = MagicMock()
    client.execute_sql.return_value = QueryResult(
        columns=["n"],
        rows=[[1]],
        row_count=1,
        result_id=None,
        query_run_id=None,
        execution_time_ms=None,
    )
    tools = make_hotdata_tools(client, cache=backend)
    sql_tool = next(t for t in tools if t.name == "hotdata_execute_sql")
    first = sql_tool.invoke({"sql": "SELECT 1"})
    second = sql_tool.invoke({"sql": "SELECT 1"})
    assert first == second
    assert client.execute_sql.call_count == 1


# --- the layered backend -------------------------------------------------------------


def test_layered_promotes_a_remote_hit_into_the_local_tier(fake_client, tmp_path):
    local = SqliteToolCache(tmp_path / "l1.db", version=VERSION)
    remote = HotdataToolCache(fake_client, version=VERSION)
    layered = LayeredToolCache(local, remote)

    key = layered.make_key("t", {"a": 1})
    remote.set(key, tool_name="t", args={"a": 1}, result="from-remote")

    assert local.get(key) is MISS
    assert layered.get(key) == "from-remote"
    assert layered.remote_hits == 1
    assert local.get(key) == "from-remote"  # promoted

    assert layered.get(key) == "from-remote"
    assert layered.local_hits == 1


def test_layered_writes_reach_both_tiers(fake_client, tmp_path):
    local = SqliteToolCache(tmp_path / "l1.db", version=VERSION)
    remote = HotdataToolCache(fake_client, version=VERSION)
    layered = LayeredToolCache(local, remote)

    key = layered.make_key("t", {"a": 1})
    layered.set(key, tool_name="t", args={"a": 1}, result={"v": 1})
    assert local.get(key) == {"v": 1}
    assert remote.get(key) == {"v": 1}


def test_layered_counts_a_full_miss(fake_client, tmp_path):
    layered = LayeredToolCache(
        SqliteToolCache(tmp_path / "l1.db", version=VERSION),
        HotdataToolCache(fake_client, version=VERSION),
    )
    assert layered.get(layered.make_key("t", {"a": 1})) is MISS
    assert layered.misses == 1


# --- properties specific to SQLite that motivate the comparison ----------------------


def test_sqlite_cache_survives_a_new_instance_on_the_same_file(tmp_path):
    path = tmp_path / "persist.db"
    first = SqliteToolCache(path, version=VERSION)
    key = first.make_key("t", {"a": 1})
    first.set(key, tool_name="t", args={"a": 1}, result="kept")
    first.close()

    second = SqliteToolCache(path, version=VERSION)
    assert second.get(key) == "kept"


def test_sqlite_memory_cache_is_invisible_to_another_instance():
    """The structural limitation the fleet benchmark measures."""
    first = SqliteToolCache(":memory:", version=VERSION)
    key = first.make_key("t", {"a": 1})
    first.set(key, tool_name="t", args={"a": 1}, result="local-only")

    second = SqliteToolCache(":memory:", version=VERSION)
    assert second.get(key) is MISS


def test_sqlite_stores_queryable_metadata_columns(tmp_path):
    cache = SqliteToolCache(tmp_path / "meta.db", version=VERSION)
    key = cache.make_key("my_tool", {"q": "hello"})
    cache.set(key, tool_name="my_tool", args={"q": "hello"}, result="hi")
    row = (
        cache._connect()
        .execute('SELECT tool_name, args_json FROM "tool_cache" WHERE cache_key = ?', (key,))
        .fetchone()
    )
    assert row[0] == "my_tool"
    assert "hello" in row[1]
    assert cache.row_count() == 1
