"""Cache backends compared against ``HotdataToolCache``.

``cached()`` requires only three methods of a cache object, so any backend implementing
them drops in unchanged. This module supplies two:

``SqliteToolCache``
    A local SQLite file. Pays no network cost, so the gap between it and
    ``HotdataToolCache`` isolates exactly what the network is costing us.

``LayeredToolCache``
    A fast local tier in front of a shared remote tier.

``ToolCache`` is the structural contract itself, written down as a ``Protocol``. The
package currently annotates ``cached(cache: HotdataToolCache)`` with the concrete class,
so a second backend does not type-check against it -- widening that annotation to this
protocol is the one source change a pluggable backend needs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from hotdata_langchain.cache import MISS, _json_default, _parse_timestamp

_KEY_COLUMN = "cache_key"

_DDL = """
CREATE TABLE IF NOT EXISTS "{table}" (
    cache_key   TEXT PRIMARY KEY,
    tool_name   TEXT NOT NULL,
    args_json   TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
)
"""


@runtime_checkable
class ToolCache(Protocol):
    """What ``cached()`` and ``make_hotdata_tools(cache=...)`` actually require."""

    def make_key(self, tool_name: str, args: Mapping[str, Any]) -> str: ...

    def get(self, key: str, *, ttl: timedelta | None = None) -> Any: ...

    def set(self, key: str, *, tool_name: str, args: Mapping[str, Any], result: Any) -> None: ...


def make_cache_key(version: str, tool_name: str, args: Mapping[str, Any]) -> str:
    """Compute a cache key the same way ``HotdataToolCache.make_key`` does.

    Kept byte-identical on purpose: a key computed for one backend must address the same
    entry in the other, or a cross-backend comparison is not measuring the same thing.
    """
    payload = json.dumps(
        {"v": version, "tool": tool_name, "args": args},
        sort_keys=True,
        default=_json_default,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SqliteToolCache:
    """Cache tool results in a local SQLite file.

    Mirrors ``HotdataToolCache``'s surface exactly: same keyword-only signatures, same
    ``MISS`` sentinel, same key scheme, same symmetric JSON round trip for type fidelity,
    and the same policy of raising on failure so ``cached()`` is the thing that fails open.

    Args:
        path: SQLite file to use. ``":memory:"`` works but is per-connection, so it is
            invisible to other processes -- which is the limitation being measured.
        table: Table holding cache entries.
        ttl: Default maximum entry age. ``None`` means entries never expire.
        version: Folded into every key, so bumping it invalidates the whole cache.
        synchronous: SQLite ``synchronous`` pragma. ``NORMAL`` is the usual WAL setting;
            ``FULL`` fsyncs on every commit.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        table: str = "tool_cache",
        ttl: timedelta | None = None,
        version: str = "v1",
        synchronous: Literal["FULL", "NORMAL", "OFF"] = "NORMAL",
    ) -> None:
        self._path = str(path)
        self._table = table
        self._ttl = ttl
        self._version = version
        self._synchronous = synchronous
        self._local = threading.local()
        self._connect().execute(_DDL.format(table=self._table))

    def _connect(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA synchronous={self._synchronous}")
            self._local.conn = conn
        return conn

    def make_key(self, tool_name: str, args: Mapping[str, Any]) -> str:
        """Derive a deterministic cache key from a tool name and its bound arguments."""
        return make_cache_key(self._version, tool_name, args)

    def get(self, key: str, *, ttl: timedelta | None = None) -> Any:
        """Return the cached value for ``key``, or ``MISS`` if absent or expired."""
        row = (
            self._connect()
            .execute(
                f'SELECT result_json, created_at FROM "{self._table}" WHERE {_KEY_COLUMN} = ?',
                (key,),
            )
            .fetchone()
        )
        if row is None:
            return MISS
        effective_ttl = ttl if ttl is not None else self._ttl
        if effective_ttl is not None:
            created_at = _parse_timestamp(row[1])
            if datetime.now(timezone.utc) - created_at > effective_ttl:
                return MISS
        return json.loads(row[0])

    def set(self, key: str, *, tool_name: str, args: Mapping[str, Any], result: Any) -> None:
        """Store ``result`` under ``key``, replacing any existing entry."""
        self._connect().execute(
            f'INSERT INTO "{self._table}" '
            f"({_KEY_COLUMN}, tool_name, args_json, result_json, created_at) "
            f"VALUES (?, ?, ?, ?, ?) "
            f"ON CONFLICT({_KEY_COLUMN}) DO UPDATE SET "
            f"result_json = excluded.result_json, created_at = excluded.created_at",
            (
                key,
                tool_name,
                json.dumps(dict(args), default=_json_default),
                json.dumps(result),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def row_count(self) -> int:
        """Number of entries currently stored."""
        cur = self._connect().execute(f'SELECT count(*) FROM "{self._table}"')
        return int(cur.fetchone()[0])

    def close(self) -> None:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


class LayeredToolCache:
    """Read a fast local tier first, fall back to a shared remote tier, write both.

    Neither tier alone is right: a local-only cache starts empty in every new worker and
    re-pays the full work cost there, while a remote-only cache pays a network round trip
    on every hit forever. Reading local-first and promoting remote hits into it gives the
    remote tier's hit rate at the local tier's latency, after one hit per process.

    Note a genuine gap in the contract this conforms to: ``get()`` receives only a key,
    but ``set()`` requires ``tool_name`` and ``args``. So a remote hit cannot be promoted
    into the local tier with faithful metadata -- the promoted row carries a placeholder
    ``tool_name`` and empty ``args``. The cached value itself is exact; only the local
    tier's debug columns are degraded.
    """

    PROMOTED = "<promoted-from-remote>"

    def __init__(self, local: ToolCache, remote: ToolCache) -> None:
        self._local = local
        self._remote = remote
        self.local_hits = 0
        self.remote_hits = 0
        self.misses = 0

    def make_key(self, tool_name: str, args: Mapping[str, Any]) -> str:
        """Derive a key from the remote tier, so keys stay portable across tiers."""
        return self._remote.make_key(tool_name, args)

    def get(self, key: str, *, ttl: timedelta | None = None) -> Any:
        """Return the cached value for ``key``, consulting local then remote."""
        hit = self._local.get(key, ttl=ttl)
        if hit is not MISS:
            self.local_hits += 1
            return hit
        hit = self._remote.get(key, ttl=ttl)
        if hit is not MISS:
            self.remote_hits += 1
            self._local.set(key, tool_name=self.PROMOTED, args={}, result=hit)
            return hit
        self.misses += 1
        return MISS

    def set(self, key: str, *, tool_name: str, args: Mapping[str, Any], result: Any) -> None:
        """Store ``result`` in both tiers."""
        self._local.set(key, tool_name=tool_name, args=args, result=result)
        self._remote.set(key, tool_name=tool_name, args=args, result=result)

    def counters(self) -> str:
        return f"local_hits={self.local_hits} remote_hits={self.remote_hits} misses={self.misses}"
