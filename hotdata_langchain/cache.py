"""Cache LangChain tool call results in a Hotdata managed table."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar, cast

import pyarrow as pa
import pyarrow.parquet as pq
from hotdata_framework import DEFAULT_SCHEMA, HotdataClient
from hotdata_framework.errors import HotdataTransientError, classify_sdk_error

logger = logging.getLogger(__name__)

_KEY_COLUMN = "cache_key"
_KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")

F = TypeVar("F", bound=Callable[..., Any])


class _Miss:
    def __repr__(self) -> str:
        return "<MISS>"


MISS = _Miss()
"""Sentinel returned by :meth:`HotdataToolCache.get` on a miss or expiry.

Distinct from ``None`` because a cached tool result may legitimately be ``None``.
"""


class HotdataToolCache:
    """Store LangChain tool results in a Hotdata managed table, keyed by (tool name, args).

    Backed by a managed table with a declared key column (``cache_key``), written via
    ``load_managed_table(..., mode="upsert")`` and read via ordinary SQL. Reads and writes
    raise on backend failure — this class does not fail open. Callers that want cache
    failures to degrade gracefully (recommended when wrapping a tool call) should catch
    around :meth:`get`/:meth:`set`; see :func:`cached`.
    """

    def __init__(
        self,
        client: HotdataClient,
        *,
        database: str = "langchain_tool_cache",
        database_id: str | None = None,
        table: str = "tool_cache",
        schema: str = DEFAULT_SCHEMA,
        ttl: timedelta | None = None,
        version: str = "v1",
    ) -> None:
        """Configure the cache backend.

        Pass ``database_id`` to pin an already-created managed database by id instead of
        resolving/creating one by name. Recommended for any multi-process deployment:
        managed-database names are not unique or identifying, so concurrent first-use
        across processes can each create a distinct database with the same name, silently
        splitting the cache. ``version`` is folded into every cache key; bump it to
        invalidate all existing entries at once (e.g. after changing what a tool returns).
        """
        self._client = client
        self._database_name = database
        self._database_id = database_id
        self._table = table
        self._schema = schema
        self._ttl = ttl
        self._version = version
        self._resolved_database_id: str | None = None

    def make_key(self, tool_name: str, args: Mapping[str, Any]) -> str:
        """Derive a deterministic cache key from a tool name and its bound arguments."""
        payload = json.dumps(
            {"v": self._version, "tool": tool_name, "args": args},
            sort_keys=True,
            default=_json_default,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str, *, ttl: timedelta | None = None) -> Any:
        """Return the cached value for ``key``, or :data:`MISS` if absent or expired."""
        _validate_key(key)
        self._ensure_ready()
        sql = (
            f'SELECT result_json, created_at FROM "default"."{self._schema}"."{self._table}" '
            f"WHERE {_KEY_COLUMN} = '{key}'"
        )
        result = self._client.execute_sql(sql, database=self._resolved_database_id)
        rows = result.to_records()
        if not rows:
            return MISS
        effective_ttl = ttl if ttl is not None else self._ttl
        if effective_ttl is not None:
            created_at = _parse_timestamp(rows[0]["created_at"])
            if datetime.now(timezone.utc) - created_at > effective_ttl:
                return MISS
        return json.loads(cast(str, rows[0]["result_json"]))

    def set(self, key: str, *, tool_name: str, args: Mapping[str, Any], result: Any) -> None:
        """Store ``result`` under ``key``, upserting by ``cache_key``."""
        _validate_key(key)
        self._ensure_ready()
        row = pa.table(
            {
                _KEY_COLUMN: [key],
                "tool_name": [tool_name],
                "args_json": [json.dumps(dict(args), default=_json_default)],
                "result_json": [json.dumps(result)],
                "created_at": [datetime.now(timezone.utc)],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "entry.parquet"
            pq.write_table(row, path)  # type: ignore[no-untyped-call]
            self._client.load_managed_table(
                cast(str, self._resolved_database_id),
                self._table,
                schema=self._schema,
                file=str(path),
                mode="upsert",
                key=[_KEY_COLUMN],
            )

    def _ensure_ready(self) -> None:
        # Memoized per instance — construct one long-lived HotdataToolCache per process
        # and reuse it. A fresh instance re-resolves (and re-races, see __init__'s
        # database_id note) on every construction.
        if self._resolved_database_id is not None:
            return
        _retry_transient(self._resolve_and_declare)

    def _resolve_and_declare(self) -> None:
        if self._database_id is not None:
            self._resolved_database_id = self._database_id
        else:
            try:
                db = self._client.resolve_managed_database(self._database_name)
            except KeyError:
                db = self._client.create_managed_database(
                    description=self._database_name,
                    schema=self._schema,
                    tables=[self._table],
                    keys={self._table: [_KEY_COLUMN]},
                )
            self._resolved_database_id = db.id
        # Best-effort: declare the table if this cache's database already existed
        # without it (e.g. shared with another table= value, or created concurrently
        # by another process without our table). There's no dedicated "does this table
        # exist" signal, so a failure here is assumed to mean it's already declared;
        # any real problem (bad auth, etc.) surfaces clearly on the get/set that follows.
        try:
            self._client.add_managed_table(
                self._resolved_database_id,
                self._table,
                schema=self._schema,
                key=[_KEY_COLUMN],
            )
        except Exception:
            logger.debug(
                "add_managed_table for %s.%s did not create a new table",
                self._schema,
                self._table,
                exc_info=True,
            )


def cached(
    fn: F,
    *,
    cache: HotdataToolCache,
    tool_name: str,
    ttl: timedelta | None = None,
) -> F:
    """Wrap ``fn`` so repeated calls with the same arguments are served from ``cache``.

    Works on any plain function, not just this package's own tools — pass the name you
    want it cached under via ``tool_name``. Cache backend failures never propagate: a
    failed lookup is treated as a miss and a failed write is dropped, so caching can only
    make a tool call faster, never break one that would otherwise succeed.
    """
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = _bind_args(fn, args, kwargs)
            key = cache.make_key(tool_name, bound)
            hit = _safe_get(cache, key, ttl)
            if hit is not MISS:
                return hit
            result = await fn(*args, **kwargs)
            _safe_set(cache, key, tool_name, bound, result)
            return result

        return cast(F, async_wrapper)

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = _bind_args(fn, args, kwargs)
        key = cache.make_key(tool_name, bound)
        hit = _safe_get(cache, key, ttl)
        if hit is not MISS:
            return hit
        result = fn(*args, **kwargs)
        _safe_set(cache, key, tool_name, bound, result)
        return result

    return cast(F, sync_wrapper)


def _bind_args(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    bound = inspect.signature(fn).bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _safe_get(cache: HotdataToolCache, key: str, ttl: timedelta | None) -> Any:
    try:
        return cache.get(key, ttl=ttl)
    except Exception:
        logger.warning("hotdata tool cache read failed; treating as a miss", exc_info=True)
        return MISS


def _safe_set(
    cache: HotdataToolCache, key: str, tool_name: str, args: dict[str, Any], result: Any
) -> None:
    try:
        cache.set(key, tool_name=tool_name, args=args, result=result)
    except Exception:
        logger.warning("hotdata tool cache write failed; result was not cached", exc_info=True)


def _validate_key(key: str) -> None:
    if not _KEY_PATTERN.fullmatch(key):
        raise ValueError(f"invalid cache key: {key!r}")


def _json_default(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            pass
    return str(value)


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _retry_transient(fn: Callable[[], None], *, attempts: int = 3, base_delay: float = 0.5) -> None:
    for attempt in range(attempts):
        try:
            fn()
            return
        except Exception as exc:
            if attempt == attempts - 1 or not _is_transient(exc):
                raise
            time.sleep(base_delay * (attempt + 1))


def _is_transient(exc: BaseException) -> bool:
    cause = exc.__cause__
    if not isinstance(cause, Exception):
        return False
    try:
        return isinstance(classify_sdk_error(cause), HotdataTransientError)
    except Exception:
        return False
