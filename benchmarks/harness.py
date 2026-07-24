"""Timing, HTTP round-trip accounting, and output formatting for the benchmarks."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import urllib3.connectionpool


class HttpCounter:
    """Count and time every HTTP request made inside the context.

    Patches ``HTTPConnectionPool.urlopen``, which both the generated ``hotdata`` API client
    and the object-storage upload bottom out at. This is the only way to see hops a single
    SDK call hides -- ``execute_sql`` resolves the database before querying, and
    ``load_managed_table`` uploads, finalizes, then loads.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float, bool]] = []

    def __enter__(self) -> HttpCounter:
        self._orig = urllib3.connectionpool.HTTPConnectionPool.urlopen
        counter = self

        def patched(pool: Any, method: str, url: str, *a: Any, **kw: Any) -> Any:
            before = pool.num_connections
            start = time.perf_counter()
            try:
                return counter._orig(pool, method, url, *a, **kw)
            finally:
                counter.calls.append(
                    (method, url, time.perf_counter() - start, pool.num_connections > before)
                )

        urllib3.connectionpool.HTTPConnectionPool.urlopen = patched  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: Any) -> None:
        urllib3.connectionpool.HTTPConnectionPool.urlopen = self._orig  # type: ignore[method-assign]

    @property
    def n(self) -> int:
        """Number of HTTP requests made."""
        return len(self.calls)

    @property
    def socket_secs(self) -> float:
        """Total wall time spent inside HTTP requests."""
        return sum(c[2] for c in self.calls)

    @property
    def new_connections(self) -> int:
        """How many requests had to open a fresh connection (TCP + TLS handshake)."""
        return sum(1 for c in self.calls if c[3])

    def dump(self, indent: str = "     ") -> None:
        for method, url, took, new in self.calls:
            flag = "  NEW CONN (TCP+TLS)" if new else ""
            print(f"{indent}{method:<6} {url[:62]:<62} {fmt(took)}{flag}")


def timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    """Run ``fn`` and return ``(seconds, result)``."""
    start = time.perf_counter()
    out = fn()
    return time.perf_counter() - start, out


# Counts transient failures retried away, so a flaky run reports rather than hides them.
RETRIED: list[str] = []


def timed_retry(fn: Callable[[], Any], attempts: int = 3) -> tuple[float, Any]:
    """Like ``timed``, but retry on exception and time only the successful attempt.

    The managed-table write path fails intermittently at the object-storage PUT
    ("the upload transfer to storage failed before any response"). ``HotdataToolCache.set``
    does not retry -- only ``_ensure_ready`` is wrapped -- so in real use a blip silently
    drops the entry, because ``cached()`` fails open. Here it would abort the benchmark, so
    retry and record it.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return timed(fn)
        except Exception as e:
            last = e
            RETRIED.append(f"{type(e).__name__}: {e}")
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"failed after {attempts} attempts: {last}") from last


def report_retries() -> None:
    """Print anything ``timed_retry`` had to retry, so the run's health is visible."""
    if not RETRIED:
        return
    print(f"\n  NOTE: {len(RETRIED)} transient failure(s) retried during this run:")
    for msg in RETRIED:
        print(f"    - {msg}")


def stats(samples: list[float]) -> dict[str, float]:
    """Summarise a list of durations."""
    s = sorted(samples)
    return {
        "n": len(s),
        "min": s[0],
        "p50": statistics.median(s),
        "p95": s[min(len(s) - 1, int(0.95 * (len(s) - 1)))],
        "max": s[-1],
        "mean": statistics.fmean(s),
    }


def fmt(secs: float) -> str:
    """Format a duration with a unit that keeps it readable across 5 orders of magnitude."""
    if secs < 0.001:
        return f"{secs * 1e6:8.1f}us"
    if secs < 1:
        return f"{secs * 1e3:8.2f}ms"
    return f"{secs:8.3f}s "


def show(label: str, st: dict[str, float]) -> None:
    """Print one summarised timing row."""
    print(
        f"  {label:<36} n={int(st['n']):<6} min={fmt(st['min'])}  "
        f"p50={fmt(st['p50'])}  p95={fmt(st['p95'])}  max={fmt(st['max'])}"
    )


def header(title: str, width: int = 100) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def memoize_resolve(client: Any) -> Callable[[], None]:
    """Memoize ``resolve_managed_database`` on ``client``; returns an undo callable.

    ``execute_sql(sql, database=X)`` resolves ``X`` on every call via an uncached
    ``GET /v1/databases/{id}``, even when ``X`` is already a resolved database ID -- which
    is exactly what ``HotdataToolCache`` passes. That is a whole round trip per cache hit
    spent re-learning something already known. This measures what removing it is worth.
    """
    original = client.resolve_managed_database
    store: dict[str, Any] = {}

    def memoized(name_or_id: str) -> Any:
        if name_or_id not in store:
            store[name_or_id] = original(name_or_id)
        return store[name_or_id]

    client.resolve_managed_database = memoized

    def undo() -> None:
        client.resolve_managed_database = original

    return undo
