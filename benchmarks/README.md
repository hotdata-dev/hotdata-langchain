# Hotdata vs SQLite as a LangChain tool-result cache

`HotdataToolCache` (PR #33) caches LangChain tool results in a Hotdata managed table. But
nothing about that design is Hotdata-specific — the same thing works over SQLite, or any
key/value store. This suite answers the question that raises: **of the latency a cache hit
costs, how much is network round trip and how much is retrieval — and who actually wins?**

The comparison is only meaningful if both backends are interchangeable, so `SqliteToolCache`
implements the identical three-method contract, with a byte-identical key scheme.
`tests/test_backend_parity.py` runs the same 20-odd behavioural assertions against both and
requires no credentials.

> **This branch sits on top of `feat/tool-result-caching`** (draft PR #33), because every module
> here imports `hotdata_langchain.cache` — `MISS`, `HotdataToolCache`, `cached`, and the
> `_json_default` / `_parse_timestamp` helpers. It therefore runs as-is, but it carries #33's
> commit as its base and is not intended to merge on its own.
>
> Parked deliberately: tool caching looks like a smaller win for Hotdata-on-LangChain than
> first thought — see `FINDINGS.md` for why, and the "which makes Redis, not SQLite, the real
> competitor" section for where the leverage actually is.

## Layout

| File | What it does |
|---|---|
| `backends.py` | `SqliteToolCache`, `LayeredToolCache`, and the `ToolCache` protocol |
| `tpch.py` | Canonical TPCH Q1/Q5, heavier probe queries, fixture helpers |
| `provision_tpch.py` | Generates TPCH with DuckDB and loads it. Idempotent |
| `harness.py` | Timing, HTTP round-trip accounting, formatting |
| `bench_querycost.py` | What does the work cost? Is the baseline honest? |
| `bench_primitives.py` | Where does a cache hit's latency go? |
| `bench_endtoend.py` | Through the real `StructuredTool`, every arm, cold and warm |
| `bench_fleet.py` | Short-lived worker processes: shared vs local cache |
| `crossover.py` | A model over the measured constants. No network needed |
| `run_all.py` | Runs everything, writes a transcript |

## Running it

Needs `HOTDATA_API_KEY` (and optionally `HOTDATA_WORKSPACE`) in the environment.

```bash
set -a; source .env; set +a

# once -- creates tpch_sf1 and langchain_tool_cache, skips if already present
uv run --with duckdb python -m benchmarks.provision_tpch

uv run python -m benchmarks.run_all --out results.txt

# or individually
uv run python -m benchmarks.bench_primitives
uv run python -m benchmarks.bench_fleet --workers 6 --work-secs 0.12 3.0 10.0
uv run python -m benchmarks.crossover --workers 20 --work-secs 5
```

The parity tests run in the normal suite, no credentials needed:

```bash
uv run pytest tests/test_backend_parity.py -v
```

## Methodology notes

Things that would otherwise quietly invalidate the numbers, and what the suite does about
each:

- **`cached()` fails open.** A silently broken backend looks like a fast arm, not an error.
  `bench_endtoend.py` installs a log handler on `hotdata_langchain.cache` and marks any arm
  where a cache operation was swallowed.
- **A server-side result cache** would make the "uncached" baseline secretly cached.
  `bench_querycost.py` re-runs each query with a literal varied (`l_quantity > -1-i`, which
  filters nothing) and compares. Same latency ⇒ no result cache.
- **A query returning nothing looks fast.** Q1's group cardinalities are asserted against
  the known sf=1 answer, so "fast" can't mean "empty".
- **Hidden round trips.** One `execute_sql` is not one HTTP request. `HttpCounter` patches
  `urllib3`'s `urlopen`, so every hop is counted and attributed, including handshakes.
- **Cold start.** RuntimeDB scales to zero under KEDA, and the first call in a process also
  pays auth plus TLS. Every benchmark warms up before timing and reports the cold cost
  separately — except `bench_fleet.py`, where paying it per process *is* the measurement.
- **Geography is a variable, not a constant.** Every remote number here is dominated by the
  round-trip time from the machine running the benchmark to `api.hotdata.dev` (us-west-2).
  Measure your own before reusing these: `curl -w '%{time_connect}\n' -o /dev/null -s
  https://api.hotdata.dev/healthz`.

## Findings

See `FINDINGS.md` in this directory for the measured results and what they imply.
