# Findings: Hotdata vs SQLite as a LangChain tool cache

Measured 2026-07-25 against a live workspace (`api.hotdata.dev`, us-west-2) and real TPCH
sf=1 data. Reproduce with `uv run python -m benchmarks.run_all`.

**Read the caveat on geography first.** The benchmark machine is ~265 ms round-trip from
us-west-2. Every remote number below is dominated by that. A benchmark run inside the same
region would compress the remote column substantially — but not the structural conclusions,
which are about *counts of round trips*, not their length.

## The short answer

> **Time spent in network round trips + time spent in pure retrieval — who wins?**

SQLite, on both, by a margin that isn't close:

| | Hotdata | SQLite |
|---|---|---|
| Network round trips per hit | 2 | 0 |
| Time on the network | 619.0 ms | 0 |
| Time in actual retrieval | ~55 ms server-side, 0.9 ms client-side | 5.2 µs |
| **Total, p50** | **619.9 ms** | **5.2 µs** |

**119,008× ratio on a cache hit.** But the ratio isn't the interesting part. This is:

**99.9% of a Hotdata cache hit is time on a socket.** 0.1% — 905 µs — is client-side work.
There is almost no "retrieval" to compete on; retrieval is already fast at both ends. The
entire contest is whether you pay a network round trip at all.

> **But that comparison is not a level playing field**, and the 119,008× number should not be
> quoted on its own. A local file against a hosted platform measures the distance to us-west-2
> as much as it measures the two designs. See
> [Removing geography](#removing-geography-the-fair-comparison) for the network-free version,
> which is much less lopsided — and is the number that actually generalises.

## The finding that matters more

A cache can only ever save the *compute*, never the *network*. So the decisive number is
not how fast a cache hit is — it's how much of an uncached call was compute in the first
place. For Hotdata's own queries, almost none of it:

| | Total | Network + engine floor | Actual compute |
|---|---|---|---|
| `SELECT 1` | 564.5 ms | 564.5 ms | — |
| TPCH Q1 (6M-row aggregation) | 682.0 ms | 564.5 ms | **117.5 ms** (17%) |
| TPCH Q5 (6-way join) | 873.2 ms | 564.5 ms | **308.7 ms** (35%) |
| Heaviest probe (4-way join + agg) | 925.4 ms | 564.5 ms | **360.9 ms** |

Verified honest: Q1 returns the canonical sf=1 cardinalities, and re-running with a varied
literal (which filters no rows) is the same speed — so there's no server-side result cache
making the baseline secretly cached.

**A Hotdata cache hit (620 ms) costs more than the TPCH Q1 it caches (682 ms) saves.** Nothing
we could find at sf=1 has compute exceeding a single round trip. Caching a Hotdata query in
Hotdata replaces a network round trip with a network round trip.

This inverts the Phase 1 conclusion. See "Why the Phase 1 numbers don't reproduce" below.

## Removing geography: the fair comparison

The engine reports its own `execution_time_ms` per query, so we can compare what a cache
lookup costs the engine against what the query it replaces costs the engine, with the network
excluded entirely. That is a claim about architecture, not about where the benchmark ran
(`bench_serverside`, medians of two runs):

| Operation | Server-side | Wall clock | Network share of wall |
|---|---|---|---|
| `SELECT 1` — engine floor | 20–24 ms | ~575–695 ms | 97% |
| Cache lookup — `SELECT ... WHERE cache_key = ?` | **71–79 ms** | ~685 ms | 88–90% |
| TPCH Q1 — 6M-row aggregation | 98–101 ms | ~680–700 ms | 85–86% |
| TPCH Q5 — 6-way join | 261–274 ms | ~835–860 ms | 67–70% |

**The ceiling for a perfectly co-located deployment**, i.e. the best a Hotdata tool cache can
ever do even at zero network latency:

| Query cached | Engine cost | Cache-lookup cost | Best possible speedup |
|---|---|---|---|
| TPCH Q1 | 98–101 ms | 71–79 ms | **1.28–1.38×** |
| TPCH Q5 | 261–274 ms | 71–79 ms | **3.47–3.68×** |

Two things follow, and they cut in opposite directions:

1. **The SQLite advantage was substantially geographic**, and shrinks a lot on a level field.
   Co-located, a Hotdata cache hit is a genuine 3.5× on Q5 — a real win, not overhead.
2. **But there is a hard architectural floor**: a cache lookup costs the engine 71–79 ms,
   which is **+55 ms above the `SELECT 1` floor on a table holding only 18 rows.** That is not
   scan cost — it's fixed per-query overhead on a managed parquet table (planning, catalog,
   parquet open). SQLite does the same single-key lookup in **86 µs**.

That 55 ms floor is the number to attack, because it is what caps the co-located case. Any
query whose compute is below ~75 ms cannot be usefully cached in Hotdata at any distance —
and Q1, a 6M-row aggregation, is only just above it.

### Result size makes it worse, not better

The intuition would be that a bigger result is more worth caching. The opposite holds, because
the cache read has to ship those bytes back over the same link:

| Cached rows | Bytes | Hotdata `set` | Hotdata `get` | SQLite `set` | SQLite `get` |
|---|---|---|---|---|---|
| 1 | 71 | 8.97 s | 756 ms | 636 µs | 86 µs |
| 100 | 7.4 KB | 8.61 s | 893 ms | 781 µs | 134 µs |
| 1,000 | 75 KB | 9.25 s | 1.34 s | 2.33 ms | 875 µs |
| 10,000 | 776 KB | 9.66 s | **2.64 s** | 10.2 ms | 4.97 ms |

`set` is flat because it's dominated by the 5-round-trip write path, not the payload. `get`
degrades 3.5× from 1 to 10,000 rows. Note that `make_hotdata_tools` defaults to
`max_rows=100`, so in the shipped configuration a cached result is capped at 100 rows —
the larger rows show what raising that cap would cost.

### The deployment that actually ships

A local SQLite file next to a developer's laptop isn't the product. A LangChain app runs on
some production backend, and Hotdata is hosted — so **the network hop is real and unavoidable
in production**. What varies is only its size. Holding the measured server-side costs fixed
and sweeping the round trip (`crossover.py`, `rtt_sweep`):

| Deployment | RTT | Cache hit | Q1 gain | Q5 gain |
|---|---|---|---|---|
| Same cluster / sidecar | 0.5 ms | 76 ms | 1.33× | **3.53×** |
| Same AWS region | 2 ms | 79 ms | 1.32× | **3.43×** |
| Same continent, different region | 15 ms | 105 ms | 1.24× | 2.83× |
| Cross-continent | 80 ms | 235 ms | 1.11× | 1.82× |
| This benchmark machine → us-west-2 | 265 ms | 605 ms | 1.04× | 1.32× |

So for a **normally deployed** app — same region as the workspace — the Hotdata tool cache is
worth **~3.4× on an expensive query** and ~1.3× on a cheap one. That is a real result, and it
is the number to quote, not the laptop one.

A local cache still serves any of these in ~0.1 ms, so it wins on latency at every distance.
What it cannot do is be **shared**, and that is where the production reality cuts the other
way:

- **Multi-replica backend** (the normal case): each replica has its own file, so the hit rate
  falls towards 1/N and a warm replica can't help a cold one.
- **Serverless / ephemeral** (Lambda, Cloud Run, Vercel): the file dies with the invocation, so
  a local cache is close to useless across requests.
- **Redeploys**: a local cache starts cold every release; a shared one doesn't.

That is the honest position for Hotdata: **the shared, durable, queryable L2 — not the fast
path.** `LayeredToolCache` is the shape that follows, with a per-replica L1 for latency.

### Which makes Redis, not SQLite, the real competitor

If the job is "shared cache tier for an agent backend," the incumbent is Redis or Memcached,
which serve a keyed GET in well under 1 ms plus RTT. Against that, Hotdata's **71–79 ms
managed-table lookup is the whole competitive gap** — roughly 40–75× more server-side work than
the alternative, on an 18-row table.

Closing it is the single highest-leverage change here, and it's already on the roadmap:
`docs/ai-native-layer-roadmap.md` tracks generalising `PointLookupProvider::fetch_by_keys`
(hotdata-langchain#34). A keyed point lookup in single-digit ms would make Hotdata competitive
as a shared tool-cache tier; at 75 ms it is not.

Hotdata's differentiators in that comparison are real but are *not* latency: the cache is
queryable SQL data (`SELECT tool_name, count(*) FROM tool_cache GROUP BY 1` for hit-rate
analysis), it needs no extra infrastructure, and it already sits where the customer's data is.

## Measured constants

Primitives, p50 (`bench_primitives`):

| Operation | Hotdata | SQLite |
|---|---|---|
| `get()` hit | 619.9 ms — 2 round trips | 5.2 µs |
| `get()` hit, resolve memoized | 350.6 ms — 1 round trip | — |
| `get()` miss | 618.2 ms | 1.9 µs |
| `set()` | 3.9–8.4 s — 5 round trips | 15.2 µs (`synchronous=NORMAL`) |
| `set()`, durable | — | 37.3 µs (`synchronous=FULL`) |
| First `set()` incl. setup | 8.4 s — 7 requests, 2 handshakes | ~0 |

`set()` is highly variable because the final managed-table load ranged 2.0–5.2 s across runs.
Treat the write cost as a wide band, not a point.

End-to-end through the real `StructuredTool`, 12 calls alternating Q1/Q5, one long-lived
process (`bench_endtoend`):

| Arm | Cold total | Warm total | Per warm call | Warm speedup |
|---|---|---|---|---|
| uncached | 9.45 s | 9.45 s | 787 ms | 1.00× |
| **sqlite** | **1.59 s** | **~0.001 s** | **105 µs** | **~7,500×** |
| hotdata (as shipped) | 22.97 s | 7.77 s | 648 ms | 1.22× |
| hotdata + resolve memoized | 19.13 s | 4.61 s | 384 ms | 2.05× |
| layered (sqlite → hotdata) | 14.66 s | ~0.004 s | 292 µs | ~2,700× |

Note the cold column: **the Hotdata cache is 2.4× slower than no cache at all** over these
12 calls (22.97 s vs 9.45 s), because two cold writes cost ~16 s of the total.

## Three actionable defects found

### 1. Every cache hit spends an avoidable round trip — 43% is recoverable for free

`execute_sql(sql, database=X)` calls `resolve_managed_database(X)` unconditionally, and that
is an uncached `GET /v1/databases/{id}` — even when `X` is already a resolved database ID,
which is exactly what `HotdataToolCache.get()` passes. Every hit pays a full round trip to
re-learn something it already knows:

```
GET  /v1/databases/dbidu20k...    359.78ms     <- pure waste
POST /v1/query                    346.94ms
```

Memoizing the resolve: **619.9 ms → 350.6 ms, a 43% cut**, one round trip instead of two.
This is in `hotdata_framework`, so it benefits every SDK caller, not just the cache.

### 2. `set()` has no retry, so transient failures silently drop cache entries

The object-storage `PUT` inside `upload_parquet` failed intermittently under back-to-back
writes, twice during this work:

```
RuntimeError: the upload transfer to storage failed before any response
```

`HotdataToolCache.set()` isn't retried — only `_ensure_ready()` is wrapped in
`_retry_transient`. And because `cached()` fails open, a dropped write produces a warning
and then a permanent silent cache miss for that key. Correctness is fine; the cache just
quietly doesn't work. `_retry_transient` already exists and should wrap the write.

### 3. A fresh process pays ~2 s before its first hit, which breaks the fleet case

`bench_fleet`, 4 fresh processes making the same call:

| Arm | work = 0.12 s | work = 3.0 s | Ran the work | HTTP/worker | Handshakes |
|---|---|---|---|---|---|
| no cache | 0.49 s | 12.01 s | 4/4 | 0 | 0 |
| sqlite, per-process file | 0.50 s | 12.02 s | 4/4 | 0 | 0 |
| **sqlite, shared file (one host)** | **0.12 s** | **3.01 s** | **1/4** | 0 | 0 |
| hotdata (shared, cross-host) | 19.45 s | 21.97 s | 1/4 | 5–10 | 2–4 |
| layered | 19.51 s | 22.89 s | 1/4 | 5–10 | 2–4 |

The shared caches do their job — only 1 of 4 workers ran the real work. It doesn't help.
A fresh worker's "cache hit" costs **~2.6 s**, not 620 ms, because it also pays the auth
token exchange and 2–4 TCP+TLS handshakes. So for short-lived workers the Hotdata cache is
**worse than no cache** unless the work costs well over ~3 s.

Note that this is the scenario Hotdata's cache is *supposed* to win — cross-host sharing is
the one thing SQLite structurally cannot do. It loses anyway, on per-process setup cost.
Fixing that (a reusable token, a warm connection pool, pinned `database_id`) is what would
make the shared-cache story real.

## Where each option actually wins

From `crossover.py`, over the measured constants:

**Break-even: how expensive must the work be before the Hotdata cache beats no cache?**
(long-lived process, setup amortised away)

| Repeats | Work must exceed |
|---|---|
| 2 | 4.64 s |
| 3 | 2.63 s |
| 10 | 1.06 s |
| 100 | 0.66 s |
| ∞ | **0.62 s** (0.35 s if the resolve is memoized) |

That asymptote is the whole story: **a remote cache can never help work cheaper than one
round trip, no matter how often it repeats.** SQLite's equivalent asymptote is 5 µs, which
is why it is never the wrong choice on latency alone.

So:

- **Caching Hotdata's own queries** — SQLite wins outright. Hotdata's engine is fast enough
  (117 ms for a 6M-row aggregation) that a remote cache is pure overhead.
- **Caching expensive external work** (a slow third-party API, an LLM call, a scrape) in
  **one long-lived process** — SQLite still wins; it's strictly cheaper on both hit and write.
- **Caching expensive external work across a fleet of hosts** — this is Hotdata's only
  structural advantage, since a local file isn't visible to another host. It needs the work
  to cost more than ~3 s *and* the per-process setup cost fixed before it pays off.
- **Best of both: two tiers.** `LayeredToolCache` (local L1, Hotdata L2) gets the shared hit
  rate at local latency after one hit per process — ~2,700× on the warm end-to-end run
  versus 1.22× for Hotdata alone.

## Why the Phase 1 numbers don't reproduce

The Phase 1 benchmark reported **3.5× faster (28.07 s → 8.08 s)**. Re-running the same shape
today gives **1.22×**. The cached column is consistent between the two runs (~0.6–0.7 s per
hit, then and now). What changed is the **uncached baseline**:

| | Phase 1 (2026-07-20) | Today (2026-07-25) |
|---|---|---|
| Q1 uncached | 2.5–2.7 s | 0.68 s |
| Q5 uncached | 1.7–1.9 s | 0.88 s |
| Cache hit | 0.6–0.7 s | 0.62 s |

I can't attribute that from here — a platform/engine change, different network conditions,
or engine warmth would all look like this, and the Phase 1 run wasn't instrumented at the
round-trip level. What I can say is that today's numbers were all measured in one session
with both arms re-measured, and the baseline was explicitly checked for the two ways it
could lie (server-side result cache; empty results). The Phase 1 speedup was real against
the baseline it measured; it is not the baseline that exists now.

**The lesson is methodological**: the Phase 1 run compared a cache against a baseline, but
never measured how much of either was network. Had it done so, a 0.6 s hit against a 0.56 s
round-trip floor would have shown immediately that there was almost nothing to win.

## Recommendations for PR #33

1. **Retry `set()`** with the existing `_retry_transient`. Cheap, and it fixes a silent
   correctness-of-caching bug.
2. **Skip the redundant resolve** when `database` is already an ID (fix in
   `hotdata_framework`, or have the cache call the query API directly). 43% off every hit.
3. **Rescope the README claim.** It currently says the cache serves "repeated calls to the
   read-only tools". On these numbers that is a ~1.2× win at best and a net loss cold. The
   honest scoping is *expensive external work, repeated, from a long-lived process* — and
   explicitly not Hotdata's own queries.
4. **Ship the `ToolCache` protocol.** `cached()` and `make_hotdata_tools()` annotate the
   concrete `HotdataToolCache`, so no alternative backend type-checks even though they all
   work at runtime. `benchmarks/backends.py` has the protocol; widening those two
   annotations is a two-line change.
5. **Consider shipping `LayeredToolCache`.** It's the only configuration in this benchmark
   that beats both alternatives, and it makes Hotdata the durable shared tier rather than
   competing with a local file on latency.
6. **Fix per-process setup** before claiming the fleet case. Until a fresh worker's first hit
   costs ~600 ms instead of ~2.6 s, the cross-host advantage is theoretical.

## Contract gap noticed while building this

`get()` receives only a key, but `set()` requires `tool_name` and `args`. A tiered cache
therefore cannot promote a remote hit into its local tier with faithful metadata — the
promoted row carries a placeholder `tool_name` and empty `args`. The cached value is exact;
only the local tier's debug columns degrade. Returning an entry object from `get()`, or
making those two `set()` parameters optional, would close it.
