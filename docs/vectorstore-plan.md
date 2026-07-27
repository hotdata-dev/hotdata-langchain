# `HotdataVectorStore` — implementation plan

Status: plan / not yet built. Not committed — decide later whether this belongs in the repo
long-term or stays a local reference doc (mirrors the convention used in
`hotdata-dlt-destination/docs/vector-search-exploration.md`).

## Problem & positioning

Hotdata is working with LangChain on deeper ecosystem integration. Phase 1 of that work — `HotdataToolCache`/`cached()`, a Hotdata-backed cache for
arbitrary LangChain tool calls — shipped as draft PR #33. The team has validated two
directions coming out of that: "Hotdata as a tool for LangChain" (the existing 4-tool
foundation in `make_hotdata_tools`) and "Tool caching" (Phase 1 itself).

`VectorStore`/RAG is the next priority, chosen specifically because it converges with Rohan's
own recent vector-search engineering:

- **`datafusion-vector-search-ext`** — the DataFusion extension that makes USearch HNSW ANN
  search a first-class SQL operator (`ORDER BY <distance_fn>(col, query) LIMIT k`, transparently
  rewritten into an index lookup). PR #31 (merged 2026-07-21) fixed the optimizer rule to see
  through `SubqueryAlias` nodes, which is what makes the fast path reachable from SQL generated
  by anything that aliases tables (ibis, ORMs, BI tools).
- **`runtimedb`** — the deployed query engine; PR #953 bumped its pin to pick up #31. **Merged
  and confirmed live in production** — the fast path is no longer pending.
- **`hotdata-ibis`** — gets its own read-side vector helper layer (a `semantic_search()` +
  distance-UDF module), planned and owned separately by Rohan; this plan cross-references it
  but doesn't depend on it.
- **`hotdata-dlt-destination`** — already flows `list<float32>` embedding columns through its
  write path untouched; a differentiated auto-embed-on-ingest adapter is a separate, later
  piece.

Put together, these converge on one story: one fast, DataFusion-backed engine, addressable
from SQL, ibis, and now LangChain's own `VectorStore` primitive — not three disconnected
integrations.

**Bigger-picture context (not yet planned, understanding-only as of 2026-07-22):** the team
has separately articulated a longer-term vision of Hotdata as an "AI-native query layer" for
LangChain — a single tool that routes across SQL, full-text, vector, and point-lookup
pathways with its own query planning and permissions, rather than several discrete tools the
agent picks between. `HotdataVectorStore` (this doc) and `HotdataToolCache` are partial
building blocks toward that vision (the SQL and caching pieces, plus this doc's vector
pathway) — not the vision itself. That larger design is intentionally not scoped here; it's
tracked separately until it moves from understanding to planning.

**This document covers `HotdataVectorStore` only** — a new class in `hotdata_langchain`. It
does not cover the `hotdata-ibis` helper or the `hotdata-dlt-destination` adapter in
implementation detail; see "Cross-repo dependency tracking" below for how those relate.

## `HotdataVectorStore` design

### File and constructor

New file: `hotdata_langchain/vectorstore.py` (sibling to `cache.py`, not folded into
`databases.py`). Constructor mirrors `HotdataToolCache`'s `database`/`database_id`/`table`/`schema`
pattern from `cache.py`:

```python
HotdataVectorStore(
    client: HotdataClient,
    embedding: Embeddings,
    *,
    database: str = "langchain_vectorstore",
    database_id: str | None = None,
    table: str = "vectors",
    schema: str = DEFAULT_SCHEMA,
    distance: Literal["cosine", "l2", "dot"] = "cosine",
    metadata_columns: Mapping[str, Literal["string", "int", "float", "bool"]] | None = None,
)
```

`embedding` is held on `self`, not passed per-call — the universal LangChain convention, and
what lets `similarity_search(self, query, k=4, **kwargs)` match the ABC's fixed signature.

### Storage schema

One managed table, key = `["id"]` (enables `mode="upsert"`/`"delete"` exactly like
`HotdataToolCache`'s `cache_key` pattern):

| column | type | purpose |
|---|---|---|
| `id` | `string` | LangChain doc id / managed-table key |
| `content` | `string` | `page_content` |
| `metadata_json` | `string` | full metadata dict (`json.dumps(..., default=_json_default)`), always kept in full for read-back fidelity |
| `embedding` | `list<float32>` | confirmed to round-trip through `load_managed_table` via a live spike in a sibling repo |
| *(promoted metadata columns)* | typed per `metadata_columns` | denormalized copy of declared metadata keys, so `WHERE` can target a real typed column — see Filtering below |

### Methods (verified against installed `langchain_core==1.4.0` source, not docs)

Only `similarity_search` and `from_texts` are truly `@abstractmethod` on `VectorStore`.
Everything else has a default or raises `NotImplementedError` until overridden.

- **`add_texts`** (implement; `add_documents` derives for free from it, confirmed via the base
  class's own delegation check). `self._embedding.embed_documents(texts)` → one pyarrow table
  (id/content/metadata_json/embedding/promoted columns) → temp parquet →
  `client.load_managed_table(..., mode="upsert", key=["id"])`. Same shape as
  `HotdataToolCache.set()`. Generate ids via `uuid.uuid4().hex` when omitted — never `None` (the
  key column can't be null).
- **`similarity_search` / `similarity_search_by_vector` / `similarity_search_with_score(_by_vector)`**
  — implement all explicitly rather than relying on ABC defaults. See "SQL-path decision" below
  for the query shape.
- **`_select_relevance_score_fn`** — mapped off `self._distance`. Default constructor value is
  `cosine` specifically because its score function (`1 - distance`) needs no scale assumption;
  see the `l2` caveat below.
- **`get_by_ids(ids)`** — `WHERE id IN (...)`, no vector math involved; the simplest method,
  built first.
- **`delete(ids=None, **kwargs)`** — **requires** `ids` (raises if omitted; no "delete
  everything" in v1, mirroring `HotdataToolCache`'s stance of never exposing an unbounded
  destructive operation). Backed by `load_managed_table(..., mode="delete", key=["id"])`.
  Raises on backend failure — deletes do **not** fail open (unlike the cache's fail-open
  policy: silently reporting a delete succeeded when it didn't is actively dangerous, a cache
  miss is not).
- **`from_texts(cls, texts, embedding, metadatas=None, *, ids=None, **kwargs)`** — classmethod;
  `client` threaded through `**kwargs` (the ABC's sanctioned per-implementation extension
  point, same pattern every real integration uses for constructor args the ABC can't
  standardize). Builds the store, calls `add_texts`, returns it. Index creation (if requested)
  happens strictly after `add_texts` — see "Dimension binding" below.
- **MMR (`max_marginal_relevance_search_by_vector`)** — **not free.** The ABC raises
  `NotImplementedError` by default (confirmed by reading `InMemoryVectorStore`, LangChain's own
  reference implementation) — every real implementation fetches `fetch_k` candidates *with
  their raw embedding vectors* and runs
  `langchain_core.vectorstores.utils.maximal_marginal_relevance`. This needs its own query
  branch that *does* select the `embedding` column, which breaks the "never surface the vector
  column" rule the primary read path relies on for the engine's fast-path rewrite (engine issue
  #508) — so this branch is always brute-force by design. Acceptable: `fetch_k` defaults small
  and is caller-bounded, so a full scan over a bounded candidate set is cheap. Own phase, own
  PR — a distinct correctness surface (raw-vector round-trip on read, which the primary path
  never needs), not bundled with the MVP.
- Everything else (`add_documents`, async variants via thread-pool wrapping, `as_retriever()`,
  `similarity_search_with_relevance_scores`) is free from the base class — verified by tests
  that they delegate correctly, no new code required.

**Internal plumbing**: reuse `HotdataToolCache`'s `_ensure_ready()`/`_resolve_and_declare()`
pattern verbatim — resolve-or-create the managed database, best-effort `add_managed_table` with
`key=["id"]`, swallow "already declared" failures at `logger.debug`.

### SQL-path decision

**Build every read query as a scalar-UDF `ORDER BY ... LIMIT`, not the `vector_search_vector(...)`
table function:**

```sql
SELECT id, content, metadata_json, <promoted cols>,
       <distance_fn>(embedding, ARRAY[...]) AS dist
FROM "default"."<schema>"."<table>"
[WHERE <promoted_col> = <literal>]
ORDER BY dist ASC
LIMIT <k>
```

using the engine's index-independent scalar distance UDFs (`cosine_distance`, `l2_distance`,
`negative_dot_product` — confirmed to work as plain row-by-row functions even with **no index
at all**, always correct, just a full-table brute-force scan without one).

Why this over the table function: this shape is correct from row one with zero
preconditions, and it transparently upgrades to the HNSW fast path the moment a
matching-metric index exists on that column — one code path, no index-vs-no-index branching
to build or test. The `vector_search_vector(...)` table function, by contrast, errors loudly
("no loaded vector index") if the index doesn't exist yet, which would make a freshly
constructed `HotdataVectorStore` unusable out of the box — a bad default for a
partnership-facing integration. The raw `embedding` column is never selected in this path
(engine issue #508: a vector column in the output declines the fast-path rewrite).

### Metadata filtering (v1 scope)

Equality filters only, and only on keys explicitly declared via the constructor's
`metadata_columns` (which promotes them to real typed columns at write time).
`filter={"key": value}` on a key not in `metadata_columns` raises `ValueError` immediately —
fail loudly at call time, not silently-wrong at query time. Free-form/undeclared metadata keys
are simply not filterable in v1.

Filter predicates always go in the *same* query, in `WHERE`, ahead of `ORDER BY`/`LIMIT` —
never as an outer query wrapping an already-computed top-k result, which would silently
return fewer than `k` rows (a filter applied after top-k selection can only shrink the result,
never re-fill it). Whether a `WHERE`-filtered query still triggers the HNSW fast path is
explicitly **unverified** (attribute-filtered ANN is a harder capability many engines don't
support natively) — brute-force-but-correct is an accepted v1 cost, not a blocker, consistent
with the engine's own "brute force is always correct, just not accelerated" design.

Ids and filter literals are charset-validated before SQL interpolation (mirroring `cache.py`'s
`_KEY_PATTERN` philosophy — reject anything outside a conservative charset rather than
attempt general SQL escaping). The query vector itself is never user-controlled text; it's a
list of floats we format ourselves.

### Dimension binding

A vector's dimension is only knowable after the first `embed_documents` call, but
`create_index` needs `dimensions` up front. Sequencing inside `from_texts`: (1) construct the
store, (2) call `add_texts` (embeds and writes rows — dimension is now known from the vectors
just embedded), (3) only then, if `create_index=True` was requested, create the index with the
now-known dimension. Index creation never precedes the first write.

### `l2` relevance-score caveat

The engine's `l2_distance` is **squared** L2 (no `sqrt`), but `VectorStore`'s default
`_euclidean_relevance_score_fn` assumes true (unsquared) Euclidean distance on
unit-normalized embeddings — using `l2` as the configured metric would produce a
wrong-scale relevance score unless corrected, and correcting it properly requires knowing
embedding normalization we don't control. Constructor defaults to `cosine` for this reason
(`1 - distance` is exact, no scale assumption); `l2`/`dot` remain available but flagged in the
docstring rather than silently "fixed."

## Testing strategy

No live embedding-provider credentials are available in this repo's `.env` (Hotdata
credentials only). Unit tests mirror `tests/test_cache.py`'s fixture style exactly: a fake
`HotdataClient` (`MagicMock`) backed by an in-memory dict, where `load_managed_table` does a
*real* `pq.read_table(file).to_pylist()` (exercising the `list<float32>` round-trip `cache.py`
never needed) and `execute_sql` does real SQL-shape parsing rather than a bare mocked return
value. For embeddings, use `langchain_core.embeddings.DeterministicFakeEmbedding` (confirmed
present in the installed `langchain_core`, zero new dependency) rather than a bespoke fake.

Coverage: schema/type correctness on write; exact SQL shape on read (distance aliased,
embedding column absent from `SELECT`, filter predicate inside the same query); `ValueError`
on an undeclared filter key or a malformed id; `delete` requiring `ids`; `from_texts`
round-tripping end to end; MMR selecting the embedding column and calling into
`maximal_marginal_relevance` with the right shapes.

**Live verification** (once real credentials or a local cluster are available — not part of
this repo's CI): round-trip `add_texts`/`similarity_search` against a real workspace with a
real embedding; `EXPLAIN` the primary query before and after provisioning a matching-metric
index, confirming the plan shows the USearch-rewritten node. Since `runtimedb` PR #953 is now
merged and live in production, this is no longer blocked on a pending deploy — it's now
directly verifiable once `HotdataVectorStore` itself exists, rather than a claim asserted from
a sibling repo's spike. `EXPLAIN` a `WHERE`-filtered query to settle whether filtered queries
still hit the fast path.

## Phasing

1. **MVP** — `add_texts`, `similarity_search(_by_vector)`,
   `similarity_search_with_score(_by_vector)`, `get_by_ids`, `delete`, `from_texts` (no
   self-provisioned index yet), promoted-column equality filtering, full unit-test suite,
   `examples/langchain_vectorstore.py`, README section, CHANGELOG entry. A complete, correct,
   mergeable `VectorStore` on its own — `as_retriever()`, chains, and evals all work once this
   lands, independent of anything below.
2. **MMR** — its own PR; isolated correctness surface (the raw-vector read path).
3. **Self-provisioning** — gated on the `sdk-python` dependency below. Adds
   `from_texts(..., create_index=True)`.
4. Docs/examples are pulled into Phase 1 rather than deferred to the end — this is the piece
   the LangChain conversation will exercise first.

## Cross-repo dependency tracking

- **`sdk-python` (`hotdata_framework.HotdataClient`) — in scope, ours to build.** A
  `create_vector_index` addition. Confirmed via full grep: zero existing index-related code in
  the package today. The raw generated `hotdata.api.indexes_api.IndexesApi.create_index` +
  `CreateIndexRequest` already support everything needed (`columns`, `metric`, `dimensions`,
  `embedding_provider_id`, `output_column`, an async job-polling path). Follows
  `create_managed_database`'s exact shape: resolve `database` → `default_connection_id`, build
  the request, call the raw API, wrap `ApiException` → `RuntimeError(api_error_message(e))`,
  return a frozen dataclass built field-by-field from `IndexInfoResponse` (or poll
  `SubmitJobResponse.status_url` for the async path, reusing the existing polling-loop style
  already in `client.py`). This addition only blocks Phase 3 (self-provisioning) above —
  Phases 1–2 work today regardless, against an existing index, a not-yet-existing index, or no
  index ever, by construction of the SQL-path decision above.
- **`runtimedb` PR #953 — merged and live in production.** Pin-bump to pick up
  `datafusion-vector-search-ext` PR #31, confirmed deployed. Our SQL was designed to be
  correct either way (the scalar UDFs work with no index at all) — this just means the HNSW
  fast path is now actually live for any query matching the contract above, not merely
  pending. No longer a dependency to track.
- **`hotdata-ibis` vector helper layer — external, owned separately, tracked for consistency
  only.** Not a dependency of this work. Cross-referenced so both surfaces target the same
  engine contract (same distance-function names, same "never select the vector column"
  constraint) rather than drifting apart.
- **`hotdata-dlt-destination` write-side embedding adapter — external, future, tracked.**
  Still a design sketch (`hotdata_adapter(data, embed=[...])`). Once built, completes the full
  pipeline this plan enables end to end: dlt ingests and auto-embeds on the way in →
  `HotdataVectorStore` reads it out for a LangChain agent. Not a blocker for this work — this
  plan's MVP works against precomputed embeddings loaded any way (including today's dlt
  destination, unchanged).
