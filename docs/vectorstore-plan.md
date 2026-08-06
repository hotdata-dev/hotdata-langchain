# `HotdataVectorStore` — implementation plan

Status: plan / not yet built. Kept in-repo while the AI-native-layer work is in flight; it can
move out once the roadmap is delivered.

This plan is **self-contained**. It depends on no unmerged branch and no parked work.

## Problem & positioning

Hotdata is working with LangChain on deeper ecosystem integration, on top of the existing tool
foundation in `make_hotdata_tools` (SQL, managed databases, BM25 search, schema discovery).

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
agent picks between. `HotdataVectorStore` is one building block toward that vision — the vector
pathway — not the vision itself. That larger design is intentionally not scoped here; see
`docs/ai-native-layer-roadmap.md` and issue #39, which covers the agent-facing *tool* surface
for semantic search and rank fusion. **#39 and this plan are different surfaces**: #39 wraps
`vector_search()` as a tool the model calls; this is LangChain's `VectorStore` primitive for
retrievers and chains. They share the engine contract, not the code.

**This document covers `HotdataVectorStore` only** — a new class in `hotdata_langchain`. It
does not cover the `hotdata-ibis` helper or the `hotdata-dlt-destination` adapter in
implementation detail; see "Cross-repo dependency tracking" below for how those relate.

## `HotdataVectorStore` design

### File and constructor

New file: `hotdata_langchain/vectorstore.py`, not folded into `databases.py`.

```python
HotdataVectorStore(
    client: HotdataClient,
    embedding: Embeddings,
    *,
    database_id: str | ManagedDatabase,       # REQUIRED — id, never a name
    table: str = "vectors",
    schema: str = DEFAULT_SCHEMA,
    distance: Literal["cosine", "l2", "dot"] = "cosine",
    metadata_columns: Mapping[str, Literal["string", "int", "float", "bool"]] | None = None,
)
```

**`database_id` is required and id-addressed** (issue #38, shipped in 0.3.0). The store never
creates a database implicitly and never resolves one by name — the caller creates it and passes
the id, the same stance the plan already takes on `delete` (never expose an unbounded
destructive operation). An already-resolved `ManagedDatabase` is accepted so a caller holding
one pays no lookup.

That single `resolve_database_by_id(client, database_id)` call at construction is the **only**
lookup in the class. Every subsequent query and load addresses the resolved `ManagedDatabase`
record, so id-addressing propagates throughout by construction.

`embedding` is held on `self`, not passed per-call — the universal LangChain convention, and
what lets `similarity_search(self, query, k=4, **kwargs)` match the ABC's fixed signature.

### Storage schema

One managed table, key = `["id"]`, which is what enables `mode="upsert"` and `mode="delete"`
on `load_managed_table`:

| column | type | purpose |
|---|---|---|
| `id` | `string` | LangChain doc id / managed-table key |
| `content` | `string` | `page_content` |
| `metadata_json` | `string` | full metadata dict (`json.dumps(..., default=_json_default)`), always kept in full for read-back fidelity |
| `embedding` | `list<float32>` | confirmed to round-trip through `load_managed_table` via a live spike in a sibling repo |
| *(promoted metadata columns)* | typed per `metadata_columns` | denormalized copy of declared metadata keys, so `WHERE` can target a real typed column — see Filtering below |

### Methods (verified against installed `langchain_core` source, not docs; re-confirmed on 1.5.1)

Only `similarity_search` and `from_texts` are truly `@abstractmethod` on `VectorStore`.
Everything else has a default or raises `NotImplementedError` until overridden.

- **`add_texts`** (implement; `add_documents` derives for free from it, confirmed via the base
  class's own delegation check). `self._embedding.embed_documents(texts)` → one pyarrow table
  (id/content/metadata_json/embedding/promoted columns) → temp parquet →
  `client.load_managed_table(self._db, table, schema=..., file=..., mode="upsert", key=["id"])`,
  passing the resolved record. Generate ids via `uuid.uuid4().hex` when omitted — never `None`
  (the key column can't be null).
- **`similarity_search` / `similarity_search_by_vector` / `similarity_search_with_score(_by_vector)`**
  — implement all explicitly rather than relying on ABC defaults. See "SQL-path decision" below
  for the query shape.
- **`_select_relevance_score_fn`** — mapped off `self._distance`. Default constructor value is
  `cosine` specifically because its score function (`1 - distance`) needs no scale assumption;
  see the `l2` caveat below.
- **`get_by_ids(ids)`** — `WHERE id IN (...)`, no vector math involved; the simplest method,
  built first.
- **`delete(ids=None, **kwargs)`** — **requires** `ids` (raises if omitted; no "delete
  everything" in v1 — never expose an unbounded destructive operation). Backed by `load_managed_table(..., mode="delete", key=["id"])`.
  Raises on backend failure — deletes do **not** fail open: silently reporting a delete
  succeeded when it didn't is actively dangerous.
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

**Internal plumbing** (self-contained, no shared base class needed):

1. **Construction** — `self._db = resolve_database_by_id(client, database_id)`. Raises `KeyError`
   for an unknown id, so a bad id fails at construction rather than on first search.
2. **Table declaration** — best-effort `client.add_managed_table(self._db, table, schema=schema,
   key=["id"])`, swallowing an "already declared" failure at `logger.debug`. The key is what
   makes `mode="upsert"`/`"delete"` work; a keyless table silently degrades to append-only, so
   this cannot be skipped.
3. **Every read** — `client.execute_sql(sql, database=self._db)`, passing the resolved record.
   Never a string: `execute_sql(database="<id>")` re-resolves per call, and a name would reach
   the framework's by-name fallback (see `docs/engine-contract.md`).
4. **Table reference in SQL** — `"default"."<schema>"."<table>"`; inside a managed database the
   built-in catalog is always `default`.

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

Ids and filter literals are charset-validated before SQL interpolation, reusing
`hotdata_langchain/_sql.py`'s `validate_identifier`/`quote_literal` — reject anything outside a
conservative charset rather than attempt general SQL escaping. The query vector itself is never user-controlled text; it's a
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

**Step 0 — confirm the embedding key's scope before writing any code.** `.env` carries
`OPENAI_EMBEDDING_KEY` (a team key) alongside `OPENAI_API_KEY`. Its scope is unconfirmed: a
key scoped to embeddings 403s on chat completions and a chat-scoped key 403s on embeddings.
One `embeddings.create` call with a single short input settles it. Do this first — discovering
it mid-implementation is the avoidable version of this problem.

Unit tests need no provider credentials. A fake `HotdataClient` (`MagicMock`) backed by an
in-memory dict, where `load_managed_table` does a *real* `pq.read_table(file).to_pylist()` (so
the `list<float32>` round-trip is genuinely exercised, not mocked away) and `execute_sql` does
real SQL-shape parsing rather than returning a bare canned value. `tests/conftest.py` already
provides `managed_db` and `databases_api` fixtures from the #38 work — reuse them so the
constructor's id resolution is stubbed the same way everywhere. For embeddings, use `langchain_core.embeddings.DeterministicFakeEmbedding` (confirmed
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

### Tracking

Each phase is its own issue and its own PR, under one tracking issue. As of writing **none of
these exist yet** — the only related issues are #39 (the agent-facing tool surface, a different
surface) and #36 item 2 (`create_index`, which gates Phase 3 alone).

| Issue to file | Scope | Blocked by |
|---|---|---|
| Epic: `HotdataVectorStore` | tracking; links this plan and the phases below | — |
| Phase 1 — MVP | `add_texts`, the four `similarity_search*`, `get_by_ids`, `delete`, `from_texts`, promoted-column filtering, unit tests, `examples/langchain_vectorstore.py`, README, CHANGELOG | nothing |
| Phase 2 — MMR | own PR; raw-vector read path | Phase 1 |
| Phase 3 — self-provisioning | `from_texts(..., create_index=True)` | #36 item 2 |

Phase 1 is a complete, mergeable `VectorStore` on its own: `as_retriever()`, chains and evals
all work once it lands, with no index provisioned and no further phases.

## Cross-repo dependency tracking

- **`hotdata_langchain` itself — id-only database addressing, shipped.** Issue #38 landed in
  0.3.0: `resolve_database_by_id` fetches by `GET /databases/{id}` with no by-name fallback, and
  `query_scope` rejects an unresolved string scope. This plan's constructor is built on it; see
  "File and constructor" above. Nothing further needed.
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
