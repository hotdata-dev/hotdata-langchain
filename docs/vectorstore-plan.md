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
- **`runtimedb`** — the deployed query engine; PR #953 bumped its pin to pick up #31, merged
  and confirmed live in production. The rewrite has since been observed firing for the queries
  this package generates (2026-08-06); see `engine-contract.md`.
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
  column" rule the primary read path relies on for the engine's fast-path rewrite (since
  observed directly; see `engine-contract.md`) — so this branch is always brute-force by
  design. Acceptable: `fetch_k` defaults small and is caller-bounded, so a full scan over a
  bounded candidate set is cheap. Own phase, own PR — a distinct correctness surface
  (raw-vector round-trip on read, which the primary path never needs), not bundled with the
  MVP. Two further points settled while building it: `maximal_marginal_relevance` needs the
  query vector as a numpy array, so `numpy` becomes a declared dependency; and it scores
  diversity by cosine similarity whatever the store's `distance` is, which every LangChain
  implementation does and which is documented rather than corrected.
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
preconditions, and upgrades transparently to the HNSW fast path once a matching-metric index
exists on that column — one code path, no index-vs-no-index branching to build or test.
(Verified 2026-08-06; the observed plans are in `engine-contract.md`.) The
`vector_search_vector(...)` table function, by contrast, errors loudly
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
never re-fill it). A `WHERE`-filtered query **does** still reach the HNSW fast path, with the
predicate pushed into the index lookup (`filtered=true` in the plan) — verified 2026-08-06,
see `engine-contract.md`. This plan had assumed the opposite and accepted
brute-force-but-correct as a v1 cost, since attribute-filtered ANN is a harder capability many
engines lack; that caveat no longer applies.

Ids and filter literals are charset-validated before SQL interpolation, reusing
`hotdata_langchain/_sql.py`'s `validate_identifier`/`quote_literal` — reject anything outside a
conservative charset rather than attempt general SQL escaping. The query vector itself is never user-controlled text; it's a
list of floats we format ourselves.

### Dimension binding

This plan assumed `create_index` would need `dimensions` up front, and that the dimension was
therefore something the store had to learn from its first `embed_documents` call and pass on.
**That turned out to be wrong, and the sequencing it implied turned out to be right anyway.**
For a plain vector index — one over a column that already holds vectors — the engine reads the
width off the *stored data*, and `dimensions` applies only to the provider-backed path where
the engine does the embedding itself. A supplied value is ignored here.

So the store passes no `dimensions` at all, and the ordering inside `from_texts` still holds
for a different reason: (1) construct the store, (2) `add_texts`, which embeds and writes,
(3) only then create the index, which now has stored rows to measure. Index creation never
precedes the first write, because before it there is nothing to read a width from.

One consequence: when detection fails, no caller argument can rescue it. That is what makes
[#52](https://github.com/hotdata-dev/hotdata-langchain/issues/52) an engine fix rather than a
client one.

### `l2` relevance-score caveat

The engine's `l2_distance` is **squared** L2 (no `sqrt`), but `VectorStore`'s default
`_euclidean_relevance_score_fn` assumes true (unsquared) Euclidean distance on
unit-normalized embeddings — using `l2` as the configured metric would produce a
wrong-scale relevance score unless corrected, and correcting it properly requires knowing
embedding normalization we don't control. Constructor defaults to `cosine` for this reason
(`1 - distance` is exact, no scale assumption); `l2`/`dot` remain available but flagged in the
docstring rather than silently "fixed."

## Testing strategy

**Step 0 — confirm the embedding key's scope before writing any code. Done, 2026-08-06.**
`.env` carries `OPENAI_EMBEDDING_KEY` (a team key) alongside `OPENAI_API_KEY`. A single
`embeddings.create` call confirmed it is embeddings-scoped and live: `text-embedding-3-small`
returns **1536 dimensions**, no 403. Live verification of the write/read round-trip is
therefore unblocked, and `1536` is the dimension the first real index will be created with.

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

**Live verification.** Done for Phase 1 on 2026-08-06 against the production workspace, via
`demo/vectorstore_demo.py` (database `dbidh4tn5esw2roy7zg4sh1fqv8rov`). Confirmed working:
the `list<float32>` embedding column round-trips through `load_managed_table`; a
1536-dimension `ARRAY[...]` literal in `cosine_distance(embedding, ARRAY[...])` is accepted
and ranks sensibly; the `WHERE` predicate filters inside the ranking query; `mode="upsert"`
with `key=["id"]` leaves 8 rows after two runs of 8 documents; `mode="delete"` accepts a
parquet carrying **only** the key column and removes the row; deleting an absent id is a
no-op; `get_by_ids` skips ids that are not present; and `as_retriever()` composes into an
LCEL retrieval chain that answers from retrieved context.

One constraint surfaced that the design had not anticipated: **an upsert must carry every
column the table has** (`upload is missing column '<name>'`). So `metadata_columns` has to
match the table a store is opened against — pointing a differently-configured store at an
existing table fails on the first write rather than silently writing partial rows. Documented
in the class docstring and README.

**MMR verified live 2026-08-08** against the same workspace, table `public.documents_mmr`,
`text-embedding-3-small` at 1536 dimensions over a 10-document corpus. `lambda_mult=1.0`
reproduced the similarity ranking document-for-document, which is the implementation's own
control and something the unit tests can only assert on synthetic vectors.

One property surfaced that the plan had not anticipated, and it is about the *embedding
model*, not the engine. Every cosine distance in the corpus fell between 0.6055 and 0.6690,
so MMR's relevance term spans ~0.06 while its redundancy term spans several times that —
near-duplicates score ~0.9 against each other. The two terms therefore do not have
comparable scale, and LangChain's `lambda_mult=0.5` default, which weights them equally, let
variety decide nearly every pick: it promoted a document that did not answer the query at
all. At 0.7 and 0.8 the behaviour was correct and identical — the near-duplicate dropped, a
genuine alternative promoted.

The library keeps `0.5` (ecosystem compatibility outweighs one corpus's evidence); the demo
defaults to `0.7` and both READMEs say to sweep it. Single corpus, single query, single
model — an observation to act on, not a law. The NanoBEIR-style eval harness is what would
turn it into a measurement.

**Fast path verified 2026-08-06.** Both outstanding `EXPLAIN` questions are settled, and the
observed plans are recorded in `engine-contract.md`. The primary query plans as a full scan
with no index and as `USearchExec` once a matching-metric index exists. A `WHERE`-filtered
query **also** reaches the fast path, with the predicate pushed into the index lookup
(`filtered=true`) — better than this plan assumed, and it retires the "brute-force-but-correct
is an accepted cost" caveat under Filtering above.

Confirmed as forfeiting the rewrite, silently: projecting the `embedding` column, a distance
function the index was not built for, and omitting `LIMIT`. The `similarity_search*` path
avoids all three by construction, which is worth a regression test now that the boundary is
known rather than assumed. MMR (Phase 2) takes the first one deliberately — it cannot compute
diversity without the stored vectors — and pays for it with a `fetch_k`-bounded full scan.

## Phasing

1. **MVP** — `add_texts`, `similarity_search(_by_vector)`,
   `similarity_search_with_score(_by_vector)`, `get_by_ids`, `delete`, `from_texts` (no
   self-provisioned index yet), promoted-column equality filtering, full unit-test suite,
   `examples/langchain_vectorstore.py`, README section, CHANGELOG entry. A complete, correct,
   mergeable `VectorStore` on its own — `as_retriever()`, chains, and evals all work once this
   lands, independent of anything below.
2. **MMR** — shipped. `max_marginal_relevance_search()` and its `_by_vector` variant, over a
   `fetch_k`-bounded candidate fetch that projects the stored vectors. The async variants came
   free from the base class, which delegates them to the sync ones. Projecting the vector
   column forfeits the index lookup (observed during Phase 3; see `engine-contract.md`), so
   this branch full-scans by design while `similarity_search` keeps the fast path.
3. **Self-provisioning** — shipped. `create_index()` and `from_texts(..., create_index=True)`,
   on `hotdata-framework` 0.10.0's `HotdataClient.create_index`. The store always builds for
   its own `distance`: the server defaults an unspecified metric to `l2` while this store
   defaults to `cosine`, and a metric the query's distance function was not built for silently
   full-scans instead of erroring.
4. Docs/examples are pulled into Phase 1 rather than deferred to the end — this is the piece
   the LangChain conversation will exercise first.

### Tracking

Each phase is its own issue and its own PR, under the epic
[#47](https://github.com/hotdata-dev/hotdata-langchain/issues/47).

| Phase | Scope | State |
|---|---|---|
| Epic: `HotdataVectorStore` | tracking; links this plan and the phases below | [#47](https://github.com/hotdata-dev/hotdata-langchain/issues/47) |
| Phase 1 — MVP | `add_texts`, the four `similarity_search*`, `get_by_ids`, `delete`, `from_texts`, promoted-column filtering, unit tests, `examples/langchain_vectorstore.py`, README, CHANGELOG | shipped in 0.4.0 ([#48](https://github.com/hotdata-dev/hotdata-langchain/issues/48)) |
| Phase 2 — MMR | own PR; raw-vector read path | [#55](https://github.com/hotdata-dev/hotdata-langchain/issues/55) |
| Phase 3 — self-provisioning | `create_index()`, `from_texts(..., create_index=True)` | shipped |

Phase 1 is a complete, mergeable `VectorStore` on its own: `as_retriever()`, chains and evals
all work once it lands, with no index provisioned and no further phases.

## Cross-repo dependency tracking

- **`hotdata_langchain` itself — id-only database addressing, shipped.** Issue #38 landed in
  0.3.0: `resolve_database_by_id` fetches by `GET /databases/{id}` with no by-name fallback, and
  `query_scope` rejects an unresolved string scope. This plan's constructor is built on it; see
  "File and constructor" above. Nothing further needed.
- **`sdk-python-framework` (`hotdata_framework.HotdataClient`) — shipped in 0.10.0.** Landed as
  the general `create_index(..., index_type=...)` rather than the `create_vector_index` this
  plan scoped, since `CreateIndexRequest` already accepted `"bm25"` and `"sorted"` too; one
  method covers self-provisioning for all three instead of building it twice. It polls the
  build job to a terminal state and raises with its `error_message`, because the submit call
  reports success for builds that later fail. It only ever blocked Phase 3 — Phases 1–2 worked
  regardless, against an existing index, a not-yet-existing index, or no index ever, by
  construction of the SQL-path decision above.
- **`runtimedb` PR #953 — merged and live in production.** Pin-bump to pick up
  `datafusion-vector-search-ext` PR #31, confirmed deployed. Our SQL was designed to be correct
  either way (the scalar UDFs work with no index at all), so this was never a blocker. The rule
  has since been observed firing for the queries this package generates; see "Live
  verification" and `engine-contract.md`. No longer a dependency to track.
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
