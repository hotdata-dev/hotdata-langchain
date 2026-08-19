# AI-native query layer — near-term roadmap

## Context

The team's shared vision for `hotdata-langchain` is a single `query_hotdata(...)` tool where
"the agent decides what to ask, not how to fetch it" — routing across four pathways (SQL,
full-text/BM25, vector/semantic, point lookups), merging/ranking results, with caching and
permissions underneath. Caching (`HotdataToolCache`) is built but still an unmerged draft
(PR #33, currently parked); `HotdataVectorStore` shipped in 0.4.0 and self-provisions its
index as of the release after it (see [`vectorstore-plan.md`](./vectorstore-plan.md)).

A verification pass (2026-07-22) checked three previously-unverified assumptions directly
against the codebase, across `monopoly`, `runtimedb`, `datafusion-vector-search-ext`,
`hotdata-ibis`, and this repo:

- **BM25/full-text is not a gap — it's already shipped and mature.** `runtimedb` has a real
  Tantivy-backed full-text index (`src/bm25/`), a `bm25_search(...)` DataFusion table
  function, its own `IndexType::Bm25` catalog entry, and a passing e2e test. `monopoly`'s
  control plane is a thin proxy — RuntimeDB owns validation/dispatch. This repo just doesn't
  expose it as a tool yet.
- **Point lookups are a scoped refactor, not a redesign.** The abstraction
  (`PointLookupProvider::fetch_by_keys` in `datafusion-vector-search-ext`) is already
  general-purpose; the coupling is in `runtimedb`'s vector-index build pipeline. Tracked as
  [hotdata-langchain#34](https://github.com/hotdata-dev/hotdata-langchain/issues/34)
  (implementation lands in `runtimedb`).
- **Routing/planning across pathways is genuinely greenfield** — nothing in the stack does
  intent-based pathway selection today. `vector_search()`/`bm25_search()` are both explicit
  SQL functions the caller must name; this repo's tools are fully independent with zero
  shared dispatch. The one reusable precedent is `runtimedb`'s
  `LazyTableProvider::select_best_index` (a narrow, single-heuristic strategy picker) and the
  catalog's `IndexType` enum, which already models `Sorted`/`Bm25`/`Vector` side by side.

## Routing, resolved into three separate problems (2026-07-27)

The earlier "is routing one problem?" framing dissolved once the engine was read directly.
See [`engine-contract.md`](./engine-contract.md) for the verified specifics.

- **Sorted index vs. table scan — already handled, by the planner.** Nobody decides. The
  sorted index has no callable function; it is a transparent substitution inside
  `hotdata_execute_sql`. A "search by number" tool would expose a choice that does not exist.
- **SQL vs. search — the model decides, and it needs telling.** This is an intent
  difference no engine can recover from SQL. It is also *not* self-evident to the model:
  text matching is expressible as computation (`LIKE`, `tsvector`), so an unguided agent
  matches text in SQL and fails. Fixed by stating the constraint in the tool description,
  which measurably works (see engine-contract.md's last section).
- **BM25 vs. vector — nobody should decide; fuse them.** Both take free text and return
  ranked rows, and they fail differently: BM25 misses paraphrase, vector misses rare exact
  tokens (ids, proper nouns). The established answer is `EnsembleRetriever`-style reciprocal
  rank fusion. Fusion must operate on **ranks, not scores** — BM25 is unbounded (~8–11
  observed) and cosine is 0–2, so the scales are not comparable.

The consequence for the tool surface: **one text-search capability that fuses underneath**,
not two tools the model picks between. `hotdata_search_text`'s description deliberately
names no mechanism (a test enforces that it never says "bm25"/"vector"/"hnsw"), so the
retrieval strategy can change without changing the contract the model was given.

**Corrected 2026-08-18: fusion does not need to go client-side, and cannot always happen at
all.** This paragraph previously argued for a client-side hybrid on round-trip grounds —
measured on a real 3-call agent run at 7,057 ms total, ~1,200 ms wall per tool call against
49–79 ms of engine execution, so ~1,100 ms of it round trip. That measurement stands; the
conclusion drawn from it does not. RRF is expressible as **one SQL query** using CTEs,
`ROW_NUMBER() OVER` and `FULL OUTER JOIN`, all of which the engine supports, so the fused
search costs one round trip rather than two and needs no engine primitive — the
`hybrid_search()` in [#37](https://github.com/hotdata-dev/hotdata-langchain/issues/37) is an
optimisation, not a prerequisite. The RRF parameters (constant, dedup key, tie-breaking) still
have to be settled empirically.

The harder constraint found at the same time: **a provider-backed vector index cannot coexist
with any other index on its table**. Semantic search is free of client-side embedding only on
such an index, so the configuration that makes semantic search cheap is exactly the one that
forbids BM25 beside it. Hybrid is therefore available only over a *plain* vector index plus
BM25, where the query vector has to be produced on this side. Both findings are recorded with
their verification in [engine-contract.md](./engine-contract.md).

## Checklist

> **Superseded in part, 2026-08-13.** This checklist predates the downstream LangGraph POC
> ([`hotdata-agents`](https://github.com/hotdata-dev/hotdata-agents)), which consumed published
> 0.6.0 with two agents and produced a 15-finding register
> ([`notes/hdlc-feedback.md`](https://github.com/hotdata-dev/hotdata-agents/blob/main/notes/hdlc-feedback.md)).
> Items below marked shipped are still shipped, but several shipped *with defects the POC
> found*. The current order of work is carried by the issues, not by this file:
> [#59](https://github.com/hotdata-dev/hotdata-langchain/issues/59) model-facing contract →
> [#41](https://github.com/hotdata-dev/hotdata-langchain/issues/41) tool layer →
> [#60](https://github.com/hotdata-dev/hotdata-langchain/issues/60) tool results →
> [#39](https://github.com/hotdata-dev/hotdata-langchain/issues/39) semantic search →
> [#40](https://github.com/hotdata-dev/hotdata-langchain/issues/40) discovery →
> [#62](https://github.com/hotdata-dev/hotdata-langchain/issues/62) cohort discipline →
> [#61](https://github.com/hotdata-dev/hotdata-langchain/issues/61) provisioning boundary.
>
> The POC's headline result: with a **one-sentence** system prompt, tool routing was correct on
> every graded question. That is the design intent working — and it is also why a false sentence
> in a tool description is a behavioural bug rather than a docs nit, since the descriptions are
> the entire briefing the model gets.

### Tier 1 — buildable now, zero blockers (this repo + `sdk-python-framework`)

- [x] **BM25 tool.** Shipped as `hotdata_langchain/search.py` — `hotdata_search_text`, with the
      corpus pinned at construction (nothing lets an agent discover which columns are indexed,
      and the engine errors outright when one is missing).
- [x] **Tool descriptions that carry the engine's contract.** The highest-leverage item found
      so far, and not originally on this list. A 12-token SQL description produced a failed
      run; the contract version produces the correct search-then-SQL path with no system
      prompt at all. Pinned by `tests/test_descriptions.py`.
- [x] **Schema discovery.** `hotdata_describe_tables` over `information_schema`, registered by
      default. Without it an agent guesses column names — and got away with it only because
      the demo fixture is a famous public dataset.
- [x] **`HotdataVectorStore` MVP.** Shipped in 0.4.0 — `add_texts`/`add_documents`, the four
      `similarity_search*` variants, `get_by_ids`, `delete`, `from_texts`, and equality metadata
      filtering over promoted typed columns. Validated against LangChain's published
      conformance suite (`langchain-tests`) as well as this package's own tests, and verified
      end to end against a live workspace. Design: [`vectorstore-plan.md`](./vectorstore-plan.md).
      Self-provisioned indexes and MMR followed (below), closing every phase under
      [#47](https://github.com/hotdata-dev/hotdata-langchain/issues/47).
- [x] **Vector fast path verified (2026-08-06).** The read path's central assumption — that a
      plain `ORDER BY <distance_fn>(...) LIMIT k` is rewritten into an index lookup — is now
      observed rather than inferred, along with the three query shapes that silently forfeit it.
      A `WHERE`-filtered query reaches the fast path too, which the plan had assumed it would
      not. Plans recorded in [`engine-contract.md`](./engine-contract.md).
- [x] **`create_index` on `HotdataClient` (`sdk-python-framework`).** Shipped in
      `hotdata-framework` 0.10.0 as the general `create_index(..., index_type=...)` rather than
      the originally-scoped `create_vector_index`, since `CreateIndexRequest` already accepted
      `"bm25"` and `"sorted"` too. It polls the build job and raises with its `error_message`,
      because the submit call reports success for builds that later fail.
- [x] **Self-provisioned vector index (Phase 3 of [#47](https://github.com/hotdata-dev/hotdata-langchain/issues/47)).**
      `HotdataVectorStore.create_index()` and `from_texts(..., create_index=True)`, always built
      for the store's own `distance` — the server would otherwise default to `l2`, and a metric
      the query's distance function was not built for full-scans silently. This is what makes
      the verified fast path reachable without leaving Python for the CLI.
- [x] **MMR search (Phase 2 of [#47](https://github.com/hotdata-dev/hotdata-langchain/issues/47)).**
      `max_marginal_relevance_search()` and its `_by_vector` variant, so
      `as_retriever(search_type="mmr")` works instead of raising. It is the one read path that
      projects the stored vectors, so its candidate fetch forfeits the index lookup and is
      bounded by `fetch_k` instead.

### Tier 2 — needs scoped backend work, not exploratory

- [x] **Semantic search tool.** Shipped: a column carrying a vector index gets
      `hotdata_search_semantic`, with the route read from the control plane at construction
      rather than chosen by the caller. Both index kinds are supported — provider-backed, where
      the engine embeds the query, and plain, where the caller supplies an `Embeddings`.
- [ ] **Hybrid fusion.** The rest of
      [#39](https://github.com/hotdata-dev/hotdata-langchain/issues/39): reciprocal-rank-fusion
      merge over BM25 and vector, exposed as one tool rather than two the model chooses between.
      Expressible as a single SQL query. Constrained to plain-vector-plus-BM25 corpora by the
      coexistence restriction above, and needs the caller to say which vector column pairs with
      which text column — the engine records no link between them for a plain index.
- [ ] **Point-lookup generalization in `runtimedb`.** Tracked as
      [hotdata-langchain#34](https://github.com/hotdata-dev/hotdata-langchain/issues/34).
      Decouple the lookup-sidecar build from the vector-index pipeline, loosen the registry,
      add a `point_lookup(...)` UDTF.

### Tier 3 — not ready to scope yet, needs a decision first

- [ ] **Structured-intent routing** (deciding *when* to reach into SQL/point-lookup, à la
      `SelfQueryRetriever`). Deferred until the hybrid retriever (Tier 2) shows whether a
      classifier is actually needed.
- [ ] **Permissions.** Entirely untouched by shipped or planned work; no scoping done.
- [ ] **Cross-source joins.** Engine-level question, likely beyond what client-side LangChain
      code alone can provide. `attach_database_catalog` may already be the supported route for
      the cross-*database* case; unverified.

## Cross-repo work this surfaced

Everything below came out of building the BM25 tool and running it against production. The
per-item evidence lives in the linked issues; the verified engine behaviour behind them is in
[`engine-contract.md`](./engine-contract.md).

Grouped by code surface:

- **`hotdata-framework` client gaps** ([#36](https://github.com/hotdata-dev/hotdata-langchain/issues/36)) — errors discard the engine's message (raises
  `e.reason`, "Bad Request", losing the actionable text); `create_index` since fixed in 0.10.0;
  `resolve_managed_database` falls back to matching non-unique display names; `from_env()`
  silently picks a workspace. The first is the one with demonstrated impact on agent behaviour.
- **`runtimedb` engine gaps** ([#37](https://github.com/hotdata-dev/hotdata-langchain/issues/37)) — some tables reject a projection naming none of
  their own columns (`COUNT(*)`, `COUNT(1)`, and even `SELECT 1`), while most tables accept
  all three, so it is neither an aggregate rule nor universal; `to_char` returns an
  unrecognised format pattern verbatim instead of raising, which silently destroys a column of
  values; no hybrid/RRF primitive for the eventual server-side fusion.
- **Vector index dimension detection** ([#52](https://github.com/hotdata-dev/hotdata-langchain/issues/52)) — building an index over an existing
  embedding column fails after a **mixed upsert** (rewritten ids plus new ids in one load), and
  fails *intermittently* on that shape. Earlier "candidate triggers ruled out" conclusions were
  each based on a single passing control and do not hold. The width is read from stored data
  rather than supplied, so no client argument works around it.
- **ANN lookup key is built from the table reference as written** ([datafusion-vector-search-ext#32](https://github.com/hotdata-dev/datafusion-vector-search-ext/issues/32)) —
  a two-part `schema.table` reference resolves correctly but forfeits the vector index, with
  nothing reported. Worked around in the SQL tool's description until the rule resolves the
  reference against session defaults first.
- **id-first addressing in this repo** ([#38](https://github.com/hotdata-dev/hotdata-langchain/issues/38)) — breaking; mirrors the `hotdata-dlt-destination`
  change (its PR #59) that removed by-name resolution entirely. Worth landing before the next
  release rather than shipping two breaking versions.
- **Retrieval surface** ([#39](https://github.com/hotdata-dev/hotdata-langchain/issues/39)) — semantic search tool **shipped**; hybrid rank fusion over it and BM25 still open, as a single
  SQL query rather than the client-side fan-out originally planned.
- **Discovery surface** ([#40](https://github.com/hotdata-dev/hotdata-langchain/issues/40)) — report which columns are searchable in `hotdata_describe_tables`.
  Newly unblocked: indexes are invisible to SQL but `IndexesApi.list_indexes` returns them, so
  this needs no engine change. It is what would let the search corpus stop being pinned.
- ~~**Tool-layer robustness**~~ ([#41](https://github.com/hotdata-dev/hotdata-langchain/issues/41)) — **done.** `with_error_feedback` and `engine_error_message` are
  package API, `make_hotdata_tools(handle_errors=True)` turns the wrapping on, and both the
  sync and async callables are wrapped — the async one is what LangChain actually calls in a
  deployed agent, and the demo's version missed it. `hotdata_load_managed_table` now accepts a
  URL, which is the only ingest route open to an agent with no filesystem of its own — bounded
  by a public-address check and a size cap, since the URL is model-chosen and therefore
  reachable by a planted instruction. Also added a name constant per tool and
  `management_tools=False`, so an application selecting a subset stops hardcoding strings.

Already tracked elsewhere and deliberately not duplicated: point-lookup generalization
(hotdata-langchain#34) and the sorted-index cost model (runtimedb#481).
