# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [0.6.0] - 2026-08-08

### Added

- `HotdataVectorStore.max_marginal_relevance_search()` and
  `max_marginal_relevance_search_by_vector()`, so `as_retriever(search_type="mmr")` works —
  it raised `NotImplementedError` before, as the `VectorStore` base class leaves both
  unimplemented. MMR ranks `fetch_k` candidates by distance, then picks `k` scored against
  both the query and what is already picked, so a top-`k` of near-duplicates becomes a top-`k`
  that covers more ground. Results are in selection order, not distance order, and `filter=`
  applies as it does on `similarity_search`.

  This is the one search that reads the stored vectors, which MMR needs and which forfeits the
  engine's index lookup. Its candidate fetch is a full scan even where an index exists, bounded
  by `fetch_k`; `similarity_search` is unaffected. `lambda_mult` keeps LangChain's `0.5`
  default so ported code behaves identically, but the README explains why you should expect to
  raise it.

- `numpy` is now a declared dependency. It arrived transitively already; the MMR path imports
  it directly, so it is declared directly.

## [0.5.0] - 2026-08-07

### Added

- `HotdataVectorStore.create_index()` builds the vector index that turns the store's searches
  into index lookups, with `from_texts(..., create_index=True)` to do it right after the first
  write. Searches were always correct without an index, just brute-forced; provisioning one
  previously meant leaving Python for the CLI.

  The index is always built for the store's own `distance`, because a query whose distance
  function is not the index's metric silently full-scans instead of erroring, and the server
  would otherwise default to `l2` while this store defaults to `cosine`. An index that already
  exists under a different metric raises, naming both. A matching one is a no-op whether it is
  built or still building, so calling this on every start-up is safe; only after a build of its
  own is rejected does it insist the index report ready, since a half-registered failure would
  otherwise pass for success.

### Changed

- Require `hotdata-framework>=0.10.0` for `create_index`. Additive: nothing this package
  already used changed.
- The SQL tool now asks the model for a full `catalog.schema.table` reference. The two-part
  form resolves and returns correct rows, but the engine's index-lookup rewrite matches on the
  reference as written, so it can forfeit an index with nothing reported
  ([datafusion-vector-search-ext#32](https://github.com/hotdata-dev/datafusion-vector-search-ext/issues/32)).
  `HotdataVectorStore` and the search tool emit the three-part form by construction and were
  never exposed; this closes the one surface where a model writes the reference.
- The vector fast path is documented as verified rather than intended: `EXPLAIN` against a live
  engine shows the index lookup, and a `WHERE`-filtered query reaches it too with the predicate
  pushed in. Observed plans and the shapes that forfeit it are in `docs/engine-contract.md`.
  No code change — 0.4.0 already emitted the correct query shape.

### Known issues

- Building an index over an existing embedding column fails on some tables with `could not
  detect dimension`, reproducibly, while structurally identical tables succeed
  ([#52](https://github.com/hotdata-dev/hotdata-langchain/issues/52)). The width is read from
  stored data rather than supplied, so there is no client-side workaround. Searches remain
  correct on an affected table; they stay full scans.

## [0.4.0] - 2026-08-06

### Added

- `HotdataVectorStore` — an implementation of LangChain's `VectorStore` backed by a managed
  table, so Hotdata works as the retrieval backend for any retriever, chain or eval built on
  that interface. Covers `add_texts`/`add_documents`, the four `similarity_search*` variants,
  `get_by_ids`, `delete` and `from_texts`, plus equality filtering on metadata keys declared
  as `metadata_columns`.

  Searches compile to a single `ORDER BY <distance_fn>(embedding, ARRAY[...]) ASC LIMIT k`
  query using the engine's scalar distance functions. That is correct with no index at all, so a
  new store is usable immediately. Today every search is a full scan. The query is
  also written to match the shape the engine's optimizer rewrites into an HNSW index lookup,
  so that one code path should serve both once an index exists; that rewrite is not yet
  confirmed end to end for these queries, and confirming it needs an index this package cannot
  yet create.

  `database_id` is required and addressed by id; it is resolved once at construction and every
  read and write afterwards addresses the resolved record. `delete` requires ids.

  Validated against LangChain's published conformance suite (`langchain-tests`) in addition to
  the package's own tests.

- `pyarrow` is now a declared dependency. It was already installed as a transitive dependency
  of `hotdata-framework`; the vector store imports it directly, so it is declared directly.

## [0.3.0] - 2026-08-06

### Added

- `resolve_database_by_id` — fetches a managed database record by id (`GET /databases/{id}`)
  with no by-name fallback, and returns an already-resolved `ManagedDatabase` untouched.
  `ManagedDatabase` is re-exported for callers that hold one.

- Full-text search tool backed by the engine's BM25 index. `make_hotdata_tools` grows
  `search_table`/`search_column`/`search_columns`/`search_k`/`search_tool_name` and appends a
  `hotdata_search_text` tool when a table and column are given; `make_hotdata_search_tool`
  builds one directly, so several searchable corpora can be registered side by side. The
  corpus is pinned at construction rather than chosen by the model, because nothing in the
  tool surface lets an agent discover which columns carry a BM25 index.
- `bm25_search_sql` and `bm25_search_json` for building and running a ranked search without
  going through a tool. `bm25_search_json` returns the same `{"metadata", "rows"}` envelope as
  `execute_sql_json`.
- `hotdata_describe_tables` tool (registered by default, `describe_tables=False` to omit) and
  `describe_tables_json`/`make_hotdata_describe_tables_tool`. With no argument it lists the
  scoped database's tables and their column counts; with a table name it returns that table's
  columns and types, capped so one wide table cannot flood the model's context. Reads
  `information_schema`, so it needs no extra permissions. Without it an agent guesses column
  names off the shape of the data it has already seen.
- `demo/` — an end-to-end script that creates a managed database, loads the public SF Airbnb
  fixture, builds a BM25 index, invokes the search tool, and then hands both the search and
  SQL tools to a LangChain agent.

### Changed

- **Breaking: managed databases are addressed by id, never by name.** A Hotdata database name
  is a display label and is not unique, so a by-name lookup can resolve to the wrong database
  — and then every query, load and drop follows it there. The agent-facing
  `hotdata_load_managed_table` made that reachable from an LLM, where a wrong target means a
  replacing load overwrites another database's table.
  - `make_hotdata_tools`, `make_hotdata_search_tool` and `make_hotdata_describe_tables_tool`
    take `database_id=` in place of `database=`. It accepts an id, or a `ManagedDatabase` to
    skip the lookup. The id is resolved once when the tools are built, so a bad id fails
    there rather than on the agent's first query, and queries no longer pay a repeat lookup.
  - The `hotdata_load_managed_table` tool's `database` argument is now `database_id`, and its
    description names the two tools that hand out ids.
  - `load_managed_table` takes `database_id=`; `execute_sql_json`, `bm25_search_json` and
    `describe_tables_json` take a resolved `ManagedDatabase` as `database=` and raise
    `TypeError` on a string, which would otherwise reach the framework's by-name fallback.
  - Passing a name anywhere raises `KeyError`, naming `hotdata_list_managed_databases` as
    where ids come from.

  Mirrors `hotdata-dlt-destination`'s move to id-only addressing. To scope tools to the same
  database as before, pass its id: `client.list_managed_databases()` reports one per database.
- Tool descriptions now state the engine's actual contract instead of a one-line summary.
  `hotdata_execute_sql` names the dialect and the supported constructs, points at the search
  tool for text relevance when one is registered, and warns that an aggregate query must
  reference a column (`COUNT(*)` alone is rejected). The database tools steer callers towards
  ids, since names are non-unique display labels. Every claim was verified against a live
  engine and is pinned by `tests/test_descriptions.py`. Without this an agent reaches for
  `to_tsvector` and the query fails; with it, the correct search-then-SQL path is taken with
  no system-prompt guidance at all.
- Require `hotdata-framework>=0.9.0` / `hotdata>=0.8.0`. The pinned 0.4.1 uploaded through
  `POST /v1/files`, which the API no longer serves, so `load_managed_table` failed with a bare
  `Not Found`; 0.9.0 uses the session/finalize upload flow.

### Fixed

- README quickstart used `create_tool_calling_agent`/`AgentExecutor`, neither of which exists
  in LangChain v1; replaced with `create_agent`.
- README and `examples/langchain_basic.py` ran SQL without a database scope, which the API
  now rejects with `a database is required`. Both now scope their queries.

## [0.2.2] - 2026-06-27

### Changed

- Release 0.2.2

## [0.2.1] - 2026-06-22

### Changed

- Pin `hotdata-framework` to `>=0.3.0` (adds the typed-error API:
  `HotdataError`/`HotdataTransientError`/`HotdataTerminalError`/`classify_sdk_error`).
  No code adoption was required: this package has no SDK error-handling call sites — its
  runtime calls are thin pass-throughs exposed as LangChain `StructuredTool`s, which let
  exceptions propagate to the LangChain runtime by design.

## [0.2.0] - 2026-06-22

### Changed

- Upgrade `hotdata` SDK pin to `>=0.4.1` and `hotdata-framework` to `>=0.2.4`.
- Raise `langchain-core` floor to `>=1.0` (verified against the test suite).

### Added

- Ruff and mypy tooling configuration in `pyproject.toml`, plus `ruff` and `mypy`
  dev dependencies. Applied `ruff check --fix` and `ruff format` cleanup across the
  codebase.

## [0.1.1] - 2026-06-01

### Changed

- Release 0.1.1

## [0.1.0] - 2026-05-19

### Added

- Initial release with LangChain tools for Hotdata managed databases.
