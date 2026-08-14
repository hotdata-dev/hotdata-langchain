# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `make_hotdata_tools(..., handle_errors=True)` returns each tool's failures as
  `{"error": "<engine message>"}` instead of raising. An exception out of a tool aborts the
  whole LangGraph run, so one invalid query ended the conversation rather than costing a turn,
  and neither obvious escape hatch applies: `create_agent` does not accept a `ToolNode`, and
  `BaseTool.handle_tool_error` only catches `ToolException` while these raise `RuntimeError`.
  Off by default — outside an agent loop, raising is still right ([#41](https://github.com/hotdata-dev/hotdata-langchain/issues/41)).
- `engine_error_message(exc)` and `with_error_feedback(tools)` are public. The framework raises
  `RuntimeError("Bad Request")` while the message the model can act on
  (`Invalid function 'date_sub'. Did you mean 'date_bin'?`) sits in the API response body
  further down the exception chain; a deployed agent recovered from two invalid queries in one
  turn each purely because it could read those. `with_error_feedback` applies the wrapping to
  tools built elsewhere, such as a retriever tool registered alongside these. Both the sync and
  async callables are wrapped: LangChain prefers `coroutine` under async, which is how
  `langgraph dev` and a deployed Agent Server run, so wrapping only `func` — as the demo's
  version did — leaves the error handling unused in exactly the environment that needs it.
- `hotdata_load_managed_table` accepts an `http(s)` URL as well as a local path, downloading it
  and removing the temporary copy afterwards whether or not the load succeeds. A deployed Agent
  Server has no filesystem the requesting user can write to, so a path-only load could ingest
  nothing the process did not already hold. A URL that answers 200 with an HTML login or error
  page is rejected on parquet's magic bytes before anything is uploaded, and a missing local
  path now says what forms are accepted instead of raising a bare `FileNotFoundError`.
  `hotdata_langchain.databases.fetch_parquet` exposes the download on its own.
- The URL fetch refuses an address that is not publicly routable, and caps the download at
  1 GiB. The URL is chosen by the model, and a model's inputs include whatever text it
  retrieved, so an instruction planted in a document is enough to pick one — without the check
  the agent process is a fetcher for whatever its own network can see, including a cloud
  metadata endpoint, and a load completes the loop by landing the response in a table the agent
  can then read. Every resolved address is checked, and again on each redirect, since a public
  URL that 302s to a private one is the standard bypass. `allow_private_hosts=True` on
  `make_hotdata_tools`, `load_managed_table` and `fetch_parquet` lifts it for a deployment whose
  data really is on an internal host; `max_bytes` on `fetch_parquet` raises the size cap. This
  narrows the reachable surface rather than sealing it — the address is resolved twice, so a DNS
  server that answers differently each time can still get through.
- `management_tools=False` on `make_hotdata_tools` leaves out the three managed-database tools,
  for an agent scoped to one fixed database that cannot use them. Not called `read_only`:
  listing databases is itself a read, so what it removes is the managed-database workflow
  rather than everything that writes.
- A name constant per tool — `DEFAULT_SQL_TOOL_NAME`, `DEFAULT_LIST_DATABASES_TOOL_NAME`,
  `DEFAULT_CREATE_DATABASE_TOOL_NAME`, `DEFAULT_LOAD_TABLE_TOOL_NAME` — joining the two that
  were already exported. Selecting a subset of the tools meant hardcoding the strings.


## [0.7.0] - 2026-08-13

### Fixed

- The SQL tool no longer tells the model that the catalog is always `default`. That held for
  managed databases only: an **attached** source's tables answer to the attachment's alias, so
  an agent scoped to one wrote `default.public.results`, got "table not found", and had no way
  to recover — every query against an attached database failed. `make_hotdata_tools` now reads
  the catalogs from `information_schema` once at build time and states the real one in the
  description; pass `catalog="…"` to skip the lookup. The database record cannot be used for
  this: `GET /databases/{id}` reports `default_catalog='default'` for both kinds.
- Corrected the `COUNT(*)` claim, which had been stated as an engine-wide rule since 0.3.0.
  Re-verified across every table in a live workspace: `COUNT(*)` and `COUNT(1)` **succeed** on
  most tables, including one of 6,001,215 rows. On the tables that do reject them, plain
  `SELECT 1 AS k FROM t LIMIT 1` is rejected too — so it is not an aggregate rule, and what
  distinguishes an affected table is unidentified ([#37](https://github.com/hotdata-dev/hotdata-langchain/issues/37)).
  The description now says some tables reject a projection naming none of their own columns,
  and that naming a column always works.
- `hotdata_list_managed_databases` no longer returns an `sql_prefix` of
  `<database_id>.{schema}.{table}`. Verified: that reference is rejected. The catalog is never
  the database id.

### Added

- The SQL tool description states the date/time dialect. Functions are DataFusion's, so format
  patterns are strftime: `to_char(<date>, 'YYYY-MM-DD')` returns the **literal pattern** on
  every row rather than raising, while `to_date` rejects the same pattern. A deployed agent hit
  this and answered with days labelled `Day 1, Day 2, Day 3` over correct numbers, with nothing
  signalling a problem. The description now gives `'%Y-%m-%d'`, notes there is no
  `date_sub`/`date_add`, and says the bad pattern fails silently.
- The SQL tool description warns that identifiers are lowercased when stored, so quoting one to
  preserve case (`r."driverId"`) fails while `r.driverId` resolves.
- `hotdata_langchain.databases.query_catalogs`, which reads the catalogs holding tables in a
  database's query scope.

### Changed

- The SQL tool description names the dialect as **Apache DataFusion, which follows
  PostgreSQL closely**, rather than as "PostgreSQL dialect". Calling it PostgreSQL
  reinforced the prior behind the one measured silent-wrong-value failure: the model wrote
  valid PostgreSQL date formatting and got a column of literal format strings back. Naming
  the engine gives a prior that holds for divergences not yet found — only date/time
  functions have been probed, so string and numeric formatting remain unverified — where
  "PostgreSQL plus a list of exceptions" only covers the ones already measured.
- The search tool's description no longer claims that **SQL cannot rank rows by textual
  relevance**, and no longer tells the model to carry the returned values into SQL as
  literals. Both tools are registered together, so those two sentences contradicted the SQL
  description in the same prompt — and the second is the measured failure itself: an agent
  pasted 100 literal ids into `WHERE id IN (...)`, capping the cohort at the tool's row
  limit rather than at intent. It now describes itself as the route for listing and
  inspecting matches, and points at ranking inside SQL when the answer aggregates over
  them. The `LIKE`/`ILIKE` guard the removed sentence carried is kept.
- `sql_tool_description` leads with `bm25_search` as a table-valued function that joins, groups
  and nests, and prefers it whenever the answer aggregates over the matches. It previously told
  the model to call the search tool and "pass the values it returns into SQL as literals",
  asserting that **SQL cannot rank text** — which is false. Measured: an agent asked to compare
  a relevance-defined cohort against the population pasted 100 literal ids into `WHERE id IN
  (...)`, capping the cohort at the tool's row limit rather than at intent. The `LIKE`/`ILIKE`
  framing is kept: saying `LIKE` merely "works" was previously observed to pull models into
  `ILIKE '%word%'` instead of searching.
- `sql_tool_description` takes `search_table`/`search_column`, so when the caller knows the
  indexed corpus the description names it concretely. BM25 has no brute-force fallback, so a
  guessed column is a hard error rather than a slow scan.

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
