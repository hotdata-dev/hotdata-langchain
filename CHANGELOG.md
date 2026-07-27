# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
