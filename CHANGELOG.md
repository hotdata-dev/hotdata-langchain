# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `HotdataToolCache` and `cached()` in `hotdata_langchain.cache`: cache LangChain tool
  results in a Hotdata managed table, keyed by tool name and arguments. Works on any
  plain function/tool, not just this package's own. `make_hotdata_tools` gains `cache`
  and `cache_ttl` parameters that wire the two read-only tools (`hotdata_execute_sql`,
  `hotdata_list_managed_databases`) through it; the mutating tools are never cached.

### Changed

- Bump `hotdata-framework` to `>=0.8.0` and `hotdata` to `>=0.8.0` — both released 0.8.0
  today, adding native key-based `mode="upsert"`/`"update"`/`"delete"` loads on managed
  tables (used by the new cache backend) and a per-call `key=` override on
  `load_managed_table`.
- Add `pyarrow` as a direct dependency (already a transitive dependency of
  `hotdata-framework`; the new cache module writes parquet directly).

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
