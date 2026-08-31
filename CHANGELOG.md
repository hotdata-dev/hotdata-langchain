# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.14.0] - 2026-08-31

### Added

- **`searchable_columns=` on `make_hotdata_tools`, so the SQL tool can name every indexed
  column rather than only the one the search tool ranks.** Takes `(table, column)` pairs written
  `catalog.schema.table`, confirms each against the control plane when the tools are built, and
  drops with a warning any a ready index does not cover. Every confirmed column gets its own
  worked `bm25_search(...)` (or `vector_search(...)`) call in the description. Order carries:
  the first is the one a model reaches for most, so lead with the table most questions are
  about. `SearchableColumn` and `verify_searchable_columns` are exported for callers assembling
  descriptions themselves.

- **The whole supported Python range is now tested, and lint and types are gated.** CI ran
  `pytest` on 3.12 alone while `requires-python` claimed `>=3.10`, so neither end of the range
  was exercised; it now runs the suite on 3.10 through 3.14 and fails the build on `ruff check`,
  `ruff format --check`, or `mypy` in strict mode. The suite passes on every version. One test is
  skipped on 3.10 only, where `tomllib` is not in the standard library.

### Fixed

- **The SQL tool no longer claims the registered column is the *only* indexed one.** It said
  "the BM25-indexed column is `<column>` on `<table>`" on the strength of what the caller had
  wired to the search tool, which is a statement about one tool's configuration and not about
  the database. On any database indexing more than one column that sentence was false, and it
  was measured being followed in preference to what `hotdata_describe_tables` reports — so the
  model searched a table the answer was not about and returned a confident, wrong number. It now
  says that column *has* an index, which is what the caller actually told us.

  Measured across 84 runs on one model (`gpt-5.1`), one dataset and one question, asking an
  aggregate whose numbers lived on a table other than the registered corpus. With the corpus
  named alone the model searched the right table in 0 runs of 12; declaring the other column
  through `searchable_columns=` took that to 7 of 12 (Fisher exact p = 0.005). Composition rose
  7 of 12 to 10 and the `ILIKE` fallback fell 4 of 12 to 2, but neither is significant at this
  sample size (p = 0.37 and p = 0.64) — read them as direction, not as result.

  Three things this does *not* fix. Naming a column without giving it its own worked call moved
  almost nothing (2 of 12). The gain is much weaker without a prior `hotdata_describe_tables`
  call in the thread, which is the likelier shape of a single-question session (2 of 6 cold
  against 5 of 6 warm). And the cohort size a model asks for is still an arbitrary `k`, so the
  answers themselves remain wrong for a different reason — 0 of 84 runs across every wording
  produced the correct figure ([#62](https://github.com/hotdata-dev/hotdata-langchain/issues/62)).

## [0.13.0] - 2026-08-26

### Added

- **`hotdata_describe_tables` reports which columns can be searched, and by what.** Each column
  of a described table carries `searchable_by` — `text relevance` where it has a keyword index,
  `meaning` where it has a vector one. Indexes are invisible to SQL (there is no `pg_indexes`
  and no `information_schema.indexes`), so this was the one fact about a table an agent could
  not look up and had to be told at construction. The capability is named rather than the
  mechanism: a model can act on "matches the words a value uses", not on "bm25".

  Only indexes the engine reports as ready are named, because a search against one still
  building fails after the model has committed to that route. The cost is one control-plane
  call per described table, and only on the per-table call — the no-argument listing stays
  index-free, so a wide database does not become N calls. `describe_search_capabilities=False`
  on `make_hotdata_tools`, or `search_capabilities=False` on the lower-level helpers, turns it
  off.
- A table's generated vector columns are left out of its description. Where a vector index was
  built by an embedding provider over a text column, the engine materialises a vector column
  beside it — `content` produces `content_embedding` — and `information_schema` reports it as an
  ordinary column with nothing marking it as generated. Describing a 1536-wide float list as
  ordinary data invites queries against it, so the text column carries the `meaning` capability
  and the generated column is not listed. Turning `search_capabilities` off also stops the
  filtering, since nothing then knows the column was generated.
- **Two tool sets over different databases can be told apart.** `make_hotdata_tools` takes
  `tool_name_suffix=`, appended to every tool name in the set, so registering a second set no
  longer puts two tools called `hotdata_execute_sql` in one prompt — a name the model can
  neither address nor choose between. Cross-references between descriptions follow the suffix,
  so the SQL tool points at the schema tool of its own set. The suffix is validated when the set
  is built (letters, digits, underscores or hyphens; the whole name within 64 characters, the
  shortest tool-name limit among the providers this package is used with) rather than when a
  provider first rejects a call. An explicit
  `search_tool_name` is used exactly as given, since naming that tool is already the caller's
  decision. `suffixed_tool_name` is exported for callers registering tools of their own
  alongside these.
- **Descriptions name the database they work on.** The SQL, schema and search tools now open
  with `Works on the 'sales' database.`, taken from the database record and overridable with
  `label=`. The instant-database tools deliberately do not: they act on the workspace, so
  naming one database in them would be false. A database with no name gets no such sentence
  rather than one naming its id, which would present an id where a name is expected.
- `SEARCH_NOUNS` and `search_nouns_by_column` name a search capability on its own, where
  `CAPABILITY_PHRASES` and `capabilities_by_column` give it as a sentence fragment. The phrases
  are now derived from the nouns, so a payload field and a description sentence cannot drift
  into naming the same capability two ways.

### Fixed

- `pip install hotdata-langchain` now has a form that provides everything the documented
  quickstart imports: `pip install "hotdata-langchain[agents]"`. The runtime needs only
  `langchain-core`, which is what a LangChain integration should depend on, but `create_agent`
  lives in `langchain` — so a clean install followed by the quickstart's first import raised
  `ModuleNotFoundError: No module named 'langchain'`. Found by an audit that ran every code
  block on the published docs page from a clean environment against production.
- **Every public JSON emitter takes `database_id=`**, accepting a database id or an
  already-resolved `ManagedDatabase`, the same as `make_hotdata_tools`. `execute_sql_json`,
  `bm25_search_json`, `semantic_search_json`, `hybrid_search_json` and `describe_tables_json`
  all took `database=` and accepted only a resolved record, so the id used everywhere else
  raised `TypeError: database must be a resolved ManagedDatabase, got str`, pointing at a
  resolution step no example named. The guard was stricter than its own reason: it existed to
  keep a name away from the framework's by-name resolver, and this package's own
  `resolve_database_by_id` is by id only. Resolution is still by id — a name raises `KeyError`
  rather than matching a non-unique display label. An id costs a lookup per call; passing the
  resolved record skips it, which is what `make_hotdata_tools` does once at construction.
- `hotdata_list_managed_databases` and `hotdata_create_managed_database` report a database's
  `name`. They reported `description`, which the v1 API's database responses do not carry at
  all: `DatabaseSummary` and `DatabaseDetailResponse` have only `name`, and `description`
  survives just as an accepted input alias on `CreateDatabaseRequest`. The framework maps
  `description=detail.name`, so this relabels the key without changing the value.
  `create_managed_database` already took `name=`, so the package named one field two ways
  depending on the direction it was travelling.

### Changed

- **Breaking, in two places.** The `database=` keyword is now `database_id=` on
  `execute_sql_json`, `bm25_search_json`, `semantic_search_json`, `hybrid_search_json` and
  `describe_tables_json`; and `hotdata_list_managed_databases`,
  `hotdata_create_managed_database` and the exported `managed_database_summary` emit `name`
  where they emitted `description`. Both raise rather than degrading quietly — an unexpected
  keyword, and a `KeyError` on the old key.

  For an agent, the payload key change is the one that shows: two tools now return `name`, and
  the list tool's description quotes `name` to match. Every other tool is unchanged.
  `ManagedDatabase.description` is a `hotdata-framework` field and is untouched here.

## [0.12.0] - 2026-08-25

### Fixed

- `hotdata_describe_tables` accepts a `catalog.schema.table` reference, which it used to
  reject. The SQL tool's description tells the model to address tables with all three parts,
  and both descriptions reach it in one prompt, so following one turned the other into an
  error the model had to recover from — measured twice in one agent run. Bare and
  `schema.table` references are unchanged, and a catalog, when given, now narrows the lookup,
  which matters on a database exposing more than one. A reference naming a catalog no managed
  table answers to reports the table as missing, rather than as declared and awaiting a load.
- `hotdata_describe_tables` names a catalog in its worked example only where the database
  exposes exactly one, resolved from `information_schema` rather than assumed, and
  `make_hotdata_describe_tables_tool` takes `catalogs=` to say which. A hardcoded `default` is
  right for an instant database and wrong for an attached source, whose tables answer to the
  attachment alias — so on one of those it disagreed with the SQL tool's description, which
  resolves the catalog per database, in the one prompt both reach.
- `hotdata_describe_tables` matches a table however the reference is cased. The engine
  lowercases identifiers when it stores them, so `information_schema` holds the lower form and
  the exact filter this built found nothing for `PUBLIC.listings` — while the same reference in
  a `FROM` clause resolves, because a bare reference in SQL is case-insensitive. Describing a
  table was the one place where the case a model happened to type decided whether it got an
  answer.

### Changed

- **The tools call a database an "instant database", not a "managed database"**, matching
  what the product now calls it. This is the text a model plans against, so anyone holding a
  snapshot of a tool description — a prompt fixture, an eval keyed on wording — will see it
  change. Tool names and the Python API are deliberately unchanged: `hotdata_execute_sql` and
  the rest keep their names, as do `ManagedDatabase`, `list_managed_databases()` and
  `create_managed_database()`. Renaming those is breaking and is tracked separately.

  The rename missed one string, because the phrase spanned two source lines and so existed on
  neither: `hotdata_load_managed_table` went on describing a table as declared on a "managed
  database" while the listing and creation tools called the same thing an "instant database".
  All three reach a model in one prompt, and two names for one thing is a contradiction it has
  to resolve. Three docstrings, which reach a maintainer rather than a model, were left behind
  the same way. Tests now fail if any description calls a database "managed" while another
  calls it "instant", if a description names a tool that is not registered, or if "managed
  table" reaches a model.
- On a fused search route, the closing advice about aggregating in SQL states the composed
  form's cohort as advice rather than as a comparison against the tool. It read "Ranking there
  goes by the words alone, so it is narrower than this tool", which put a reason to decline
  immediately after the instruction to compose. The qualifier remains, since only the text half
  of a fusion is expressible in SQL, but it now says how to use the composed form.

  Whether this changes what a model does is **not established**. It was prompted by a fused
  route falling back to `ILIKE` and to pasted id literals, but on resampling the unfused route —
  which never carried the sentence — falls back at least as often, so the earlier reading was
  drawn from noise. The change stands on the copy being wrong either way.

## [0.11.0] - 2026-08-21

### Added

- Hybrid search. Where a table carries a BM25 index on the searched column *and* a plain
  vector index beside it, `hotdata_search_text` now ranks by wording and by meaning at once,
  merging the two with reciprocal rank fusion
  ([#39](https://github.com/hotdata-dev/hotdata-langchain/issues/39)). It stays one tool with
  one name and one `score` column: the two searches fail differently — BM25 misses a
  paraphrase sharing no words with the query, vector search misses rare exact tokens like ids
  and model numbers — so doing both beats making the model choose between them.
- Fusion is one SQL query rather than two searches merged on this side, so it costs one round
  trip. `hybrid_search_sql` and `hybrid_search_json` build and run it, and `RRF_K` (60) and
  the per-pathway candidate depth (`max(4k, 20)`) are both overridable there. The vector half
  is shaped so the query still resolves through the vector index rather than scanning the
  table, which `EXPLAIN` confirms and nothing in the result would reveal.
- `search_semantic_column=` on `make_hotdata_tools`, and `semantic_column=` on
  `make_hotdata_search_tool`, name the vector column to pair with the text one. Needed only
  when a table carries more than one plain vector index; with exactly one it is inferred. The
  engine records no link between a vector column and the text it was derived from, so the
  pairing is a statement the caller makes rather than something that can be read back.
- `search_strategy="hybrid"` raises when a fusion cannot be built, rather than falling back
  the way `"auto"` does. `search_strategy="text"` opts back out to plain BM25.
- `Fusion`, `RRF_K`, `hybrid_search_sql`, `hybrid_search_json` and `fusable_vector_indexes`
  are exported.

### Changed

- Passing `search_embedding=` alongside a BM25-indexed column now fuses the two searches
  where the table supports it. Previously it was accepted and had no effect on a text route.
  Callers wanting the old behaviour can pass `search_strategy="text"`. Fusion needs both
  halves and a key column to join the two rankings on, so a table missing a ready BM25 index
  on the searched column, or missing the key, keeps the search it had.
- The text tool's description says it also matches meaning when the route is fused, and
  qualifies its advice about aggregating in SQL: only the text half of a fusion is
  expressible there, so the composed form is narrower than the tool. The SQL tool's
  description no longer says the search tool "does the same ranking" on such a route. Both
  descriptions reach the model in one prompt, and understating the tool costs exactly the
  recall fusion was added to win back.

## [0.10.0] - 2026-08-19

### Added

- Search by meaning. A column carrying a vector index gets a `hotdata_search_semantic` tool
  that ranks rows by how close they are in meaning to a query, alongside the existing
  text-relevance search ([#39](https://github.com/hotdata-dev/hotdata-langchain/issues/39)).
  Hits carry `_distance`, the engine's own column name, where **lower is nearer** — the
  reverse of what `score` means on the text route, which both tool descriptions state.
- Which of the two a column gets is read from its indexes when the tools are built, not
  chosen by the caller. Indexes are invisible to SQL, so the new `hotdata_langchain.indexes`
  asks the control plane once and reports each column's capability. Callers configure a
  corpus and get whichever search the data supports, with one agent-facing contract either
  way. Introspection fails open: a control plane that cannot be reached leaves the text
  route, which is what this did before it could ask.
- `search_embedding=` on `make_hotdata_tools`, required only for a *plain* vector index —
  one built over a column that already holds vectors, where the engine has no record of how
  they were produced and cannot embed a query to match them. An index built with an embedding
  provider needs nothing on this side: the engine embeds both the column and the query.
  Omitting it where it is required fails when the tools are built, not on the agent's first
  query.
- A vector index whose reported metric is absent, or is one this package has no distance
  function for, is refused when the tools are built rather than assumed to be `cosine`.
  Emitting `cosine_distance` against an `l2` index is not an error the engine reports: the
  query returns rows, by full scan, ranked by a function the vectors were never indexed for.
  A provider-backed index is exempt, since the engine resolves the function from the index.
- `search_strategy=` to force a route. `"semantic"` raises if no vector index covers the
  column; `"text"` does not, because index introspection fails open and a failed listing
  should not stop a tool being built over a column that really is BM25-indexed.
- `DEFAULT_SEMANTIC_TOOL_NAME`, `SearchRoute`, `SearchStrategy`, `resolve_search_route`,
  `generated_vector_columns`, `indexes_for_column` and `DistanceMetric` are exported. The
  tool name matters most: it now varies with the data, so a consumer filtering tools by name
  has to be able to ask for it rather than hardcode a string that changes when a corpus gains
  an index.

### Changed

- A pinned corpus whose column carries a vector index now gets the semantic tool rather than
  the text one, and the tool is named `hotdata_search_semantic` rather than
  `hotdata_search_text`. The name reaches the model, and calling a search that ranks by
  meaning "search_text" states the one thing it does not do. Pass `search_tool_name=` to pin
  the name. There is no route to keep: a column that searches by meaning cannot also be
  BM25-indexed, since a provider-backed index excludes every other index on its table and a
  plain one sits on a vector column.
- The SQL tool's description follows the retrieval route, naming `vector_search` where the
  pinned column is searchable by meaning and `bm25_search` where it is searchable by text,
  and carrying the `ORDER BY _distance ASC` that `vector_search` needs — its rows come back
  unsorted, so a trailing `LIMIT` without that sort returns arbitrary rows rather than the
  nearest ones. Where the route has no composed form at all — a plain vector index, which
  would need a query vector SQL cannot express — the description says so rather than
  advertising a route the agent cannot take. The two descriptions reach the model in one
  prompt, so both are resolved from a single route rather than written independently.
- `DISTANCE_FUNCTIONS` and `DistanceMetric` are defined in `hotdata_langchain._sql` rather
  than in `hotdata_langchain.vectorstore`, since the search tools emit the same engine
  functions and two copies of that mapping could drift apart without anything failing. No
  import breaks: `DISTANCE_FUNCTIONS` is exported from `hotdata_langchain` as before,
  `DistanceMetric` now is too, and both remain bound in `hotdata_langchain.vectorstore`.

### Fixed

- `docs/engine-contract.md` claimed `vector_search` takes only a vector, so a meaning-defined
  cohort could not be expressed in SQL by an agent unaided. That holds for a plain vector index
  and not for a provider-backed one, where `vector_search(table, column, 'query text', k)` takes
  text and composes exactly like `bm25_search`. Verified against the live engine, along with the
  restriction that makes this awkward: a provider-backed index cannot coexist with any other
  index on its table, so the arrangement that makes semantic search free is the one that forbids
  BM25 beside it.
- `docs/ai-native-layer-roadmap.md` argued rank fusion should go client-side to avoid a second
  round trip. The latency measurement it cites stands, but the conclusion does not: reciprocal
  rank fusion is expressible as a single SQL query, since CTEs, `ROW_NUMBER() OVER` and
  `FULL OUTER JOIN` all work, so no engine-side fusion primitive is a prerequisite.

## [0.9.0] - 2026-08-18

### Added

- `metadata.client_warning`, a warning channel of this package's own, on the SQL and search
  envelopes. `metadata.warning` belongs to the engine — the SDK populates it from the query
  response and this package only passes it through — so writing our own text there could
  overwrite an engine-supplied warning. The two now stay separable, and the key is absent when
  there is nothing to say ([#60](https://github.com/hotdata-dev/hotdata-langchain/issues/60)).
- A result cut at `max_rows` says so, naming how many rows the query matched and where the cap
  fell. `row_count` was already the pre-cap total on the SQL path, and a deployed agent did
  spot the gap and paginate unprompted — but nothing stated the boundary, so it guessed
  `OFFSET 96` and re-read four rows it already had. The SQL tool's description now states the
  cap as well.
- A `k` above the search tool's row limit is reported in `client_warning`. The clamp runs
  *before* the query, so the engine only ever ranks `max_rows` rows and `row_count` honestly
  reports them: "asked for 200, got 100" was indistinguishable from "only 100 matched", and an
  agent was measured reporting a cohort it believed was 200 listings. The ceiling is now stated
  in the tool description too.
- SQL passing a date/time format pattern with no `%` to `to_char`, `to_date`, `to_timestamp`
  or `date_format` is flagged in `client_warning`, with the strftime equivalent when it can be
  worked out. The engine's patterns are strftime, so `to_char(d, 'YYYY-MM-DD')` returns the
  literal text `YYYY-MM-DD` on every row with no error — a deployed agent over OpenTelemetry
  spans answered with `Day 1, Day 2, Day 3, Day 4` from correct numbers whose labels were all
  that string. The check is engine-independent and catches a hand-written query too. The same
  hint is raised when the query *fails*, ahead of the engine's own message and as a
  `HotdataToolError` — applying a template to a column rather than a literal was measured
  returning nothing more specific than "An internal server error occurred", so on that path
  the hint is all the model has.
- `hotdata_describe_tables` reports each column's `non_null` count and the table's `row_count`.
  Types alone say a column exists, not that anything is in it: asked what was worth analysing,
  an agent recommended a column that is NULL on all 7,535 rows, and a 63-column spans table
  presents 46 sparse `attr_*` columns as equally available. One aggregate per table described;
  turn it off with `describe_column_stats=False` on `make_hotdata_tools`, or `column_stats=False`
  on the tool factory and `describe_tables_json`.
- Every tool's arguments now carry descriptions in the JSON schema the model sees. Tools reach
  a model through two channels — the description and the argument schema — and only the first
  was used; `k` in particular arrived as `{"title": "K", "type": "integer"}` and nothing else.
- `search_key_column` (default `"id"`) on `make_hotdata_tools` and `make_hotdata_search_tool`,
  and `HotdataToolError`, `result_payload`, `result_json` and `CLIENT_WARNING_KEY` as public
  exports. `engine_error_message` returns a `HotdataToolError`'s message as it stands rather
  than walking past it to the response body it was built from.

### Changed

- A search hit carries the table's `id` alongside the searched column by default, where before
  it carried the searched column alone. The id is what joins a hit back to the fact table, so
  the old default quietly disabled this integration's central claim, that a retrieved row is an
  ordinary SQL value; the downstream application had to discover this and pass `search_columns`
  by hand. The column is looked up once when the tool is built and dropped when the table has
  none. Pass `search_key_column=None` for the previous behaviour, or `search_columns` to name
  the projection outright.
- A table declared on a managed database but never loaded is reported as declared and empty
  rather than as a missing table. It has no rows in `information_schema.columns`, so its schema
  lookup was indistinguishable from a table that does not exist.

## [0.8.0] - 2026-08-14

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
  A successful result is passed through untouched, so a tool declaring
  `response_format="content_and_artifact"` keeps its `(content, artifact)` pair, and LangGraph's
  control-flow exceptions are re-raised rather than reported, so a tool calling `interrupt()`
  for human approval still pauses the graph instead of returning its pause as an error.
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
