# Engine contract — what the SQL and search surface actually does

Every claim here was checked against a live workspace (`api.hotdata.dev`, RuntimeDB behind it)
rather than read off a spec, because several of them contradict what the docs or the dialect
imply. They are the facts the tool descriptions in `hotdata_langchain/` encode, so if one of
them changes, a description somewhere is now lying to the model.

Last verified 2026-07-27 against `hotdata-framework` 0.9.0 / `hotdata` 0.8.0.

## SQL

Postgres dialect, and the following are confirmed working: joins, CTEs, subqueries, `GROUP BY`,
window functions, `ORDER BY`/`LIMIT`, ordinary scalar functions, `LIKE` and `ILIKE`, and
schema-qualified as well as bare table names. Those shorter forms *resolve*, but see "Index
types and how each is reached" below — a reference that is not fully qualified can forfeit the
vector index while still returning correct rows.

Constraints that matter enough to state in a tool description:

- **Some tables reject a projection that names none of their own columns** with `must either
  specify a row count or at least one column`. It is *not* an aggregate rule and *not*
  universal (re-verified 2026-08-13 across every table in the test workspace):
  `SELECT COUNT(*) FROM t` succeeds on most tables, including a 6,001,215-row one, and on
  `default.public.listings` it fails — as does `SELECT 1 AS k FROM listings LIMIT 1`, which is
  not an aggregate at all. Naming a column always works, so the description gives
  `COUNT(<column>)` as the safe form. What distinguishes an affected table is unidentified;
  tracked as [#37](https://github.com/hotdata-dev/hotdata-langchain/issues/37).
- **A declared managed table with no data rejects every query**, with `Managed table
  'default.public.customer' is declared but has no data; POST a load before querying`, and
  reports zero rows in `information_schema.columns`. Worth knowing because it is easily
  mistaken for the constraint above — the two produce different messages for what looks like
  the same failing query.
- **Date and time functions are DataFusion's, not PostgreSQL's**, and one wrong guess fails
  silently. Verified 2026-08-13:

  | Expression | Result |
  |---|---|
  | `to_char(cast('2026-08-12' AS DATE), 'YYYY-MM-DD')` | `'YYYY-MM-DD'` — the pattern itself, **no error** |
  | `to_char(cast('2026-08-12' AS DATE), '%Y-%m-%d')` | `'2026-08-12'` |
  | `to_date('2026-08-12', 'YYYY-MM-DD')` | error |
  | `to_date('2026-08-12', '%Y-%m-%d')` | `2026-08-12` |
  | `date_sub(cast('2026-08-12' AS DATE), 6)` | error: `Invalid function 'date_sub'. Did you mean 'date_bin'?` |
  | `cast('2026-08-12' AS DATE) - INTERVAL '6 days'` | `2026-08-06` |
  | `date_trunc('day', …)`, `now()`, `current_date` | work as expected |

  Format patterns are strftime. A PostgreSQL template contains no `%` directives, so `to_char`
  emits it verbatim for every row while `to_date` rejects it — the two are inconsistent with
  each other, which is what makes this easy to walk into. Engine-side, tracked in
  [#37](https://github.com/hotdata-dev/hotdata-langchain/issues/37).
- **Identifiers are lowercased when stored, and quoting to preserve case fails.** The attached
  F1 source declares `driverId`; the engine exposes `driverid`. `r.driverId` and `r."raceid"`
  both resolve, `r."driverId"` fails with a bare `RuntimeError: Bad Request` carrying no body
  at all — so the usual cause-chain recovery finds nothing to report.
- **There is no full-text matching in SQL.** No `to_tsvector`, no `plainto_tsquery`. The engine
  answers with `Invalid function 'to_tsvector'. Did you mean 'to_char'?`. `LIKE`/`ILIKE` work as
  substring tests but cannot rank. Note this is about the *PostgreSQL* full-text functions:
  ranking by relevance inside SQL is available, via `bm25_search` — see below.

**A database scope is required.** An unscoped query fails with `a database is required: set the
X-Database-Id header or the database_id body field`.

**There is no universal catalog name.** A managed database's tables answer to `default`; an
attached source's tables answer to the *attachment alias*, not to `default`. Verified
2026-08-13 against `f1_db`: `default.public.results` fails with "table not found" while
`f1.public.results` returns rows, and `information_schema.tables` lists catalog `f1` across all
seven of its schemas.

The database record does not settle it either — `GET /databases/{id}` reports
`default_catalog='default'` for **both** kinds, and `default_schema='main'` for both when the
actual schema is `public`. `information_schema.tables.table_catalog` is the only authoritative
source, which is what `hotdata_langchain.databases.query_catalogs` reads.

## Full-text search

```sql
bm25_search('catalog.schema.table', 'column', 'query text' [, limit])
```

Returns the table's columns plus a trailing `score` (Float32). Four properties shape
`hotdata_langchain/search.py`:

- **It is a table-valued function, so it composes.** The result is a relation like any other:
  it joins, groups, and nests in subqueries and CTEs. A cohort defined by relevance can
  therefore be aggregated *inside one query*, rather than retrieved and passed back as SQL
  literals. This is the most strategically important property in this file — it is what makes
  "find the rows about X, then aggregate over all of them" a single query, and it is why
  `sql_tool_description` names the function rather than pointing only at the search tool.

  The vector side does **not** compose this way: `vector_search` takes a *vector*, not text, so
  a meaning-defined cohort cannot be expressed in SQL by an agent unaided. Something has to
  embed the concept first. That asymmetry is tracked in
  [#39](https://github.com/hotdata-dev/hotdata-langchain/issues/39).
- **Results are not sorted.** Rows come back in rowid order, like SQLite FTS5. Verified: without
  `ORDER BY` the scores came back `8.788, 8.092, 8.034, 8.254, 8.496`. Ranking must be asked for.
- **The fourth argument is the real bound.** BM25 is top-k, so tantivy needs the bound before
  planning. A bare `LIMIT n` pushes down and drives it, but `ORDER BY score DESC LIMIT n` does
  not — the sort blocks limit pushdown and the scan falls back to the engine's much larger
  default. Correctness is unaffected (explicit-`k` and trailing-`LIMIT` returned identical
  top-3), and at 7.5k rows the cost was not measurable (40 ms vs 38 ms median), so this is a
  scan-bound difference rather than an observed slowdown. Passing `k` explicitly is free, so we do.
- **The index is a hard prerequisite.** No brute-force fallback: a column without a BM25 index
  gives `No BM25 index found on column 'name' for <conn>.public.listings`. This differs from
  vector search, where scalar distance UDFs still work without an index.

Scores are comparable within one result set, not across queries. Observed BM25 range on real
data: roughly 8–11. Cosine distance is 0–2. **Never compare or average across the two** — this is
why fusion must work on ranks (RRF), not scores.

## Index types and how each is reached

RuntimeDB has three (`IndexType` in `src/catalog/manager.rs`: `Sorted`, `Bm25`, `Vector`), but
they are not three of the same kind of thing:

| Index | Reached by | Named by the caller? |
|---|---|---|
| Sorted | the planner substitutes the sorted parquet when a pushed-down filter matches the index's **leading** sort column | no — transparent |
| BM25 | `bm25_search(...)` table function | yes |
| Vector | `vector_search(...)` table function | yes |

So the sorted index needs no tool: it is already served through `hotdata_execute_sql`. There is
no callable function for it.

### The vector index is also reached without naming it (verified 2026-08-06)

A plain `ORDER BY <distance_fn>(col, ARRAY[...]) ASC LIMIT k` is rewritten into an index lookup
when a vector index built on the **same metric** exists on that column. This is what
`HotdataVectorStore` relies on, and it was confirmed by `EXPLAIN` against `api.hotdata.dev`
before and after building a cosine index on a 1536-dimension `List(Float32)` column.

Without an index the physical plan is a full scan:

```
SortExec: TopK(fetch=3), expr=[dist@3 ASC NULLS LAST]
  ... DataSourceExec: file_groups={...parquet}, projection=[id, content, ...]
```

With one, the same query text plans as:

```
USearchExec: table=default::public::clean_docs::embedding, k=3, filtered=false
```

Observed behaviour of the rewrite:

| Query shape | Fast path? | Note |
|---|---|---|
| `SELECT id, content, <dist_fn>(embedding, ARRAY[...]) AS d … ORDER BY d ASC LIMIT k` | yes | the shape the vector store emits |
| Three-part reference, quoted (`"default"."public"."t"`) or unquoted | yes | quoting is not part of the match |
| Two-part `schema.table` reference | **reported no** | see below — not observed here |
| Same query plus `WHERE col = <literal>` | yes, `filtered=true` | the predicate is pushed **into** the index lookup |
| Projecting the `embedding` column | no | a vector column in the output declines the rewrite |
| Distance function that is not the index's metric | no | a cosine index does not serve `l2_distance` |
| No `LIMIT` | no | the bound is part of the matched shape |

The fallbacks are silent — a correct answer, computed by full scan, with no warning. So the
"no" rows are the ones worth guarding in code.

One path takes an embedding-projecting "no" deliberately:
`HotdataVectorStore.max_marginal_relevance_search` needs the stored vectors to compute
diversity, so its candidate fetch is a full scan whatever indexes exist, bounded by `fetch_k`.

**How the table is written matters, not just what it resolves to.** The rewrite builds its
lookup key from the reference as written, so only a full `catalog.schema.table` matches the key
the index was registered under. A two-part `schema.table` reference resolves to the same rows
and forfeits the index. **Not verified here** — this is reported against the optimizer rule in
[datafusion-vector-search-ext#32](https://github.com/hotdata-dev/datafusion-vector-search-ext/issues/32),
where the fix is to resolve the reference against session defaults before building the key.
The paths this package controls are unaffected either way: `HotdataVectorStore` hardcodes the
three-part form, and `search.py` rejects anything else. Only `hotdata_execute_sql`, where the
model writes the reference, is exposed, which is why its description asks for all three parts.

**Index creation is `HotdataClient.create_index(...)`** as of `hotdata-framework` 0.10.0, with
`index_type="vector"`, `metric=` and `columns=[...]`. It runs as an async job and **a failure
only appears on the job record**, not on the create call, so the submit reports success for
builds that later fail; `create_index` polls the job to a terminal state and raises with its
`error_message`. Before 0.10.0 this meant calling
`hotdata.IndexesApi(client.api).create_index(...)` and polling `JobsApi.get_job(id)` by hand.

Index *existence* is still not on the client. `IndexesApi(client.api).list_indexes(
connection_id, schema, table)` is how `HotdataVectorStore.create_index` checks before building.
Each listed entry carries `index_name`, `index_type`, `columns`, `metric`, `source_column` and
`status` (verified 2026-08-07 against two live indexes, one `vector` and one `bm25`). Two
details matter to a caller reading them:

- **`status` comes back as an `IndexStatus` enum, `metric` as a plain string.** Observed
  `status=<IndexStatus.READY: 'ready'>` alongside `metric='cosine'`. `IndexStatus` has exactly
  two members, `ready` and `pending`, so a listed index is not necessarily a built one.
- **`metric` is echoed in the same lowercase form it was requested in**, for `cosine`. The
  `l2` and `dot` renderings have not been observed.

**`dimensions` does not apply to a plain vector index.** When the indexed column already holds
vectors, the engine reads the width off the stored data; `dimensions` only picks an output
width for providers that support several. So an index must be built *after* the first write,
and a caller cannot assert the width.

**Known rough edge:** index creation fails with `could not detect dimension for 'embedding'`
after a **mixed upsert** — one load that rewrites ids the table already holds *while also*
adding new ones. Measured across a 25-probe matrix on fresh tables: 0 failures in 6 non-mixed
shapes (including a pure rewrite and a subset rewrite), 12 failures in 20 mixed runs. It is
**intermittent, not deterministic** — the same shape gave FAIL, PASS, PASS, PASS on four
identical tables — so a single passing control proves nothing, and an earlier round of
"hypotheses ruled out" on that basis is unsound. `List(Float32)` is indexable. Because the
width is read from data rather than supplied, there is no client-side workaround. Tracked as
[#52](https://github.com/hotdata-dev/hotdata-langchain/issues/52).

The failure surfaces only on the async job record (`JobsApi.get_job(id).error_message`); the
`create_index` call itself returns success with status `pending`.

**There is no cross-modality routing in the engine.** `LazyTableProvider::select_best_index` and
`IndexAwareManagedProvider::select_catalog_index` query the catalog with
`list_indexes(..., Some(IndexType::Sorted))` — they only ever see sorted indexes, and choose
index-scan versus table-scan. The code's own comment notes a proper cost model is still needed
(runtimedb#481). Nothing in the engine chooses between BM25, vector and sorted, and a grep for
hybrid/RRF/fusion across the engine finds nothing.

## Schema and index discovery

Working in SQL: `information_schema.tables`, `information_schema.columns` (with
`table_catalog`, `table_schema`, `table_name`, `column_name`, `ordinal_position`, `data_type`,
`is_nullable`), `SHOW TABLES`, and `DESCRIBE <table>`. `hotdata_langchain/schema.py` builds on
`information_schema.columns`, so it needs no extra permissions.

**Indexes are not visible in SQL** — no `pg_indexes`, no `information_schema.indexes`. They are
only reachable through the control plane, `IndexesApi.list_indexes(connection_id, schema, table)`,
which returns index name, type, columns and status. This is why an agent cannot currently
discover which columns are searchable, and why the search tool pins its corpus.

## Databases and workspaces

- **One client can query many databases.** `execute_sql(sql, database=...)` takes the scope per
  call; the same client read from two different managed databases in one session.
- **Cross-database references inside a single query fail** by default:
  `SELECT id FROM f1_db.public.drivers` from within another database's scope gives
  `table 'f1_db.public.drivers' not found`. `DatabasesApi` does expose
  `attach_database_catalog`/`detach_database_catalog`, and `bm25_search`'s scope resolution
  translates "an attachment alias or `default`", so attachment is presumably the supported route
  — **not verified here**.
- **Database names are not unique.** `name` is a display label; `resolve_managed_database` tries
  the id first and then scans `list_databases()` matching on name. Ids are the only safe handle,
  so this package never calls that resolver: `resolve_database_by_id` goes straight to
  `GET /databases/{id}`, and a resolved `ManagedDatabase` is what scopes every query.
- **`from_env()` picks a workspace silently** when `HOTDATA_WORKSPACE` is unset — first active,
  else first overall, no warning. `HotdataClient(api_key, workspace_id)` takes it explicitly.

## Error reporting

The framework raises `RuntimeError(e.reason)`, which is the bare HTTP reason (`"Bad Request"`).
The engine's actual message survives only in the underlying `ApiException`'s `body`, further down
the `__cause__` chain. This is not cosmetic: an agent shown `"Bad Request"` cannot correct
itself, while the real text (`Invalid function 'to_tsvector'…`) is directly actionable. See the
cross-repo list in [`ai-native-layer-roadmap.md`](./ai-native-layer-roadmap.md).

`hl.engine_error_message(exc)` walks that chain and returns the engine's text, and
`make_hotdata_tools(handle_errors=True)` hands it to the model instead of raising. That is a
recovery, not a fix — the framework surfacing its own message is tracked in
[#36](https://github.com/hotdata-dev/hotdata-langchain/issues/36), and until it lands every
consumer either uses this or sees `"Bad Request"`.

## What an agent does without guidance

Both observed with a small tool-calling model and the tools from `make_hotdata_tools`:

- **It matches text in SQL.** With a one-line SQL tool description — even *with* a system prompt
  spelling out the rule — it wrote `to_tsvector`/`plainto_tsquery`, the query failed, and the
  exception aborted the whole LangGraph run. With the constraint in the SQL tool's own
  description it uses the search tool correctly, with no system-prompt guidance at all.
- **It guesses column names.** It produced `AVG(review_scores_rating)` for a column that was
  never in any tool output — correct only because the SF Airbnb fixture is a well-known public
  dataset. On proprietary data that guess fails. With `hotdata_describe_tables` registered it
  calls the overview, drills into the table, and then writes the query.
