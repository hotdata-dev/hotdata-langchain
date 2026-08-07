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

Two constraints matter enough to state in a tool description:

- **An aggregate query must reference at least one column.** `SELECT COUNT(*) FROM t` and
  `SELECT COUNT(1) FROM t` are rejected with `must either specify a row count or at least one
  column`; `SELECT COUNT(id) FROM t`, `SELECT MIN(id), MAX(id) FROM t` and
  `SELECT room_type, COUNT(*) … GROUP BY room_type` all work. The failing shape is the one an
  agent writes first, so the description gives the workaround.
- **There is no full-text matching in SQL.** No `to_tsvector`, no `plainto_tsquery`. The engine
  answers with `Invalid function 'to_tsvector'. Did you mean 'to_char'?`. `LIKE`/`ILIKE` work as
  substring tests but cannot rank.

**A database scope is required.** An unscoped query fails with `a database is required: set the
X-Database-Id header or the database_id body field`. Inside a managed database the built-in
catalog is always `default`.

## Full-text search

```sql
bm25_search('catalog.schema.table', 'column', 'query text' [, limit])
```

Returns the table's columns plus a trailing `score` (Float32). Three properties shape
`hotdata_langchain/search.py`:

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

**`dimensions` does not apply to a plain vector index.** When the indexed column already holds
vectors, the engine reads the width off the stored data; `dimensions` only picks an output
width for providers that support several. So an index must be built *after* the first write,
and a caller cannot assert the width.

**Known rough edge:** one table failed index creation with `could not detect dimension for
'embedding'` and reproduced on retry, while six tables created fresh — including ones with
repeated upserts, a delete, and promoted metadata columns — all indexed successfully. The
trigger was not identified. `List(Float32)` is indexable; the failure is specific to some
table state, not to the column type. Because the width is read from data rather than supplied,
there is no client-side workaround. Tracked as
[#52](https://github.com/hotdata-dev/hotdata-langchain/issues/52).

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
