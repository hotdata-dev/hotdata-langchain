# Engine contract — what the SQL and search surface actually does

Every claim here was checked against a live workspace (`api.hotdata.dev`, RuntimeDB behind it)
rather than read off a spec, because several of them contradict what the docs or the dialect
imply. They are the facts the tool descriptions in `hotdata_langchain/` encode, so if one of
them changes, a description somewhere is now lying to the model.

Last verified 2026-07-27 against `hotdata-framework` 0.9.0 / `hotdata` 0.8.0.

## SQL

Postgres dialect, and the following are confirmed working: joins, CTEs, subqueries, `GROUP BY`,
window functions, `ORDER BY`/`LIMIT`, ordinary scalar functions, `LIKE` and `ILIKE`, and
schema-qualified as well as bare table names.

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
  the id first and then scans `list_databases()` matching on name. Ids are the only safe handle.
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
