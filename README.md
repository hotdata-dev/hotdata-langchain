# hotdata-langchain

Connect [LangChain](https://python.langchain.com/) to [Hotdata](https://hotdata.dev) — tools that let an agent run SQL against your workspace connections, full-text search indexed columns and work with managed databases, plus a `VectorStore` implementation so Hotdata can back any LangChain retriever or chain.

## Install

```bash
pip install hotdata-langchain
```

## Authentication

Set `HOTDATA_API_KEY` in your environment. Optionally set `HOTDATA_WORKSPACE` to pin a specific workspace (the first available workspace is used if unset).

## Quickstart

Of the LangChain packages, this one needs only `langchain-core`, and works with any
tool-calling model. Running an agent additionally needs the `langchain` package and the
integration for whichever model provider you use; using `HotdataVectorStore` needs an
embedding provider's integration, such as `langchain-openai`.

```python
from langchain.agents import create_agent
import hotdata_langchain as hl

client = hl.from_env()
tools = hl.make_hotdata_tools(client, database_id="dbid...")

agent = create_agent(model=your_model, tools=tools)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Which product categories have the most orders?"}]}
)
print(result["messages"][-1].content)
```

Queries run against a database scope, so pass `database_id=` (a managed database id).
`hl.from_env().list_managed_databases()` shows what is available in the workspace, with the
id of each.

## Tools

`make_hotdata_tools(client)` returns a list of LangChain `StructuredTool` objects ready to pass to any agent:

| Tool | What it does |
|------|-------------|
| `hotdata_execute_sql` | Run a SQL query and return rows as JSON |
| `hotdata_list_managed_databases` | List available managed databases, with the id of each |
| `hotdata_create_managed_database` | Create a new managed database and return its id |
| `hotdata_load_managed_table` | Load a parquet file — local path or URL — into a managed table, addressed by database id |
| `hotdata_describe_tables` | List tables, or one table's columns, types and how many rows hold a value |
| `hotdata_search_text` | Full-text search an indexed column, ranked by relevance (opt-in — see below) |

The descriptions carry the engine's contract — dialect, what SQL can and cannot do, and where
to look things up — so an agent does not need a system prompt explaining the query engine.

Every name in that table is exported as a constant (`hl.DEFAULT_SQL_TOOL_NAME` and so on), so
an application selecting a subset does not have to hardcode strings.

## Choosing which tools an agent gets

An agent that reads one fixed database cannot use the three managed-database tools, and giving
it tools it can only misuse costs context on every turn. Two flags decide the set:

```python
tools = hl.make_hotdata_tools(client, database_id="dbid...")                        # all of them
tools = hl.make_hotdata_tools(client, database_id="dbid...", management_tools=False)  # SQL + describe
tools = hl.make_hotdata_tools(client, database_id="dbid...", describe_tables=False)
```

`management_tools=False` drops listing, creating and loading. It is not called `read_only`:
listing databases is itself a read, so the set it removes is the managed-database workflow
rather than everything that writes.

## Letting the model recover from a failed call

The tools raise on failure, which is right in a script and wrong in an agent: an exception out
of a tool aborts the whole LangGraph run, so one invalid query ends the conversation instead of
costing a turn. `handle_errors=True` returns the failure as `{"error": "..."}` instead:

```python
tools = hl.make_hotdata_tools(client, database_id="dbid...", handle_errors=True)
```

What reaches the model is the engine's own message, not the framework's `RuntimeError("Bad
Request")` — `Invalid function 'date_sub'. Did you mean 'date_bin'?` is something a model can
act on, and was observed correcting its query on the next call. The exception is a failure this
package can explain better than the engine can, such as a [date format
pattern](#warnings-about-results-that-are-not-what-they-look-like) that will not be
interpreted: that message leads, with the engine's after it. `hl.engine_error_message(exc)`
exposes that lookup on its own, and `hl.with_error_feedback(tools)` applies the same wrapping to
tools built elsewhere, such as a retriever tool registered alongside these:

```python
from langchain_core.tools import create_retriever_tool

tools = hl.with_error_feedback([*tools, create_retriever_tool(retriever, "search_docs", "...")])
```

Both `func` and `coroutine` are wrapped. Wrapping only the sync callable is a trap: LangChain
prefers `coroutine` under async, which is how `langgraph dev` and a deployed Agent Server run,
so a sync-only wrapper goes unused in exactly the environment that needs it.

Only failures are touched. A successful result comes back exactly as the tool produced it, so a
tool declaring `response_format="content_and_artifact"` keeps its pair intact, and LangGraph's
control-flow exceptions are re-raised — a tool calling `interrupt()` for human approval still
pauses the graph rather than reporting the pause as an error.

## Letting the agent discover the schema

`hotdata_describe_tables` is registered by default. Called with no arguments it lists every
table with its column count; called with a table name it returns that table's columns and
types. Without it an agent has to guess column names, and a guess that misses fails the query.

```python
tools = hl.make_hotdata_tools(client, database_id="dbid...")            # included
tools = hl.make_hotdata_tools(client, database_id="dbid...", describe_tables=False)  # omitted
```

It reads `information_schema` in whichever database the tools are scoped to, so it needs no
extra permissions. With it turned off, the SQL tool's description tells the agent to query
`information_schema` directly instead.

Each column also reports `non_null`, how many of the table's rows hold a value in it,
alongside the table's `row_count`:

```json
{
  "table": "public.listings",
  "row_count": 7535,
  "columns": [
    {"name": "id", "type": "Int64", "non_null": 7535},
    {"name": "price", "type": "Float64", "non_null": 0}
  ]
}
```

A type alone says a column exists, not that anything is in it — an agent given only types was
measured recommending an analysis of a column that is NULL on every row. The counts cost one
aggregate query per table described; pass `describe_column_stats=False` where describing a
table must not scan it.

A table that is declared on the database but has never been loaded has no columns at all in
`information_schema`, which reads as a missing table. It is reported as declared and empty
instead, which is the state a load fixes.

## Calling tools directly

You can also invoke tools outside of an agent loop:

```python
import json

tools = {t.name: t for t in hl.make_hotdata_tools(client, database_id="dbid...")}

result = tools["hotdata_execute_sql"].invoke({"sql": "SELECT * FROM orders LIMIT 10"})
print(result)  # JSON rows

created = tools["hotdata_create_managed_database"].invoke({
    "name": "sales",            # a display label, not an identifier
    "schema_name": "public",
    "tables": "orders,customers",
})

tools["hotdata_load_managed_table"].invoke({
    "database_id": json.loads(created)["id"],
    "table": "orders",
    "file": "/path/to/orders.parquet",   # or "https://example.com/orders.parquet"
})
```

A URL is downloaded and uploaded for you, and the temporary copy is removed afterwards whether
or not the load succeeds. This is the form that works from a deployed agent: an Agent Server
has no filesystem the requesting user can put a file on, so a path-only load can ingest nothing
the process did not already hold. Only parquet is accepted, and a URL that answers 200 with an
HTML login or error page is rejected before anything is uploaded.

Two limits apply, because the URL is chosen by the model and a model's inputs include whatever
text it retrieved — an instruction planted in a document is enough to pick one:

- **It must resolve to a public address.** Otherwise the agent process becomes a fetcher for
  whatever its own network can see, which in a deployment is usually more than the public
  internet: a cloud metadata endpoint on `169.254.169.254`, an internal service on a private
  range. A load completes the loop, since an internal URL serving parquet would land in the
  workspace and be readable from SQL on the next turn. Every address the host resolves to is
  checked, and again on each redirect — a public URL that 302s inwards is the standard bypass.
  Pass `allow_private_hosts=True` to `make_hotdata_tools` when your data genuinely sits on an
  internal host.
- **It is capped at 1 GiB.** `Content-Length` is checked first, so an oversized file is usually
  refused before any of it transfers, and the stream is counted as well because that header is
  optional and can lie. `hl.databases.fetch_parquet(..., max_bytes=...)` raises the cap.

This narrows what the fetch can reach rather than sealing it: the address is resolved for the
check and again by the HTTP client, so a DNS server answering differently each time can still
get through. A deployment on an untrusted network wants an egress proxy in front of this.

## Full-text search

Point the agent at a text column carrying a BM25 index and it gets a search tool alongside SQL:

```python
tools = hl.make_hotdata_tools(
    client,
    database_id="dbid...",
    search_table="default.public.listings",   # catalog.schema.table
    search_column="description",              # must have a BM25 index
    search_columns=["id", "name", "price", "description"],  # what each hit returns
    search_k=5,
)

hits = {t.name: t for t in tools}["hotdata_search_text"].invoke(
    {"query": "cozy apartment with a view"}
)
```

Rows come back ranked, each with a `score`. The agent supplies only `query` and an optional
`k`; the table and column are fixed when you build the tool. That is deliberate — nothing in
the tool surface lets an agent discover which columns are indexed, and the engine errors
outright rather than falling back to a scan when a column has no BM25 index.

`search_columns` is optional. Left unset, a hit carries the searched column plus the table's
`id`, so it can be joined back to the table it came from — a hit holding only the matched text
references nothing. The key column is looked up once when the tools are built and dropped if
the table has no such column; `search_key_column="listing_id"` names a different one, and
`search_key_column=None` returns the searched column alone.

`k` is governed by `max_rows`: a larger `k` from the model is cut to it *before* the search
runs, so those rows are never ranked at all. The ceiling is stated in the tool's description
and in its `k` argument, and a call that was cut says so in `metadata.client_warning`. To
reason over a wider cohort, call `bm25_search` inside SQL and aggregate there.

A managed database's tables read as `default.<schema>.<table>`. An **attached** source's do
not — its tables answer to the attachment's alias, and `default.<schema>.<table>` is not
found there. Nothing on the database record distinguishes the two, so there is no constant
the tools can assume.

`make_hotdata_tools` therefore reads the catalog from `information_schema` once, when the
tools are built, and states it in the SQL tool's description — so the model is told the real
catalog rather than a rule that holds for only one kind of database. Pass `catalog="…"` to
skip that lookup.

Write all three parts either way: a two-part `schema.table` reference resolves and returns the
same rows, but the engine matches its index lookup on the reference as written, so the short
form can quietly forfeit an index. `HotdataVectorStore` and the search tool emit the full form
themselves.

For more than one searchable corpus, build the tools yourself and give each a distinct name
and description — the agent then routes on the descriptions:

```python
tools = [
    # Configure the first corpus here, so the SQL tool's description still names a search
    # tool to defer text matching to. Passing no search_table/search_column drops that,
    # and the agent goes back to trying to match text in SQL.
    *hl.make_hotdata_tools(
        client,
        database_id="dbid...",
        search_table="default.public.listings",
        search_column="description",
        search_tool_name="search_listings",
    ),
    hl.make_hotdata_search_tool(
        client, table="default.public.reviews", column="comments",
        name="search_reviews", database_id="dbid...",
    ),
]
```

Provisioning the index itself is not yet part of this package; create it through the Hotdata
API or CLI. `demo/` has a script that does the whole flow — managed database, data load, BM25
index, then an agent that picks between search and SQL.

## Vector store

`HotdataVectorStore` implements LangChain's `VectorStore`, so Hotdata works as the retrieval
backend for any retriever, chain or eval built on that interface.

It is a primitive rather than a tool: it is not part of `make_hotdata_tools`, and a model cannot
call it directly because it has no name, description or argument schema. You compose it into a
chain, hand `as_retriever()` to anything expecting a retriever, or wrap it as a tool so an agent
*can* call it — see [below](#letting-an-agent-search-the-store).

```python
from langchain_openai import OpenAIEmbeddings

store = hl.HotdataVectorStore(
    client,
    OpenAIEmbeddings(model="text-embedding-3-small"),
    database_id="dbid...",
    table="documents",
)

store.add_texts(
    ["Cozy studio with great light", "Two-bedroom near the park"],
    [{"city": "sf"}, {"city": "nyc"}],
)

docs = store.similarity_search("somewhere bright to stay", k=3)
retriever = store.as_retriever(search_kwargs={"k": 3})   # composes into any chain
```

Rows are stored in one managed table keyed on `id`, so re-adding a document with an existing
id replaces it rather than duplicating it. `delete(ids=[...])` requires ids — there is no
delete-everything call.

The store declares that table itself. If you pre-create the database, leave the table out of
`tables=[...]` and let the store declare it, or declare it with `key=["id"]` yourself — a
managed table with no key takes writes as appends, so re-adding a document would duplicate it,
and an existing table's key cannot be read back to warn you.

Searches run as a single SQL query using the engine's scalar distance functions:

```sql
SELECT id, content, metadata_json,
       cosine_distance(embedding, ARRAY[...]) AS dist
FROM "default"."public"."documents"
ORDER BY dist ASC
LIMIT 4
```

That query is correct with **no index at all** — it brute-forces the table — so a store is
usable the moment you create it, before any indexing exists.

Once a vector index built on the same metric exists on the embedding column, the engine
rewrites that identical query into an index lookup, with nothing in your code changing. This
is confirmed against a live engine: the query plan switches to a `USearchExec` node, and a
`WHERE` filter is pushed *into* the index lookup rather than costing you the fast path. See
[`docs/engine-contract.md`](docs/engine-contract.md) for the observed plans.

Three things forfeit the rewrite and fall back to a full scan, silently and without error:
projecting the raw `embedding` column, querying with a distance function the index was not
built for, and omitting `LIMIT`. Similarity search does none of them; MMR
[does the first](#diverse-results-with-mmr), by necessity.

The store builds that index for you:

```python
store.create_index()                       # or, in one step:
store = hl.HotdataVectorStore.from_texts(
    texts, embeddings, client=client, database_id="dbid...", create_index=True
)
```

Build it *after* the first write. The engine reads the vector width off stored data, so there
is nothing to measure before then. The metric always comes from this store's `distance`, which
is what earns the rewrite — leaving it to the server would build an `l2` index, its default,
that never serves a `cosine` search. Calling `create_index()` when a matching index already
exists does nothing and returns `None`, so it is safe on every start-up; an index that already
exists under a *different* metric raises, since only you know whether the index or the
`distance=` is the mistake.

Builds are polled to completion, up to `timeout_s=900`. Pass `wait=False` to return as soon as
the build is accepted and check the job yourself.

`distance=` accepts `"cosine"` (default), `"l2"` and `"dot"`. Prefer `cosine`: its relevance
score is exact, whereas the engine's `l2_distance` is *squared* L2 and LangChain's Euclidean
relevance score expects true Euclidean distance, so `similarity_search_with_relevance_scores`
under `l2` returns scores on the wrong scale. Ranking is correct under all three.

### Diverse results with MMR

The `k` nearest documents are often near-duplicates of each other — all genuinely close to
the query, all making the same point. Maximal marginal relevance ranks a wider pool by
distance, then picks `k` from it one at a time, scoring each candidate against both the query
and what it has already picked:

```python
docs = store.max_marginal_relevance_search("somewhere bright to stay", k=3, fetch_k=20)

retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 20})
```

`lambda_mult` is the balance: `1.0` is pure relevance, `0.0` is pure variety. `fetch_k` is the
candidate pool, and is raised to `k` if you pass less. `filter=` works the same as it does on
`similarity_search`. Results come back in selection order — only the first is the nearest to
the query, and a later pick is often further away than one it was chosen over.

This is the one search that reads the stored vectors, which is what MMR needs and what
forfeits the index lookup — the candidate fetch is a full scan even where an index exists,
bounded by `fetch_k`. Use it where variety in the retrieved set matters more than the cost of
scanning; `similarity_search` stays the fast path.

Both halves of that score use cosine similarity whatever `distance=` is set to. That is
LangChain's own convention, shared by every implementation of this interface: under `l2` the
candidate pool is L2-nearest while the selection among those candidates is cosine-based. So
`lambda_mult=1.0` gives back this store's similarity ranking under `cosine` only — under `l2`
and `dot` it reorders the candidate pool by cosine instead.

**Expect to tune `lambda_mult` upward.** The `0.5` default is LangChain's, kept so code ported
from another vector store behaves identically. It weights relevance and variety equally, and
those two terms rarely have equal spread: an embedding model that packs its distances into a
narrow band leaves the variety term varying far more than the relevance term, so variety
quietly decides most picks. On the demo corpus through `text-embedding-3-small` every distance
fell between 0.60 and 0.67, and `0.5` promoted a listing that did not answer the question at
all, while `0.7` and `0.8` both dropped a near-duplicate for a genuine alternative. That is one
corpus and one model — a reason to sweep the value on your own data, not a number to copy.

### Letting an agent search the store

The store is not a tool, but a retriever becomes one with LangChain's own
`create_retriever_tool` — so an agent decides *whether* to search and *what* to search for,
alongside the SQL tools:

```python
from langchain_core.tools.retriever import create_retriever_tool

search_docs = create_retriever_tool(
    store.as_retriever(search_kwargs={"k": 4}),
    name="search_listings",
    description="Find listings whose description matches what the guest is describing.",
)

tools = [*hl.make_hotdata_tools(client, database_id="dbid..."), search_docs]
```

Use a chain when every question needs the corpus — one retrieval, predictable cost. Wrap it as
a tool when the model should choose, reformulate a query, or search more than once.

Note the two return different things: `create_retriever_tool` gives the model concatenated
document text, whereas `hotdata_search_text` returns the `{"metadata", "rows"}` envelope the
other Hotdata tools use, so values from a hit can be carried into a follow-up SQL query.

### Filtering on metadata

Metadata always round-trips in full. To *filter* on a key, declare it up front so it is stored
as a real typed column:

```python
store = hl.HotdataVectorStore(
    client,
    embeddings,
    database_id="dbid...",
    metadata_columns={"city": "string", "beds": "int"},
)

store.similarity_search("bright and quiet", k=3, filter={"city": "sf"})
```

Equality only, for now. Filtering on an undeclared key raises `ValueError` rather than quietly
returning unfiltered results. The predicate goes into the search query itself, not around it —
filtering *after* a top-k selection can only shrink the result, never re-fill it back to `k`.

`metadata_columns` has to match the table it points at. An upsert must carry every column the
table has, so opening an existing store with different promoted columns fails on the first
write with `upload is missing column '<name>'`.

## Scoping queries to a managed database

`database_id=` scopes all SQL the agent runs to one managed database. The API requires a
database scope, so queries fail with `a database is required` without it:

```python
tools = hl.make_hotdata_tools(client, database_id="dbid...")
```

**Databases are addressed by id, never by name.** A database name is a display label and is
not unique, so a name lookup can silently resolve to the wrong database — and the agent's
`hotdata_load_managed_table` overwrites the table it loads into. Passing a name raises
`KeyError` — or, under `handle_errors=True`, returns it to the model rather than resolving
anything. Ids come from `client.list_managed_databases()`, the
`hotdata_list_managed_databases` tool, or the response of a create.

The id is resolved once when the tools are built, so a bad id fails there rather than on the
agent's first query, and no query pays a repeat lookup. If you already hold a
`ManagedDatabase` — from `list_managed_databases()` or `create_managed_database()` — pass it
instead of its id to skip the lookup entirely:

```python
db = client.create_managed_database(description="sales", schema="public", tables=["orders"])
tools = hl.make_hotdata_tools(client, database_id=db)
```

## Controlling result size

Limit how many rows are returned to the LLM. Useful for keeping responses within context limits (default: 100):

```python
tools = hl.make_hotdata_tools(client, max_rows=50)
```

The cap is stated in the SQL tool's description, and `metadata.row_count` is the total the
query matched *before* it — so a result that was cut says so twice, in the numbers and in
`metadata.client_warning`.

## Warnings about results that are not what they look like

Some calls succeed while quietly meaning something other than what was asked: rows cut at the
cap, a `k` clamped before the search ran, a date format pattern the engine will not interpret.
Each of those returns an ordinary successful result, so nothing signals that anything went
wrong. They are reported in `metadata.client_warning`:

```json
{
  "metadata": {
    "row_count": 7535,
    "warning": null,
    "client_warning": "Returned the first 100 rows of the 7535 this query matched. ..."
  },
  "rows": ["… the first 100 …"]
}
```

`warning` is the engine's own field and is passed through untouched; `client_warning` is this
package's, so the two never overwrite each other and a consumer can tell which side noticed.
The key is absent when there is nothing to say.

The format check is the one worth knowing about outside an agent loop. The engine's date
patterns are strftime, so `to_char(d, 'YYYY-MM-DD')` is not a pattern at all — it returns the
literal text `YYYY-MM-DD` on every row, with no error. Any format pattern containing no `%`
is flagged, with the strftime equivalent when it can be worked out.

The same query applied to a *column* rather than a literal fails instead of returning a wrong
value, and the engine answers with nothing more specific than an internal error. So the hint
is raised on that path too, as a `HotdataToolError` carrying it ahead of the engine's message
— which is what `handle_errors=True` then hands to the model.

## Run the examples

```bash
uv run python examples/langchain_basic.py
uv run python examples/langchain_managed_db.py

# needs an embedding provider key and the langchain-openai integration
uv run --group demo python examples/langchain_vectorstore.py
```

For full end-to-end runs against a real workspace, see [`demo/`](demo/README.md): one takes a
workspace from empty through a data load and BM25 index build to an agent choosing between
search and SQL; the other writes embedded documents into a managed table and answers a
question with a stock LangChain retrieval chain over `HotdataVectorStore`.

## Development

```bash
uv sync --locked
uv run pytest
```
