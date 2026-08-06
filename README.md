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
| `hotdata_load_managed_table` | Load a parquet file into a managed table, addressed by database id |
| `hotdata_describe_tables` | List tables, or one table's columns and types |
| `hotdata_search_text` | Full-text search an indexed column, ranked by relevance (opt-in — see below) |

The descriptions carry the engine's contract — dialect, what SQL can and cannot do, and where
to look things up — so an agent does not need a system prompt explaining the query engine.

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
    "file": "/path/to/orders.parquet",
})
```

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

Inside a managed database the built-in catalog is always `default`, so a managed table reads
as `default.<schema>.<table>` when `database_id=` scopes the query to it.

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

It is a primitive, not a tool — it is not part of `make_hotdata_tools` and an agent never calls
it directly. You compose it into a chain, or hand `as_retriever()` to something that expects a
retriever:

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
usable the moment you create it. Today every search is a full scan.

It is also written to match the shape the engine's optimizer rewrites into an HNSW index
lookup: a plain column, a literal `ARRAY[...]`, `ASC`, a `LIMIT`, no vector column in the
output, and an index built on the same metric. The intent is that the same query gets faster
once such an index exists, with nothing in your code changing. **That rewrite has not yet been
confirmed end to end for these queries** — the conditions come from reading the engine's
optimizer rule, and verifying it needs an index this package cannot yet create. Tracked in
[`docs/vectorstore-plan.md`](docs/vectorstore-plan.md); until then, treat the fast path as the
design intent rather than a measured property.

Provisioning an index is not part of this package yet; create one through the Hotdata API or
CLI, matching its metric to the `distance=` you configured.

`distance=` accepts `"cosine"` (default), `"l2"` and `"dot"`. Prefer `cosine`: its relevance
score is exact, whereas the engine's `l2_distance` is *squared* L2 and LangChain's Euclidean
relevance score expects true Euclidean distance, so `similarity_search_with_relevance_scores`
under `l2` returns scores on the wrong scale. Ranking is correct under all three.

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
`KeyError`. Ids come from `client.list_managed_databases()`, the
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
