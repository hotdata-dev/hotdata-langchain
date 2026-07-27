# hotdata-langchain

Give your [LangChain](https://python.langchain.com/) agents access to [Hotdata](https://hotdata.dev) — run SQL against your workspace connections, full-text search indexed columns, and work with managed databases.

## Install

```bash
pip install hotdata-langchain
```

## Authentication

Set `HOTDATA_API_KEY` in your environment. Optionally set `HOTDATA_WORKSPACE` to pin a specific workspace (the first available workspace is used if unset).

## Quickstart

The package itself depends only on `langchain-core`, and works with any tool-calling model.
Running an agent additionally needs the `langchain` package and the integration for whichever
model provider you use.

```python
from langchain.agents import create_agent
import hotdata_langchain as hl

client = hl.from_env()
tools = hl.make_hotdata_tools(client, database="sales")

agent = create_agent(model=your_model, tools=tools)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Which product categories have the most orders?"}]}
)
print(result["messages"][-1].content)
```

Queries run against a database scope, so pass `database=` (a managed database name or id).
`hl.from_env().list_managed_databases()` shows what is available in the workspace.

## Tools

`make_hotdata_tools(client)` returns a list of LangChain `StructuredTool` objects ready to pass to any agent:

| Tool | What it does |
|------|-------------|
| `hotdata_execute_sql` | Run a SQL query and return rows as JSON |
| `hotdata_list_managed_databases` | List available managed databases |
| `hotdata_create_managed_database` | Create a new managed database |
| `hotdata_load_managed_table` | Load a parquet file into a managed table |
| `hotdata_describe_tables` | List tables, or one table's columns and types |
| `hotdata_search_text` | Full-text search an indexed column, ranked by relevance (opt-in — see below) |

The descriptions carry the engine's contract — dialect, what SQL can and cannot do, and where
to look things up — so an agent does not need a system prompt explaining the query engine.

## Letting the agent discover the schema

`hotdata_describe_tables` is registered by default. Called with no arguments it lists every
table with its column count; called with a table name it returns that table's columns and
types. Without it an agent has to guess column names, and a guess that misses fails the query.

```python
tools = hl.make_hotdata_tools(client, database="sales")            # included
tools = hl.make_hotdata_tools(client, database="sales", describe_tables=False)  # omitted
```

It reads `information_schema` in whichever database the tools are scoped to, so it needs no
extra permissions. With it turned off, the SQL tool's description tells the agent to query
`information_schema` directly instead.

## Calling tools directly

You can also invoke tools outside of an agent loop:

```python
tools = {t.name: t for t in hl.make_hotdata_tools(client, database="sales")}

result = tools["hotdata_execute_sql"].invoke({"sql": "SELECT * FROM orders LIMIT 10"})
print(result)  # JSON rows

tools["hotdata_create_managed_database"].invoke({
    "name": "sales",
    "schema_name": "public",
    "tables": "orders,customers",
})

tools["hotdata_load_managed_table"].invoke({
    "database": "sales",
    "table": "orders",
    "file": "/path/to/orders.parquet",
})
```

## Full-text search

Point the agent at a text column carrying a BM25 index and it gets a search tool alongside SQL:

```python
tools = hl.make_hotdata_tools(
    client,
    database="sf_airbnb",
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
as `default.<schema>.<table>` when `database=` scopes the query to it.

For more than one searchable corpus, build the tools yourself and give each a distinct name
and description — the agent then routes on the descriptions:

```python
tools = [
    *hl.make_hotdata_tools(client, database="sf_airbnb"),
    hl.make_hotdata_search_tool(
        client, table="default.public.listings", column="description",
        name="search_listings", database="sf_airbnb",
    ),
    hl.make_hotdata_search_tool(
        client, table="default.public.reviews", column="comments",
        name="search_reviews", database="sf_airbnb",
    ),
]
```

Provisioning the index itself is not yet part of this package; create it through the Hotdata
API or CLI. `demo/` has a script that does the whole flow — managed database, data load, BM25
index, then an agent that picks between search and SQL.

## Scoping queries to a managed database

`database=` resolves all SQL the agent runs against a specific managed database, by name or
id. The API requires a database scope, so queries fail with `a database is required` without
it:

```python
tools = hl.make_hotdata_tools(client, database="sales")
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
```

For a full end-to-end run against a real workspace — data load, BM25 index build, then an
agent choosing between search and SQL — see [`demo/`](demo/README.md).

## Development

```bash
uv sync --locked
uv run pytest
```
