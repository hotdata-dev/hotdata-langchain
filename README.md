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
```

For a full end-to-end run against a real workspace — data load, BM25 index build, then an
agent choosing between search and SQL — see [`demo/`](demo/README.md).

## Development

```bash
uv sync --locked
uv run pytest
```
