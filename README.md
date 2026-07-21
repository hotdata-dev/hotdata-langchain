# hotdata-langchain

Give your [LangChain](https://python.langchain.com/) agents access to [Hotdata](https://hotdata.dev) — run SQL against your workspace connections and work with managed databases.

## Install

```bash
pip install hotdata-langchain
```

## Authentication

Set `HOTDATA_API_KEY` in your environment. Optionally set `HOTDATA_WORKSPACE` to pin a specific workspace (the first available workspace is used if unset).

## Quickstart

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
import hotdata_langchain as hl

client = hl.from_env()
tools = hl.make_hotdata_tools(client)

agent = create_tool_calling_agent(llm=your_llm, tools=tools, prompt=your_prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": "How many rows are in the orders table?"})
```

## Tools

`make_hotdata_tools(client)` returns a list of LangChain `StructuredTool` objects ready to pass to any agent:

| Tool | What it does |
|------|-------------|
| `hotdata_execute_sql` | Run a SQL query and return rows as JSON |
| `hotdata_list_managed_databases` | List available managed databases |
| `hotdata_create_managed_database` | Create a new managed database |
| `hotdata_load_managed_table` | Load a parquet file into a managed table |

## Calling tools directly

You can also invoke tools outside of an agent loop:

```python
tools = {t.name: t for t in hl.make_hotdata_tools(client)}

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

## Scoping queries to a managed database

Pass `database=` so all SQL the agent runs resolves against a specific managed database:

```python
tools = hl.make_hotdata_tools(client, database="sales")
```

## Controlling result size

Limit how many rows are returned to the LLM. Useful for keeping responses within context limits (default: 100):

```python
tools = hl.make_hotdata_tools(client, max_rows=50)
```

## Caching tool calls

LangChain caches LLM calls (`set_llm_cache`) but has no equivalent for tool calls — every
`StructuredTool` invocation re-executes, even when an agent retries the same query.
`HotdataToolCache` fills that gap using a Hotdata managed table as the backing store, keyed
by tool name and arguments:

```python
from hotdata_langchain import HotdataToolCache

cache = HotdataToolCache(client)  # construct once per process and reuse
tools = hl.make_hotdata_tools(client, cache=cache)
```

This serves repeated calls to the read-only tools (`hotdata_execute_sql`,
`hotdata_list_managed_databases`) from the cache instead of re-running them.
`hotdata_create_managed_database` and `hotdata_load_managed_table` are never cached —
skipping a mutation on a cache hit would be a correctness bug, not caching.

Pass `cache_ttl=timedelta(...)` to `make_hotdata_tools` to expire entries after a fixed age.

### Caching arbitrary tools

`cached()` works on any plain function or tool, not just this package's own — Hotdata can
act as a shared, queryable cache backend for database queries, API calls, or search
results from any part of your agent:

```python
from hotdata_langchain.cache import cached

cached_search = cached(my_search_function, cache=cache, tool_name="search")
```

### Known limitations

- **Database-resolution race**: on first use, `HotdataToolCache` resolves-or-creates a
  managed database by name. Managed-database names are not unique or identifying, so two
  processes racing on first use can each create a distinct database with the same name,
  silently splitting the cache. Pass `database_id=` (a database you created once, e.g. at
  deploy time) to pin a specific database and avoid this entirely — recommended for any
  multi-process deployment.
- **Shared-table write ceiling**: managed-table loads serialize at the per-table lock
  regardless of key, so all cache writes funnel through one lock. Fine for moderate
  concurrency; sharding by tool/hash-prefix is not implemented.
- **Per-instance memoization**: construct one long-lived `HotdataToolCache` per process and
  reuse it — a fresh instance per call defeats memoization and re-races the database
  resolution on every call.

## Run the examples

```bash
uv run python examples/langchain_basic.py
uv run python examples/langchain_managed_db.py
uv run python examples/langchain_cached_tools.py
```

## Development

```bash
uv sync --locked
uv run pytest
```
