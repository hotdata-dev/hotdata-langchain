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
    "tables": "orders\ncustomers",
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

## Run the examples

```bash
uv run python examples/langchain_basic.py
uv run python examples/langchain_managed_db.py
```

## Development

```bash
uv sync --locked
uv run pytest
```
