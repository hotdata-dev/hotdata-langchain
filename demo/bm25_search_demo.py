"""End-to-end demo of the Hotdata BM25 search tool, from empty workspace to agent.

Stands up everything the tool needs, then exercises it twice — once by invoking the
tool directly, once through a LangChain agent that also has the SQL tool and has to
pick between them.

    uv run --group demo --env-file .env python demo/bm25_search_demo.py

The agent step works with any tool-calling model — pass one with --model or DEMO_MODEL
— and is skipped when its provider key is not set; every step before it needs only
HOTDATA_API_KEY. Set LANGSMITH_API_KEY and LANGSMITH_TRACING=true to trace the run to
LangSmith.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import hotdata

import hotdata_langchain as hl

PARQUET_URL = "https://www.hotdata.dev/data/sf-airbnb-listings.parquet"
USER_AGENT = "hotdata-langchain-demo"
DATABASE = "langchain_bm25_demo"
SCHEMA = "public"
TABLE = "listings"
SEARCH_COLUMN = "description"
INDEX_NAME = "listings_description_bm25"

# The managed database is addressable as the `default` catalog inside its own scope,
# so the search tool's table reference is catalog-qualified against that.
TABLE_REF = f"default.{SCHEMA}.{TABLE}"

# Narrow the columns each hit returns: the listings table is 85 columns wide and all
# of them would land in the agent's context. Filtered against the real schema below.
# `price` is excluded deliberately — it is NULL for every row in this fixture.
PREFERRED_COLUMNS = ["id", "name", "room_type", "neighbourhood_cleansed", "description"]

DEFAULT_QUERY = "cozy apartment with a view"
# Env var holding the provider's key, looked up from the provider prefix of --model so the
# agent step can be skipped with a useful message rather than failing inside the provider.
# An unlisted provider simply skips the pre-check.
PROVIDER_KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
}
INDEX_TIMEOUT_SECONDS = 600

# Deliberately not answerable from search results alone: the ratings and counts span
# every listing in the matched neighbourhoods, not just the handful search returned.
# That forces the agent to use search to identify the neighbourhoods and SQL to
# aggregate over them, which is the routing behaviour the demo exists to show.
AGENT_TASK = (
    "Find listings whose descriptions mention a quiet garden studio, and tell me which "
    "neighbourhoods they are in. Then, across all listings in those neighbourhoods, "
    "report the number of listings and the average review score for each one."
)


def step(message: str) -> None:
    print(f"\n=== {message} ===")


def download_parquet() -> Path:
    path = Path(tempfile.gettempdir()) / "sf-airbnb-listings.parquet"
    if path.exists() and path.stat().st_size > 0:
        print(f"Using cached fixture at {path} ({path.stat().st_size / 1_000_000:.1f} MB)")
        return path
    print(f"Downloading {PARQUET_URL}")
    # The default urllib user-agent is rejected with a 403 by the asset host.
    request = urllib.request.Request(PARQUET_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        path.write_bytes(response.read())
    print(f"Saved {path} ({path.stat().st_size / 1_000_000:.1f} MB)")
    return path


def find_database(client: hl.HotdataClient, name: str) -> Any | None:
    for db in client.list_managed_databases():
        if db.description == name or db.id == name:
            return db
    return None


def ensure_database(client: hl.HotdataClient) -> Any:
    existing = find_database(client, DATABASE)
    if existing is not None:
        print(f"Reusing managed database {DATABASE!r} (id={existing.id})")
        return existing
    db = client.create_managed_database(description=DATABASE, schema=SCHEMA, tables=[TABLE])
    print(f"Created managed database {DATABASE!r} (id={db.id}) with {SCHEMA}.{TABLE} declared")
    return db


def load_listings(client: hl.HotdataClient, parquet: Path) -> None:
    loaded = client.load_managed_table(DATABASE, TABLE, schema=SCHEMA, file=str(parquet))
    print(f"Loaded {loaded.row_count} rows into {loaded.full_name}")


def table_columns(client: hl.HotdataClient) -> list[str]:
    result = client.execute_sql(f"SELECT * FROM {TABLE_REF} LIMIT 1", database=DATABASE)
    return list(result.columns)


def ensure_bm25_index(client: hl.HotdataClient, connection_id: str) -> None:
    indexes_api = hotdata.IndexesApi(client.api)
    existing = indexes_api.list_indexes(connection_id, SCHEMA, TABLE).indexes
    match = next((idx for idx in existing if idx.index_name == INDEX_NAME), None)

    if match is None:
        print(f"Creating BM25 index {INDEX_NAME!r} on {SCHEMA}.{TABLE}.{SEARCH_COLUMN}")
        indexes_api.create_index(
            connection_id,
            SCHEMA,
            TABLE,
            hotdata.CreateIndexRequest(
                index_name=INDEX_NAME,
                columns=[SEARCH_COLUMN],
                index_type="bm25",
                var_async=True,
            ),
        )
    else:
        print(f"Index {INDEX_NAME!r} already exists (status={match.status})")

    deadline = time.monotonic() + INDEX_TIMEOUT_SECONDS
    while True:
        indexes = indexes_api.list_indexes(connection_id, SCHEMA, TABLE).indexes
        current = next((idx for idx in indexes if idx.index_name == INDEX_NAME), None)
        if current is not None and current.status == hotdata.IndexStatus.READY:
            print(f"Index ready (type={current.index_type}, columns={current.columns})")
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"index {INDEX_NAME!r} not ready after {INDEX_TIMEOUT_SECONDS}s "
                f"(status={getattr(current, 'status', 'missing')})"
            )
        status = getattr(current, "status", "missing")
        print(f"  waiting for index build (status={status})…")
        time.sleep(5)


def print_hits(payload: str, *, query: str) -> None:
    parsed = json.loads(payload)
    rows = parsed["rows"]
    print(f"Top {len(rows)} hits for {query!r} (took {parsed['metadata']['execution_time_ms']}ms)")
    for rank, row in enumerate(rows, start=1):
        description = str(row.get(SEARCH_COLUMN, ""))
        snippet = description[:110].replace("\n", " ")
        label = row.get("name") or row.get("id") or "—"
        print(f"  {rank}. score={row['score']:.3f}  {label}")
        print(f"     {snippet}…")


def engine_error_message(exc: BaseException) -> str:
    """Pull the engine's own message out of a failed call.

    The framework raises ``RuntimeError(e.reason)`` — "Bad Request" — while the useful
    text ("Invalid function 'to_tsvector'. Did you mean 'to_char'?") stays in the
    ApiException body further down the ``__cause__`` chain.
    """
    node: BaseException | None = exc
    while node is not None:
        body = getattr(node, "body", None)
        if body:
            try:
                return str(json.loads(body)["error"]["message"])
            except (ValueError, KeyError, TypeError):
                return str(body)[:500]
        node = node.__cause__
    return str(exc)


def with_error_feedback(tools: list[Any]) -> list[Any]:
    """Return tools that hand failures back to the model instead of raising.

    An agent that cannot see why a call failed cannot correct it, and an exception
    out of a tool aborts the whole graph. Arguably belongs in the package itself.
    """

    def wrap(tool: Any) -> Any:
        inner = tool.func

        def safe(*args: Any, **kwargs: Any) -> str:
            try:
                return str(inner(*args, **kwargs))
            except Exception as e:
                message = engine_error_message(e)
                print(f"  [tool error fed back to the model] {message[:120]}")
                return json.dumps({"error": message})

        return tool.model_copy(update={"func": safe})

    return [wrap(t) for t in tools]


def run_agent(tools: list[Any], *, model: str) -> None:
    from langchain.agents import create_agent

    tracing = os.environ.get("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}
    has_langsmith_key = bool(os.environ.get("LANGSMITH_API_KEY"))
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    print(f"LangSmith tracing={tracing} (api key present={has_langsmith_key}, project={project!r})")

    agent = create_agent(
        model=model,
        tools=with_error_feedback(tools),
        # Role only. Which tool to reach for, and the engine's constraints, come from
        # the tool descriptions themselves — an app should not have to teach the model
        # how the query engine behaves.
        system_prompt=(
            "You are a data analyst working with a dataset of San Francisco Airbnb "
            "listings. Cite the concrete numbers you retrieve."
        ),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": AGENT_TASK}]})

    print("\n--- tool calls the agent made ---")
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            print(f"  {call['name']}({json.dumps(call['args'])[:160]})")

    print("\n--- final answer ---")
    print(result["messages"][-1].content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY, help="search text to run")
    parser.add_argument("--k", type=int, default=5, help="how many ranked hits to return")
    parser.add_argument(
        "--model",
        default=os.environ.get("DEMO_MODEL"),
        help="tool-calling model for the agent step, e.g. '<provider>:<model>' "
        "(or set DEMO_MODEL); the agent step is skipped without one",
    )
    parser.add_argument("--skip-agent", action="store_true", help="stop after direct tool use")
    parser.add_argument("--reload", action="store_true", help="reload the parquet even if loaded")
    parser.add_argument(
        "--cleanup", action="store_true", help="delete the demo managed database and exit"
    )
    args = parser.parse_args()

    client = hl.from_env()
    print(f"Connected to {client.host} (workspace={client.workspace_id})")

    if args.cleanup:
        existing = find_database(client, DATABASE)
        if existing is None:
            print(f"No managed database named {DATABASE!r} to delete")
        else:
            client.delete_managed_database(existing.id)
            print(f"Deleted managed database {DATABASE!r} (id={existing.id})")
        client.close()
        return

    try:
        step("1. Managed database")
        db = ensure_database(client)

        step("2. Listings data")
        already_loaded = False
        if not args.reload:
            # Probing with a row read rather than COUNT(*): the engine rejects a
            # projection that is aggregates only.
            try:
                probe = client.execute_sql(f"SELECT id FROM {TABLE_REF} LIMIT 1", database=DATABASE)
                already_loaded = bool(probe.rows)
                if already_loaded:
                    print(f"{TABLE_REF} already holds data; skipping load (--reload to force)")
            except Exception as e:
                print(f"Table not queryable yet ({type(e).__name__}); loading fixture")
        if not already_loaded:
            load_listings(client, download_parquet())

        step("3. BM25 index")
        ensure_bm25_index(client, db.default_connection_id)

        step("4. Tools")
        available = table_columns(client)
        columns = [c for c in PREFERRED_COLUMNS if c in available]
        if SEARCH_COLUMN not in columns:
            columns.append(SEARCH_COLUMN)
        print(f"Table has {len(available)} columns; projecting {columns}")

        tools = hl.make_hotdata_tools(
            client,
            database=DATABASE,
            max_rows=args.k,
            search_table=TABLE_REF,
            search_column=SEARCH_COLUMN,
            search_columns=columns,
            search_k=args.k,
        )
        by_name = {tool.name: tool for tool in tools}
        print(f"Tools exposed to the agent: {sorted(by_name)}")

        step("5. Direct tool invocation")
        search = by_name["hotdata_search_text"]
        sql = hl.bm25_search_sql(
            table=TABLE_REF,
            column=SEARCH_COLUMN,
            query=args.query,
            k=args.k,
            columns=columns,
        )
        print(f"SQL the tool will run:\n  {sql}")
        print_hits(search.invoke({"query": args.query}), query=args.query)

        done = "Everything above is the tool working end to end against the real engine."
        if args.skip_agent:
            print("\n--skip-agent set; stopping before the agent run")
            return
        if not args.model:
            print("\nNo model given — skipping the agent run.")
            print("Pass --model '<provider>:<model>' (or set DEMO_MODEL) to run it.")
            print(done)
            return
        provider_key_var = PROVIDER_KEY_VARS.get(args.model.split(":", 1)[0])
        if provider_key_var and not os.environ.get(provider_key_var):
            print(f"\n{provider_key_var} is not set — skipping the agent run.")
            print(done)
            return

        step("6. LangChain agent choosing between search and SQL")
        run_agent(tools, model=args.model)
    finally:
        client.close()


if __name__ == "__main__":
    main()
