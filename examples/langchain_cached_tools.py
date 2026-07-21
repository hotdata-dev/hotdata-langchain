"""Cache LangChain tool results in a Hotdata managed table."""

import time

import hotdata_langchain as hl
from hotdata_langchain.cache import cached


def main() -> None:
    client = hl.from_env()

    # HotdataToolCache is a pluggable cache backend for the two read/idempotent tools —
    # repeated calls with the same SQL are served from a managed table instead of
    # re-running the query. Construct one instance per process and reuse it.
    cache = hl.HotdataToolCache(client)
    tools = hl.make_hotdata_tools(client, cache=cache)
    by_name = {tool.name: tool for tool in tools}

    sql_tool = by_name["hotdata_execute_sql"]
    print("First call (miss, runs the query):")
    print(sql_tool.invoke({"sql": "SELECT 1 AS ok"}))

    print("\nSecond call, same SQL (hit, served from the cache):")
    print(sql_tool.invoke({"sql": "SELECT 1 AS ok"}))

    # cached() also works on any plain function — not just this package's own tools —
    # which is the more general story: Hotdata as a cache backend for arbitrary
    # LangChain tools (database queries, API calls, search results).
    calls = {"n": 0}

    def slow_search(query: str) -> str:
        calls["n"] += 1
        time.sleep(1)  # stand in for a real API call or search request
        return f"{calls['n']} result(s) for {query!r}"

    cached_search = cached(slow_search, cache=cache, tool_name="slow_search")

    print("\nArbitrary function, first call (miss, ~1s):")
    start = time.monotonic()
    print(cached_search("hotdata langchain"))
    print(f"took {time.monotonic() - start:.2f}s")

    print("\nSame function, same query (hit, instant):")
    start = time.monotonic()
    print(cached_search("hotdata langchain"))
    print(f"took {time.monotonic() - start:.2f}s")

    client.close()


if __name__ == "__main__":
    main()
