"""Minimal LangChain tool usage with hotdata-langchain."""

import hotdata_langchain as hl


def main() -> None:
    client = hl.from_env()
    tools = hl.make_hotdata_tools(client)
    by_name = {tool.name: tool for tool in tools}

    sql_tool = by_name["hotdata_execute_sql"]
    print(sql_tool.invoke({"sql": "SELECT 1 AS ok"}))

    list_tool = by_name["hotdata_list_managed_databases"]
    print(list_tool.invoke({}))

    client.close()


if __name__ == "__main__":
    main()
