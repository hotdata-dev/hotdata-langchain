"""Minimal LangChain tool usage with hotdata-langchain."""

import hotdata_langchain as hl


def main() -> None:
    client = hl.from_env()

    tools = {tool.name: tool for tool in hl.make_hotdata_tools(client)}
    print(tools["hotdata_list_managed_databases"].invoke({}))

    # Queries need a database scope, addressed by id — a database name is a display
    # label and is not unique. Pass the listed record straight through to reuse the
    # lookup that produced it.
    databases = client.list_managed_databases()
    if not databases:
        print("No instant databases in this workspace; create one to run a query.")
        client.close()
        return

    scoped = hl.make_hotdata_tools(client, database_id=databases[0])
    by_name = {tool.name: tool for tool in scoped}
    print(by_name["hotdata_execute_sql"].invoke({"sql": "SELECT 1 AS ok"}))

    client.close()


if __name__ == "__main__":
    main()
