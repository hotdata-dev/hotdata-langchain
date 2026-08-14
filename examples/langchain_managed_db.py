"""Managed database tools for LangChain agents."""

import json

import hotdata_langchain as hl


def main() -> None:
    client = hl.from_env()
    tools = hl.make_hotdata_tools(client)
    by_name = {tool.name: tool for tool in tools}

    create = by_name[hl.DEFAULT_CREATE_DATABASE_TOOL_NAME]
    created = create.invoke(
        {
            "name": "demo_sales",
            "schema_name": "public",
            "tables": "orders\ncustomers",
        }
    )
    print(created)

    # 'name' was a display label; the id is what addresses the database from here on.
    database_id = json.loads(created)["id"]

    load = by_name[hl.DEFAULT_LOAD_TABLE_TOOL_NAME]
    print(
        load.invoke(
            {
                "database_id": database_id,
                "table": "orders",
                # A URL works too, and is the only form reachable from a deployed agent.
                "file": "/path/to/orders.parquet",
                "schema_name": "public",
            }
        )
    )

    client.close()


if __name__ == "__main__":
    main()
