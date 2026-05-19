"""Managed database tools for LangChain agents."""

import hotdata_langchain as hl


def main() -> None:
    client = hl.from_env()
    tools = hl.make_hotdata_tools(client)
    by_name = {tool.name: tool for tool in tools}

    create = by_name["hotdata_create_managed_database"]
    print(
        create.invoke(
            {
                "name": "demo_sales",
                "schema_name": "public",
                "tables": "orders\ncustomers",
            }
        )
    )

    load = by_name["hotdata_load_managed_table"]
    print(
        load.invoke(
            {
                "database": "demo_sales",
                "table": "orders",
                "file": "/path/to/orders.parquet",
                "schema_name": "public",
            }
        )
    )

    client.close()


if __name__ == "__main__":
    main()
