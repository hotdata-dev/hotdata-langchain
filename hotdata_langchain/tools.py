"""LangChain tools built on hotdata-runtime."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool

from hotdata_runtime import DEFAULT_SCHEMA, HotdataClient, QueryResult

from hotdata_langchain.databases import (
    create_managed_database,
    list_managed_databases_json,
    load_managed_table,
    load_result_summary,
    managed_database_summary,
)


def result_rows_for_llm(result: QueryResult, *, max_rows: int = 20) -> list[dict[str, Any]]:
    return result.to_records(max_rows=max_rows)


def execute_sql_json(
    client: HotdataClient,
    sql: str,
    *,
    max_rows: int = 100,
    database: str | None = None,
) -> str:
    result = client.execute_sql(sql, database=database)
    payload = {
        "metadata": result.metadata_dict(),
        "rows": result.to_records(max_rows=max_rows),
    }
    return json.dumps(payload, indent=2)


def make_hotdata_tools(
    client: HotdataClient,
    *,
    max_rows: int = 100,
    database: str | None = None,
) -> list[StructuredTool]:
    """Return LangChain tools for SQL and managed database workflows."""

    def hotdata_execute_sql(sql: str) -> str:
        """Run SQL against the Hotdata workspace and return JSON rows."""
        return execute_sql_json(client, sql, max_rows=max_rows, database=database)

    def hotdata_list_managed_databases() -> str:
        """List Hotdata-managed databases in the workspace."""
        return list_managed_databases_json(client)

    def hotdata_create_managed_database(
        name: str,
        schema_name: str = DEFAULT_SCHEMA,
        tables: str = "",
    ) -> str:
        """Create a Hotdata-managed database and optionally declare tables (one per line)."""
        table_names = [line.strip() for line in tables.splitlines() if line.strip()]
        db = create_managed_database(
            client,
            name=name,
            schema=schema_name or DEFAULT_SCHEMA,
            tables=table_names or None,
        )
        return json.dumps(managed_database_summary(db), indent=2)

    def hotdata_load_managed_table(
        database: str,
        table: str,
        file: str,
        schema_name: str = DEFAULT_SCHEMA,
    ) -> str:
        """Load a local parquet file into a declared managed table."""
        loaded = load_managed_table(
            client,
            database=database,
            table=table,
            file=file,
            schema=schema_name or DEFAULT_SCHEMA,
        )
        return json.dumps(load_result_summary(loaded), indent=2)

    return [
        StructuredTool.from_function(
            func=hotdata_execute_sql,
            name="hotdata_execute_sql",
        ),
        StructuredTool.from_function(
            func=hotdata_list_managed_databases,
            name="hotdata_list_managed_databases",
        ),
        StructuredTool.from_function(
            func=hotdata_create_managed_database,
            name="hotdata_create_managed_database",
        ),
        StructuredTool.from_function(
            func=hotdata_load_managed_table,
            name="hotdata_load_managed_table",
        ),
    ]
