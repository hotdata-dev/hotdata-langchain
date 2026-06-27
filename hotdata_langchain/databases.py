"""Managed database helpers for LangChain agents."""

from __future__ import annotations

import json
from typing import Any

from hotdata_framework import (
    DEFAULT_SCHEMA,
    HotdataClient,
    LoadManagedTableResult,
    ManagedDatabase,
)


def list_managed_databases_json(client: HotdataClient) -> str:
    rows = [
        {
            "description": db.description,
            "id": db.id,
            "sql_prefix": f"{db.id}.{{schema}}.{{table}}",
        }
        for db in client.list_managed_databases()
    ]
    return json.dumps(rows, indent=2)


def create_managed_database(
    client: HotdataClient,
    *,
    name: str,
    schema: str = DEFAULT_SCHEMA,
    tables: list[str] | None = None,
) -> ManagedDatabase:
    return client.create_managed_database(description=name, schema=schema, tables=tables)


def load_managed_table(
    client: HotdataClient,
    *,
    database: str,
    table: str,
    file: str,
    schema: str = DEFAULT_SCHEMA,
) -> LoadManagedTableResult:
    return client.load_managed_table(database, table, schema=schema, file=file)


def managed_database_summary(db: ManagedDatabase) -> dict[str, str]:
    return {"id": db.id, "description": db.description or db.id}


def load_result_summary(result: LoadManagedTableResult) -> dict[str, Any]:
    return {
        "connection_id": result.connection_id,
        "schema_name": result.schema_name,
        "table_name": result.table_name,
        "row_count": result.row_count,
        "full_name": result.full_name,
    }
