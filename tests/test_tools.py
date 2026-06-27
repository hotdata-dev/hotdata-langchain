from __future__ import annotations

import json

from hotdata_framework import LoadManagedTableResult, ManagedDatabase

from hotdata_langchain.databases import (
    create_managed_database,
    list_managed_databases_json,
    load_managed_table,
)
from hotdata_langchain.tools import execute_sql_json, make_hotdata_tools, result_rows_for_llm


def test_result_rows_for_llm(sample_result):
    rows = result_rows_for_llm(sample_result, max_rows=1)
    assert rows == [{"n": 1}]


def test_execute_sql_json(mock_client, sample_result):
    payload = json.loads(execute_sql_json(mock_client, "select 1"))
    assert payload["metadata"]["row_count"] == 2
    assert payload["rows"] == [{"n": 1}, {"n": 2}]
    mock_client.execute_sql.assert_called_once_with("select 1", database=None)


def test_execute_sql_json_with_database(mock_client, sample_result):
    execute_sql_json(mock_client, "select 1", database="my_db")
    mock_client.execute_sql.assert_called_once_with("select 1", database="my_db")


def test_list_managed_databases_json(mock_client):
    mock_client.list_managed_databases.return_value = [
        ManagedDatabase(id="c1", description="sales", default_connection_id="conn_c1"),
    ]
    payload = json.loads(list_managed_databases_json(mock_client))
    assert payload[0]["description"] == "sales"


def test_create_managed_database_delegates(mock_client):
    mock_client.create_managed_database.return_value = ManagedDatabase(
        id="c1",
        description="sales",
        default_connection_id="conn_c1",
    )
    db = create_managed_database(mock_client, name="sales", tables=["orders"])
    mock_client.create_managed_database.assert_called_once_with(
        description="sales",
        schema="public",
        tables=["orders"],
    )
    assert db.description == "sales"


def test_load_managed_table_delegates(mock_client):
    mock_client.load_managed_table.return_value = LoadManagedTableResult(
        connection_id="c1",
        schema_name="public",
        table_name="orders",
        row_count=3,
        full_name="sales.public.orders",
    )
    loaded = load_managed_table(
        mock_client,
        database="sales",
        table="orders",
        file="/tmp/orders.parquet",
    )
    mock_client.load_managed_table.assert_called_once_with(
        "sales",
        "orders",
        schema="public",
        file="/tmp/orders.parquet",
    )
    assert loaded.row_count == 3


def test_make_hotdata_tools(mock_client, sample_result):
    mock_client.create_managed_database.return_value = ManagedDatabase(
        id="c1",
        description="sales",
        default_connection_id="conn_c1",
    )
    mock_client.load_managed_table.return_value = LoadManagedTableResult(
        connection_id="c1",
        schema_name="public",
        table_name="orders",
        row_count=1,
        full_name="sales.public.orders",
    )
    tools = make_hotdata_tools(mock_client)
    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == {
        "hotdata_execute_sql",
        "hotdata_list_managed_databases",
        "hotdata_create_managed_database",
        "hotdata_load_managed_table",
    }

    json.loads(by_name["hotdata_execute_sql"].invoke({"sql": "select 1"}))
    json.loads(by_name["hotdata_list_managed_databases"].invoke({}))
    json.loads(
        by_name["hotdata_create_managed_database"].invoke({"name": "sales", "tables": "orders"})
    )
    json.loads(
        by_name["hotdata_load_managed_table"].invoke(
            {
                "database": "sales",
                "table": "orders",
                "file": "/tmp/orders.parquet",
            }
        )
    )
