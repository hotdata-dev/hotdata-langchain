from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hotdata_framework import LoadManagedTableResult, ManagedDatabase

from hotdata_langchain.databases import (
    create_managed_database,
    fetch_parquet,
    list_managed_databases_json,
    load_managed_table,
)
from hotdata_langchain.schema import DEFAULT_DESCRIBE_TOOL_NAME
from hotdata_langchain.search import DEFAULT_SEARCH_TOOL_NAME
from hotdata_langchain.tools import (
    DEFAULT_CREATE_DATABASE_TOOL_NAME,
    DEFAULT_LIST_DATABASES_TOOL_NAME,
    DEFAULT_LOAD_TABLE_TOOL_NAME,
    DEFAULT_SQL_TOOL_NAME,
    execute_sql_json,
    make_hotdata_tools,
    result_rows_for_llm,
)
from tests.conftest import fake_response


def test_result_rows_for_llm(sample_result):
    rows = result_rows_for_llm(sample_result, max_rows=1)
    assert rows == [{"n": 1}]


def test_execute_sql_json(mock_client, sample_result):
    payload = json.loads(execute_sql_json(mock_client, "select 1"))
    assert payload["metadata"]["row_count"] == 2
    assert payload["rows"] == [{"n": 1}, {"n": 2}]
    mock_client.execute_sql.assert_called_once_with("select 1", database=None)


def test_execute_sql_json_with_database(mock_client, sample_result, managed_db):
    execute_sql_json(mock_client, "select 1", database=managed_db)
    mock_client.execute_sql.assert_called_once_with("select 1", database=managed_db)


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


def test_load_managed_table_delegates(mock_client, managed_db, parquet_file):
    """The load addresses the resolved record, so no name can select the target."""
    mock_client.load_managed_table.return_value = LoadManagedTableResult(
        connection_id="c1",
        schema_name="public",
        table_name="orders",
        row_count=3,
        full_name="sales.public.orders",
    )
    loaded = load_managed_table(
        mock_client,
        database_id=managed_db,
        table="orders",
        file=str(parquet_file),
    )
    mock_client.load_managed_table.assert_called_once_with(
        managed_db,
        "orders",
        schema="public",
        file=str(parquet_file),
    )
    assert loaded.row_count == 3


def test_load_managed_table_rejects_a_path_that_is_not_there(mock_client, managed_db):
    """A raw FileNotFoundError says nothing about what the tool would have accepted."""
    with pytest.raises(FileNotFoundError, match="https:// URL"):
        load_managed_table(
            mock_client,
            database_id=managed_db,
            table="orders",
            file="/tmp/not-here.parquet",
        )
    mock_client.load_managed_table.assert_not_called()


def test_load_managed_table_accepts_a_url(mock_client, managed_db, parquet_file, monkeypatch):
    """The only ingest route open to a deployed agent, which has no filesystem of its own."""
    mock_client.load_managed_table.return_value = LoadManagedTableResult(
        connection_id="c1",
        schema_name="public",
        table_name="orders",
        row_count=3,
        full_name="sales.public.orders",
    )
    monkeypatch.setattr(
        "hotdata_langchain.databases.urlopen",
        lambda request, timeout: fake_response(parquet_file.read_bytes()),
    )
    loaded = load_managed_table(
        mock_client,
        database_id=managed_db,
        table="orders",
        file="https://example.test/orders.parquet",
    )
    uploaded = mock_client.load_managed_table.call_args.kwargs["file"]
    assert uploaded.endswith(".parquet")
    assert loaded.row_count == 3


def test_the_downloaded_copy_is_removed_after_the_load(
    mock_client, managed_db, parquet_file, monkeypatch
):
    """A tool an agent calls in a loop must not leave a file behind on every call."""
    monkeypatch.setattr(
        "hotdata_langchain.databases.urlopen",
        lambda request, timeout: fake_response(parquet_file.read_bytes()),
    )
    load_managed_table(
        mock_client,
        database_id=managed_db,
        table="orders",
        file="https://example.test/orders.parquet",
    )
    assert not Path(mock_client.load_managed_table.call_args.kwargs["file"]).exists()


def test_the_downloaded_copy_is_removed_when_the_load_fails(
    mock_client, managed_db, parquet_file, monkeypatch
):
    mock_client.load_managed_table.side_effect = RuntimeError("Bad Request")
    monkeypatch.setattr(
        "hotdata_langchain.databases.urlopen",
        lambda request, timeout: fake_response(parquet_file.read_bytes()),
    )
    with pytest.raises(RuntimeError):
        load_managed_table(
            mock_client,
            database_id=managed_db,
            table="orders",
            file="https://example.test/orders.parquet",
        )
    assert not Path(mock_client.load_managed_table.call_args.kwargs["file"]).exists()


def test_a_url_that_returns_a_login_page_is_rejected_before_upload(monkeypatch):
    """It answers 200 with HTML, so nothing before the magic-byte check notices."""
    monkeypatch.setattr(
        "hotdata_langchain.databases.urlopen",
        lambda request, timeout: fake_response(b"<html>sign in</html>"),
    )
    with pytest.raises(ValueError, match="did not return a parquet file"):
        fetch_parquet("https://example.test/orders.parquet")


def test_a_failed_fetch_leaves_no_temporary_file(monkeypatch):
    created: list[str] = []
    real_named_temp = tempfile.NamedTemporaryFile

    def record(*args, **kwargs):
        handle = real_named_temp(*args, **kwargs)
        created.append(handle.name)
        return handle

    monkeypatch.setattr("hotdata_langchain.databases.tempfile.NamedTemporaryFile", record)
    monkeypatch.setattr(
        "hotdata_langchain.databases.urlopen",
        lambda request, timeout: fake_response(b"<html>sign in</html>"),
    )
    with pytest.raises(ValueError, match="did not return a parquet file"):
        fetch_parquet("https://example.test/orders.parquet")
    assert created and not any(Path(name).exists() for name in created)


def test_fetch_parquet_refuses_a_non_http_url():
    """urlopen would honour file://, which would turn this into a local file reader."""
    with pytest.raises(ValueError, match="http:// or https:// URL"):
        fetch_parquet("file:///etc/passwd")


def test_the_fetch_sets_a_user_agent(monkeypatch, parquet_file):
    """Asset hosts 403 urllib's default, which the demo hit and worked around by hand."""
    seen: dict[str, str] = {}

    def capture(request, timeout):
        seen.update(request.headers)
        return fake_response(parquet_file.read_bytes())

    monkeypatch.setattr("hotdata_langchain.databases.urlopen", capture)
    path = fetch_parquet("https://example.test/orders.parquet")
    Path(path).unlink()
    assert seen.get("User-agent") == "hotdata-langchain"


def test_make_hotdata_tools(mock_client, sample_result, managed_db, databases_api, parquet_file):
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
        "hotdata_describe_tables",
    }

    json.loads(by_name["hotdata_execute_sql"].invoke({"sql": "select 1"}))
    json.loads(by_name["hotdata_list_managed_databases"].invoke({}))
    json.loads(
        by_name["hotdata_create_managed_database"].invoke({"name": "sales", "tables": "orders"})
    )
    json.loads(
        by_name["hotdata_load_managed_table"].invoke(
            {
                "database_id": managed_db.id,
                "table": "orders",
                "file": str(parquet_file),
            }
        )
    )


def test_every_tool_name_is_an_exported_constant(mock_client):
    """Consumers filter tools by name, so a name that only exists as a literal is API.

    Two agents against one workspace already need two different subsets, and each
    hardcoded string is a rename away from silently selecting nothing.
    """
    tools = make_hotdata_tools(
        mock_client,
        search_table="default.public.listings",
        search_column="description",
    )
    constants = {
        DEFAULT_SQL_TOOL_NAME,
        DEFAULT_LIST_DATABASES_TOOL_NAME,
        DEFAULT_CREATE_DATABASE_TOOL_NAME,
        DEFAULT_LOAD_TABLE_TOOL_NAME,
        DEFAULT_DESCRIBE_TOOL_NAME,
        DEFAULT_SEARCH_TOOL_NAME,
    }
    assert {tool.name for tool in tools} == constants


def test_management_tools_can_be_left_out(mock_client):
    """An agent reading one fixed database cannot use them, and can only misuse them."""
    tools = make_hotdata_tools(mock_client, management_tools=False)
    assert {tool.name for tool in tools} == {
        DEFAULT_SQL_TOOL_NAME,
        DEFAULT_DESCRIBE_TOOL_NAME,
    }


def test_management_tools_are_on_by_default(mock_client):
    tools = {tool.name for tool in make_hotdata_tools(mock_client)}
    assert DEFAULT_CREATE_DATABASE_TOOL_NAME in tools
    assert DEFAULT_LOAD_TABLE_TOOL_NAME in tools


def test_the_sql_tool_stays_first_whichever_tools_are_included(mock_client):
    """It is the one every agent needs; the ordering the model sees should not shift."""
    for kwargs in ({}, {"management_tools": False}, {"describe_tables": False}):
        tools = make_hotdata_tools(mock_client, **kwargs)
        assert tools[0].name == DEFAULT_SQL_TOOL_NAME
