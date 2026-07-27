from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from hotdata_framework import ManagedDatabase, QueryResult

from hotdata_langchain.schema import (
    DEFAULT_MAX_COLUMNS,
    default_describe_description,
    describe_tables_json,
    make_hotdata_describe_tables_tool,
    table_columns_sql,
    table_overview_sql,
)
from hotdata_langchain.tools import make_hotdata_tools


def result(columns: list[str], rows: list[list[object]]) -> QueryResult:
    return QueryResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        result_id="res",
        query_run_id="run",
        execution_time_ms=3,
        warning=None,
        error_message=None,
    )


OVERVIEW = result(
    ["table_schema", "table_name", "column_count"],
    [["public", "listings", 85], ["public", "reviews", 6]],
)
COLUMNS = result(
    ["table_schema", "table_name", "column_name", "data_type"],
    [
        ["public", "listings", "id", "Int64"],
        ["public", "listings", "description", "LargeUtf8"],
    ],
)


def executed_sql(client: MagicMock) -> str:
    sql = client.execute_sql.call_args.args[0]
    assert isinstance(sql, str)
    return sql


# --- SQL construction -------------------------------------------------------------


def test_overview_sql_groups_by_table() -> None:
    sql = table_overview_sql()
    assert "information_schema.columns" in sql
    # COUNT(column_name), not COUNT(*): the engine rejects an aggregate that names no column.
    assert "COUNT(column_name)" in sql
    assert "GROUP BY table_schema, table_name" in sql


def test_columns_sql_filters_by_bare_table_name() -> None:
    sql = table_columns_sql("listings")
    assert "WHERE table_name = 'listings'" in sql
    assert "table_schema =" not in sql
    assert "ORDER BY table_schema, table_name, ordinal_position" in sql


def test_columns_sql_filters_by_schema_qualified_name() -> None:
    sql = table_columns_sql("public.listings")
    assert "WHERE table_name = 'listings' AND table_schema = 'public'" in sql


def test_columns_sql_applies_the_column_cap() -> None:
    assert table_columns_sql("listings").endswith(f"LIMIT {DEFAULT_MAX_COLUMNS}")
    assert table_columns_sql("listings", limit=10).endswith("LIMIT 10")


@pytest.mark.parametrize("table", ["a.b.c", "list ings", "list'ings", "1listings", "", "public."])
def test_columns_sql_rejects_bad_table_reference(table: str) -> None:
    with pytest.raises(ValueError):
        table_columns_sql(table)


# --- JSON payloads ----------------------------------------------------------------


def test_describe_without_table_lists_tables_and_counts() -> None:
    client = MagicMock()
    client.execute_sql.return_value = OVERVIEW
    payload = json.loads(describe_tables_json(client))
    assert payload == {
        "tables": [
            {"table": "public.listings", "column_count": 85},
            {"table": "public.reviews", "column_count": 6},
        ]
    }


def test_describe_with_table_lists_columns_and_types() -> None:
    client = MagicMock()
    client.execute_sql.return_value = COLUMNS
    payload = json.loads(describe_tables_json(client, table="listings"))
    assert payload["table"] == "public.listings"
    assert payload["columns"] == [
        {"name": "id", "type": "Int64"},
        {"name": "description", "type": "LargeUtf8"},
    ]
    assert "truncated_at" not in payload


def test_describe_reports_truncation_only_when_rows_exceed_the_cap() -> None:
    """A table with exactly max_columns columns is complete, not truncated."""
    client = MagicMock()
    client.execute_sql.return_value = COLUMNS  # two column rows
    payload = json.loads(describe_tables_json(client, table="listings", max_columns=2))
    assert len(payload["columns"]) == 2
    assert "truncated_at" not in payload


def test_describe_reports_truncation_and_trims_to_the_cap() -> None:
    client = MagicMock()
    client.execute_sql.return_value = COLUMNS  # two column rows, cap of one
    payload = json.loads(describe_tables_json(client, table="listings", max_columns=1))
    assert payload["truncated_at"] == 1
    assert [c["name"] for c in payload["columns"]] == ["id"]


def test_describe_rejects_a_non_positive_max_columns() -> None:
    """Reading one row past the cap makes a zero cap slice to empty and then index it."""
    client = MagicMock()
    client.execute_sql.return_value = COLUMNS
    with pytest.raises(ValueError, match="max_columns must be >= 1"):
        describe_tables_json(client, table="listings", max_columns=0)


def test_describe_tool_rejects_a_non_positive_max_columns() -> None:
    with pytest.raises(ValueError, match="max_columns must be >= 1"):
        make_hotdata_describe_tables_tool(MagicMock(), max_columns=0)


def test_describe_queries_one_row_past_the_cap() -> None:
    """Distinguishing exact-fit from truncated needs the extra row."""
    client = MagicMock()
    client.execute_sql.return_value = COLUMNS
    describe_tables_json(client, table="listings", max_columns=25)
    assert executed_sql(client).endswith("LIMIT 26")


def test_describe_reports_an_unknown_table_rather_than_empty_success() -> None:
    client = MagicMock()
    client.execute_sql.return_value = result(
        ["table_schema", "table_name", "column_name", "data_type"], []
    )
    payload = json.loads(describe_tables_json(client, table="nope"))
    assert payload["columns"] == []
    assert "no table named" in payload["error"]


def test_describe_scopes_queries_to_the_database(managed_db: ManagedDatabase) -> None:
    client = MagicMock()
    client.execute_sql.return_value = OVERVIEW
    describe_tables_json(client, database=managed_db)
    assert client.execute_sql.call_args.kwargs == {"database": managed_db}


def test_describe_refuses_an_unresolved_database_scope() -> None:
    """A bare string would reach the framework's by-name fallback."""
    with pytest.raises(TypeError, match="resolve_database_by_id"):
        describe_tables_json(MagicMock(), database="sf_airbnb")  # type: ignore[arg-type]


# --- Tool surface -----------------------------------------------------------------


def test_describe_tool_shape() -> None:
    tool = make_hotdata_describe_tables_tool(MagicMock())
    assert tool.name == "hotdata_describe_tables"
    assert set(tool.args) == {"table"}
    assert tool.description == default_describe_description()


def test_describe_tool_defaults_to_the_overview() -> None:
    client = MagicMock()
    client.execute_sql.return_value = OVERVIEW
    tool = make_hotdata_describe_tables_tool(client)
    payload = json.loads(tool.invoke({}))
    assert [t["table"] for t in payload["tables"]] == ["public.listings", "public.reviews"]
    assert "GROUP BY" in executed_sql(client)


def test_describe_tool_drills_into_one_table() -> None:
    client = MagicMock()
    client.execute_sql.return_value = COLUMNS
    tool = make_hotdata_describe_tables_tool(client)
    tool.invoke({"table": "public.listings"})
    assert "WHERE table_name = 'listings'" in executed_sql(client)


def test_describe_tool_is_registered_by_default() -> None:
    names = {t.name for t in make_hotdata_tools(MagicMock())}
    assert "hotdata_describe_tables" in names


def test_describe_tool_can_be_turned_off() -> None:
    names = {t.name for t in make_hotdata_tools(MagicMock(), describe_tables=False)}
    assert "hotdata_describe_tables" not in names


def test_sql_description_points_at_the_schema_tool_when_registered() -> None:
    tools = {t.name: t for t in make_hotdata_tools(MagicMock())}
    assert "hotdata_describe_tables" in (tools["hotdata_execute_sql"].description or "")


def test_sql_description_falls_back_to_information_schema_without_the_tool() -> None:
    tools = {t.name: t for t in make_hotdata_tools(MagicMock(), describe_tables=False)}
    description = tools["hotdata_execute_sql"].description or ""
    assert "information_schema" in description
    assert "hotdata_describe_tables" not in description
