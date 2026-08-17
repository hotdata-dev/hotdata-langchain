from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from hotdata_framework import ManagedDatabase, QueryResult

from hotdata_langchain.schema import (
    DEFAULT_MAX_COLUMNS,
    column_stats_sql,
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


def executed_sqls(client: MagicMock) -> list[str]:
    """Every SQL statement the client was asked to run, in order.

    Describing one table runs the schema lookup and then the column-stats aggregate, so
    a test about the first one cannot read the last call.
    """
    return [call.args[0] for call in client.execute_sql.call_args_list]


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
    assert executed_sqls(client)[0].endswith("LIMIT 26")


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
    assert "WHERE table_name = 'listings'" in executed_sqls(client)[0]


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


# --- Populated-ness -----------------------------------------------------------------


STATS = result(["row_count", "n0", "n1"], [[7535, 7535, 0]])


def describing_listings(client: MagicMock, **kwargs: object) -> dict[str, object]:
    """Describe 'listings' with the schema lookup answered first, then the stats query."""
    client.execute_sql.side_effect = [COLUMNS, STATS]
    payload = json.loads(describe_tables_json(client, table="listings", **kwargs))  # type: ignore[arg-type]
    assert isinstance(payload, dict)
    return payload


def test_column_stats_sql_counts_rows_and_every_column() -> None:
    """One aggregate for the whole table, not one query per column."""
    assert column_stats_sql("public.listings", ["id", "price"]) == (
        "SELECT COUNT(*) AS row_count, COUNT(id) AS n0, COUNT(price) AS n1 FROM public.listings"
    )


def test_column_stats_sql_rejects_a_column_that_is_not_an_identifier() -> None:
    with pytest.raises(ValueError, match="column must be a bare SQL identifier"):
        column_stats_sql("public.listings", ["id; DROP TABLE listings"])


def test_describe_reports_how_many_rows_hold_a_value() -> None:
    """price exists, is typed, and is NULL on all 7,535 rows."""
    payload = describing_listings(MagicMock())
    assert payload["row_count"] == 7535
    assert payload["columns"] == [
        {"name": "id", "type": "Int64", "non_null": 7535},
        {"name": "description", "type": "LargeUtf8", "non_null": 0},
    ]


def test_describe_says_what_non_null_means() -> None:
    payload = describing_listings(MagicMock())
    assert "empty" in str(payload["column_stats"])


def test_column_stats_can_be_turned_off() -> None:
    client = MagicMock()
    client.execute_sql.return_value = COLUMNS
    payload = json.loads(describe_tables_json(client, table="listings", column_stats=False))
    assert len(executed_sqls(client)) == 1
    assert "row_count" not in payload
    assert "non_null" not in payload["columns"][0]


def test_a_failed_stats_query_still_describes_the_schema() -> None:
    """Fails open: the schema is what the tool promised before the counts existed."""
    client = MagicMock()
    client.execute_sql.side_effect = [COLUMNS, RuntimeError("scan timed out")]
    payload = json.loads(describe_tables_json(client, table="listings"))
    assert [c["name"] for c in payload["columns"]] == ["id", "description"]
    assert "row_count" not in payload


def test_stats_are_counted_over_the_qualified_table() -> None:
    client = MagicMock()
    describing_listings(client)
    assert executed_sqls(client)[1].endswith("FROM public.listings")


def test_declared_but_unloaded_table_is_not_reported_as_missing(
    managed_db: ManagedDatabase,
) -> None:
    """It reports zero columns like a missing table, and every query against it fails."""
    client = MagicMock()
    client.execute_sql.return_value = result(
        ["table_schema", "table_name", "column_name", "data_type"], []
    )
    client.list_managed_tables.return_value = [
        SimpleNamespace(table="customer", schema="public", full_name="db.public.customer")
    ]
    payload = json.loads(describe_tables_json(client, table="public.customer", database=managed_db))
    assert "error" not in payload
    assert payload["row_count"] == 0
    assert "no data yet" in payload["note"]


def test_a_genuinely_missing_table_still_reports_an_error(managed_db: ManagedDatabase) -> None:
    client = MagicMock()
    client.execute_sql.return_value = result(
        ["table_schema", "table_name", "column_name", "data_type"], []
    )
    client.list_managed_tables.return_value = []
    payload = json.loads(describe_tables_json(client, table="nope", database=managed_db))
    assert "no table named" in payload["error"]


def test_describe_description_tells_the_model_to_check_non_null() -> None:
    assert "non_null" in default_describe_description()
    assert "non_null" not in default_describe_description(column_stats=False)


def test_describe_tool_describes_its_table_argument() -> None:
    tool = make_hotdata_describe_tables_tool(MagicMock())
    assert "description" in tool.args["table"]


def test_a_column_that_cannot_be_counted_does_not_take_the_stats_down() -> None:
    """Column names come from the table, so 'list price' is a data property, not a mistake."""
    client = MagicMock()
    client.execute_sql.side_effect = [
        result(
            ["table_schema", "table_name", "column_name", "data_type"],
            [
                ["public", "listings", "id", "Int64"],
                ["public", "listings", "list price", "Float64"],
            ],
        ),
        result(["row_count", "n0"], [[7535, 7535]]),
    ]
    payload = json.loads(describe_tables_json(client, table="listings"))
    assert payload["row_count"] == 7535
    assert payload["columns"][0]["non_null"] == 7535
    assert "non_null" not in payload["columns"][1]
    assert executed_sqls(client)[1] == (
        "SELECT COUNT(*) AS row_count, COUNT(id) AS n0 FROM public.listings"
    )
