"""Instant databases are addressed by id, never by name.

A Hotdata database name is a display label and is not unique, so a by-name lookup can
resolve to the wrong database — and then every query, load and drop follows it there.
The framework's ``resolve_managed_database`` still offers that by-name fallback; these
tests pin that this package never reaches it, and that ``GET /databases/{id}`` is the
only lookup it makes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hotdata.exceptions import ApiException
from hotdata_framework import LoadManagedTableResult, ManagedDatabase, QueryResult

from hotdata_langchain.databases import (
    CATALOG_QUERY,
    query_catalogs,
    query_scope,
    resolve_database_by_id,
)
from hotdata_langchain.schema import make_hotdata_describe_tables_tool
from hotdata_langchain.search import make_hotdata_search_tool
from hotdata_langchain.tools import execute_sql_json, make_hotdata_tools

TABLE = "default.public.listings"
COLUMN = "description"


# --- resolution --------------------------------------------------------------------


def test_resolve_fetches_the_record_by_id(
    mock_client: MagicMock, managed_db: ManagedDatabase, databases_api: MagicMock
) -> None:
    resolved = resolve_database_by_id(mock_client, managed_db.id)
    assert resolved == managed_db
    databases_api.return_value.get_database.assert_called_once_with(managed_db.id)


def test_resolve_never_lists_or_matches_on_a_name(
    mock_client: MagicMock, managed_db: ManagedDatabase, databases_api: MagicMock
) -> None:
    """Listing is how a name match would be found, so it must not happen at all."""
    resolve_database_by_id(mock_client, managed_db.id)
    databases_api.return_value.list_databases.assert_not_called()
    mock_client.list_managed_databases.assert_not_called()
    mock_client.resolve_managed_database.assert_not_called()


def test_resolve_passes_an_already_resolved_record_through_without_a_lookup(
    mock_client: MagicMock, managed_db: ManagedDatabase, databases_api: MagicMock
) -> None:
    assert resolve_database_by_id(mock_client, managed_db) is managed_db
    databases_api.return_value.get_database.assert_not_called()


def test_resolve_raises_key_error_for_an_unknown_id(
    mock_client: MagicMock, databases_api: MagicMock
) -> None:
    """A name lands here: it is not an id, so it 404s rather than matching a label."""
    databases_api.return_value.get_database.side_effect = ApiException(
        status=404, reason="Not Found"
    )
    with pytest.raises(KeyError) as excinfo:
        resolve_database_by_id(mock_client, "sf_airbnb")
    message = str(excinfo.value)
    assert "hotdata_list_managed_databases" in message
    assert "not unique" in message


def test_resolve_surfaces_other_api_errors(
    mock_client: MagicMock, databases_api: MagicMock
) -> None:
    databases_api.return_value.get_database.side_effect = ApiException(
        status=403, reason="Forbidden", body="workspace does not permit reads"
    )
    with pytest.raises(RuntimeError, match="workspace does not permit reads"):
        resolve_database_by_id(mock_client, "dbid000000000000000000000000x")


# --- query scopes ------------------------------------------------------------------


def test_query_scope_accepts_a_resolved_record_and_none(managed_db: ManagedDatabase) -> None:
    assert query_scope(managed_db) is managed_db
    assert query_scope(None) is None


def test_query_scope_rejects_an_id_or_name_string() -> None:
    """Strings reach the framework's name-or-id resolver, which is what we are avoiding."""
    with pytest.raises(TypeError, match="resolve_database_by_id"):
        query_scope("dbid000000000000000000000000x")  # type: ignore[arg-type]


def test_execute_sql_json_resolves_an_id_string_before_scoping(
    mock_client: MagicMock, managed_db: ManagedDatabase, databases_api: MagicMock
) -> None:
    """The id goes through this package's own resolver, so the scope is a fetched record."""
    execute_sql_json(mock_client, "SELECT 1", database_id=managed_db.id)
    databases_api.return_value.get_database.assert_called_once_with(managed_db.id)
    mock_client.execute_sql.assert_called_once_with("SELECT 1", database=managed_db)


def test_execute_sql_json_takes_a_resolved_record_without_a_second_lookup(
    mock_client: MagicMock, managed_db: ManagedDatabase, databases_api: MagicMock
) -> None:
    execute_sql_json(mock_client, "SELECT 1", database_id=managed_db)
    databases_api.return_value.get_database.assert_not_called()
    mock_client.execute_sql.assert_called_once_with("SELECT 1", database=managed_db)


def test_execute_sql_json_refuses_a_name(mock_client: MagicMock, databases_api: MagicMock) -> None:
    """A name is not an id, so it 404s here rather than scoping the query elsewhere."""
    databases_api.return_value.get_database.side_effect = ApiException(
        status=404, reason="Not Found"
    )
    with pytest.raises(KeyError, match="not accepted here"):
        execute_sql_json(mock_client, "SELECT 1", database_id="sf_airbnb")
    mock_client.execute_sql.assert_not_called()


# --- factories resolve once --------------------------------------------------------


def test_tool_set_resolves_the_database_exactly_once(
    mock_client: MagicMock, managed_db: ManagedDatabase, databases_api: MagicMock
) -> None:
    """SQL, schema and search tools share one resolved record — not one lookup each."""
    make_hotdata_tools(
        mock_client,
        database_id=managed_db.id,
        search_table=TABLE,
        search_column=COLUMN,
    )
    databases_api.return_value.get_database.assert_called_once_with(managed_db.id)


def test_tool_set_scopes_every_query_to_the_resolved_record(
    mock_client: MagicMock, managed_db: ManagedDatabase, databases_api: MagicMock
) -> None:
    tools = {t.name: t for t in make_hotdata_tools(mock_client, database_id=managed_db.id)}
    tools["hotdata_execute_sql"].invoke({"sql": "SELECT 1"})
    assert mock_client.execute_sql.call_args.kwargs == {"database": managed_db}


def test_tool_set_rejects_a_database_name_at_build_time(
    mock_client: MagicMock, databases_api: MagicMock
) -> None:
    """Fail while wiring the tools up, not on the agent's first query."""
    databases_api.return_value.get_database.side_effect = ApiException(
        status=404, reason="Not Found"
    )
    with pytest.raises(KeyError):
        make_hotdata_tools(mock_client, database_id="sf_airbnb")


def test_an_unscoped_tool_set_makes_no_lookup(
    mock_client: MagicMock, databases_api: MagicMock
) -> None:
    make_hotdata_tools(mock_client)
    databases_api.return_value.get_database.assert_not_called()


@pytest.mark.parametrize(
    "factory",
    [
        lambda client, database_id: make_hotdata_describe_tables_tool(
            client, database_id=database_id
        ),
        lambda client, database_id: make_hotdata_search_tool(
            client, table=TABLE, column=COLUMN, database_id=database_id
        ),
    ],
)
def test_standalone_factories_resolve_by_id(
    factory: object,
    mock_client: MagicMock,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
) -> None:
    factory(mock_client, managed_db.id)  # type: ignore[operator]
    databases_api.return_value.get_database.assert_called_once_with(managed_db.id)


# --- the agent-facing load tool ----------------------------------------------------


def load_tool(client: MagicMock) -> object:
    return {t.name: t for t in make_hotdata_tools(client)}["hotdata_load_managed_table"]


def test_load_tool_takes_a_database_id_argument(mock_client: MagicMock) -> None:
    """The argument name is what the model sees in the schema, so it must say 'id'."""
    tool = load_tool(mock_client)
    assert set(tool.args) == {  # type: ignore[attr-defined]
        "database_id",
        "table",
        "file",
        "schema_name",
        "mode",
        "key",
    }


def test_load_tool_resolves_the_agent_supplied_id_by_id(
    mock_client: MagicMock,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
    parquet_file: Path,
) -> None:
    mock_client.load_managed_table.return_value = LoadManagedTableResult(
        connection_id=managed_db.default_connection_id,
        schema_name="public",
        table_name="orders",
        row_count=1,
        full_name=f"{managed_db.id}.public.orders",
    )
    payload = json.loads(
        load_tool(mock_client).invoke(  # type: ignore[attr-defined]
            {
                "database_id": managed_db.id,
                "table": "orders",
                "file": str(parquet_file),
            }
        )
    )
    databases_api.return_value.get_database.assert_called_once_with(managed_db.id)
    # The resolved record addresses the load, so no name can select the overwrite target.
    assert mock_client.load_managed_table.call_args.args[0] == managed_db
    assert payload["row_count"] == 1


def test_load_tool_rejects_a_database_name(
    mock_client: MagicMock, databases_api: MagicMock
) -> None:
    """An agent that passes a label gets an error naming the tool that yields ids."""
    databases_api.return_value.get_database.side_effect = ApiException(
        status=404, reason="Not Found"
    )
    with pytest.raises(KeyError, match="hotdata_list_managed_databases"):
        load_tool(mock_client).invoke(  # type: ignore[attr-defined]
            {"database_id": "sales", "table": "orders", "file": "/tmp/orders.parquet"}
        )
    mock_client.load_managed_table.assert_not_called()


def test_load_description_tells_the_model_where_ids_come_from(mock_client: MagicMock) -> None:
    description = load_tool(mock_client).description or ""  # type: ignore[attr-defined]
    assert "hotdata_list_managed_databases" in description
    assert "hotdata_create_managed_database" in description


# --- catalog lookup ----------------------------------------------------------------
#
# There is no catalog name that holds for both database kinds: an instant database's
# tables answer to `default`, an attached source's answer to the attachment's alias, and
# the database record reports `default_catalog='default'` either way. So the SQL tool
# description names the catalog only because this lookup found it, which makes the
# lookup part of the model-facing contract rather than an internal detail.


def catalog_result(*catalogs: str) -> QueryResult:
    """A ``QueryResult`` shaped as ``CATALOG_QUERY`` returns one: one column, one row each."""
    return QueryResult(
        columns=["table_catalog"],
        rows=[[c] for c in catalogs],
        row_count=len(catalogs),
        result_id="res_catalogs",
        query_run_id="run_catalogs",
        execution_time_ms=9,
        warning=None,
        error_message=None,
    )


def test_query_catalogs_reads_information_schema_in_the_database_scope(
    mock_client: MagicMock, managed_db: ManagedDatabase
) -> None:
    mock_client.execute_sql.return_value = catalog_result("default")
    assert query_catalogs(mock_client, managed_db) == ["default"]
    sql, kwargs = mock_client.execute_sql.call_args[0][0], mock_client.execute_sql.call_args[1]
    assert sql == CATALOG_QUERY
    assert "information_schema.tables" in sql
    assert kwargs["database"] == managed_db


def test_query_catalogs_reports_an_attachment_alias(
    mock_client: MagicMock, managed_db: ManagedDatabase
) -> None:
    """Verified live against an attached Postgres source: the catalog is 'f1', not 'default'."""
    mock_client.execute_sql.return_value = catalog_result("f1")
    assert query_catalogs(mock_client, managed_db) == ["f1"]


def test_query_catalogs_excludes_information_schema_itself(
    mock_client: MagicMock, managed_db: ManagedDatabase
) -> None:
    """A catalog holding nothing else would otherwise mask the one worth naming."""
    assert "table_schema <> 'information_schema'" in CATALOG_QUERY


def test_query_catalogs_dedupes_and_sorts(
    mock_client: MagicMock, managed_db: ManagedDatabase
) -> None:
    mock_client.execute_sql.return_value = catalog_result("f1", "default", "f1")
    assert query_catalogs(mock_client, managed_db) == ["default", "f1"]


def test_query_catalogs_returns_empty_when_the_query_fails(
    mock_client: MagicMock, managed_db: ManagedDatabase
) -> None:
    """Building tools must not fail because the description could not name the catalog."""
    mock_client.execute_sql.side_effect = RuntimeError("Bad Request")
    assert query_catalogs(mock_client, managed_db) == []


def test_query_catalogs_warns_when_it_degrades(
    mock_client: MagicMock, managed_db: ManagedDatabase, caplog: pytest.LogCaptureFixture
) -> None:
    """Swallowing is right; swallowing silently hides a weaker model-facing contract."""
    mock_client.execute_sql.side_effect = RuntimeError("Bad Request")
    with caplog.at_level(logging.WARNING, logger="hotdata_langchain.databases"):
        query_catalogs(mock_client, managed_db)
    assert any(managed_db.id in r.getMessage() for r in caplog.records)
    assert all(r.levelno >= logging.WARNING for r in caplog.records)


def test_scoped_tools_name_the_catalog_the_lookup_found(
    mock_client: MagicMock, managed_db: ManagedDatabase, databases_api: MagicMock
) -> None:
    """The whole path: resolve the database, read its catalog, state it to the model."""
    mock_client.execute_sql.return_value = catalog_result("f1")
    tools = make_hotdata_tools(mock_client, database_id=managed_db.id)
    description = next(t for t in tools if t.name == "hotdata_execute_sql").description or ""
    assert "the catalog is 'f1'" in description


def test_an_explicit_catalog_skips_the_lookup(
    mock_client: MagicMock, managed_db: ManagedDatabase, databases_api: MagicMock
) -> None:
    tools = make_hotdata_tools(mock_client, database_id=managed_db.id, catalog="warehouse")
    description = next(t for t in tools if t.name == "hotdata_execute_sql").description or ""
    assert "the catalog is 'warehouse'" in description
    mock_client.execute_sql.assert_not_called()
