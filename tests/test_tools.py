from __future__ import annotations

import json
import socket
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import MagicMock
from urllib.request import Request

import pytest
from hotdata_framework import LoadManagedTableResult, ManagedDatabase, QueryResult

from hotdata_langchain.databases import (
    _ValidatingRedirectHandler,
    create_managed_database,
    fetch_parquet,
    list_managed_databases_json,
    load_managed_table,
)
from hotdata_langchain.errors import HotdataToolError, engine_error_message
from hotdata_langchain.results import CLIENT_WARNING_KEY
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
from tests.conftest import FakeOpener


def test_result_rows_for_llm(sample_result: QueryResult) -> None:
    rows = result_rows_for_llm(sample_result, max_rows=1)
    assert rows == [{"n": 1}]


def test_execute_sql_json(mock_client: MagicMock, sample_result: QueryResult) -> None:
    payload = json.loads(execute_sql_json(mock_client, "select 1"))
    assert payload["metadata"]["row_count"] == 2
    assert payload["rows"] == [{"n": 1}, {"n": 2}]
    mock_client.execute_sql.assert_called_once_with("select 1", database=None)


def test_execute_sql_json_with_database(
    mock_client: MagicMock, sample_result: QueryResult, managed_db: ManagedDatabase
) -> None:
    execute_sql_json(mock_client, "select 1", database_id=managed_db)
    mock_client.execute_sql.assert_called_once_with("select 1", database=managed_db)


def test_list_managed_databases_json(mock_client: MagicMock) -> None:
    mock_client.list_managed_databases.return_value = [
        ManagedDatabase(id="c1", description="sales", default_connection_id="conn_c1"),
    ]
    payload = json.loads(list_managed_databases_json(mock_client))
    assert payload[0] == {"id": "c1", "name": "sales"}


def test_create_managed_database_delegates(mock_client: MagicMock) -> None:
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


def test_the_create_tool_reports_the_database_under_name(mock_client: MagicMock) -> None:
    """The create tool's payload shares the list tool's key, so the rename has to hold here too.

    `name` is also this tool's argument name, so the quoted-key guard in test_descriptions
    cannot tell the two apart — the payload is pinned directly instead.
    """
    mock_client.create_managed_database.return_value = ManagedDatabase(
        id="c1", description="sales", default_connection_id="conn_c1"
    )
    tools = {t.name: t for t in make_hotdata_tools(mock_client)}
    payload = json.loads(tools["hotdata_create_managed_database"].invoke({"name": "sales"}))
    assert payload == {"id": "c1", "name": "sales"}


def test_load_managed_table_delegates(
    mock_client: MagicMock, managed_db: ManagedDatabase, parquet_file: Path
) -> None:
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


def test_load_managed_table_rejects_a_path_that_is_not_there(
    mock_client: MagicMock, managed_db: ManagedDatabase
) -> None:
    """A raw FileNotFoundError says nothing about what the tool would have accepted."""
    with pytest.raises(FileNotFoundError, match="https:// URL"):
        load_managed_table(
            mock_client,
            database_id=managed_db,
            table="orders",
            file="/tmp/not-here.parquet",
        )
    mock_client.load_managed_table.assert_not_called()


def test_load_managed_table_accepts_a_url(
    mock_client: MagicMock,
    managed_db: ManagedDatabase,
    parquet_file: Path,
    serve: Callable[..., FakeOpener],
) -> None:
    """The only ingest route open to a deployed agent, which has no filesystem of its own."""
    mock_client.load_managed_table.return_value = LoadManagedTableResult(
        connection_id="c1",
        schema_name="public",
        table_name="orders",
        row_count=3,
        full_name="sales.public.orders",
    )
    serve(parquet_file.read_bytes())
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
    mock_client: MagicMock,
    managed_db: ManagedDatabase,
    parquet_file: Path,
    serve: Callable[..., FakeOpener],
) -> None:
    """A tool an agent calls in a loop must not leave a file behind on every call."""
    serve(parquet_file.read_bytes())
    load_managed_table(
        mock_client,
        database_id=managed_db,
        table="orders",
        file="https://example.test/orders.parquet",
    )
    assert not Path(mock_client.load_managed_table.call_args.kwargs["file"]).exists()


def test_the_downloaded_copy_is_removed_when_the_load_fails(
    mock_client: MagicMock,
    managed_db: ManagedDatabase,
    parquet_file: Path,
    serve: Callable[..., FakeOpener],
) -> None:
    mock_client.load_managed_table.side_effect = RuntimeError("Bad Request")
    serve(parquet_file.read_bytes())
    with pytest.raises(RuntimeError):
        load_managed_table(
            mock_client,
            database_id=managed_db,
            table="orders",
            file="https://example.test/orders.parquet",
        )
    assert not Path(mock_client.load_managed_table.call_args.kwargs["file"]).exists()


def test_a_url_that_returns_a_login_page_is_rejected_before_upload(
    serve: Callable[..., FakeOpener],
) -> None:
    """It answers 200 with HTML, so nothing before the magic-byte check notices."""
    serve(b"<html>sign in</html>")
    with pytest.raises(ValueError, match="did not return a parquet file"):
        fetch_parquet("https://example.test/orders.parquet")


def test_a_failed_fetch_leaves_no_temporary_file(
    serve: Callable[..., FakeOpener], monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[str] = []
    real_named_temp = tempfile.NamedTemporaryFile

    def record(*args: Any, **kwargs: Any) -> Any:
        handle = real_named_temp(*args, **kwargs)
        created.append(handle.name)
        return handle

    monkeypatch.setattr("hotdata_langchain.databases.tempfile.NamedTemporaryFile", record)
    serve(b"<html>sign in</html>")
    with pytest.raises(ValueError, match="did not return a parquet file"):
        fetch_parquet("https://example.test/orders.parquet")
    assert created and not any(Path(name).exists() for name in created)


def test_fetch_parquet_refuses_a_non_http_url() -> None:
    """urlopen would honour file://, which would turn this into a local file reader."""
    with pytest.raises(ValueError, match="http:// or https:// URL"):
        fetch_parquet("file:///etc/passwd")


def test_the_fetch_sets_a_user_agent(serve: Callable[..., FakeOpener], parquet_file: Path) -> None:
    """Asset hosts 403 urllib's default, which the demo hit and worked around by hand."""
    opener = serve(parquet_file.read_bytes())
    Path(fetch_parquet("https://example.test/orders.parquet")).unlink()
    assert opener.requests[0].headers.get("User-agent") == "hotdata-langchain"


# The URL is chosen by the model, and the model's inputs include text it retrieved, so a
# planted link is enough to pick one. These pin that the fetch cannot be steered inwards.


@pytest.mark.parametrize(
    "address",
    ["169.254.169.254", "127.0.0.1", "10.0.0.1", "192.168.1.5", "::1", "::ffff:127.0.0.1"],
)
def test_a_url_resolving_to_a_private_address_is_refused(
    address: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cloud metadata endpoint and an internal service are both one planted link away."""
    monkeypatch.setattr(
        "hotdata_langchain.databases.socket.getaddrinfo",
        lambda host, port: [(0, 0, 0, "", (address, 0))],
    )
    with pytest.raises(ValueError, match="not a public address"):
        fetch_parquet("https://internal.test/orders.parquet")


def test_every_resolved_address_is_checked_not_just_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host answering with one public and one private address must not pass on the public one."""
    monkeypatch.setattr(
        "hotdata_langchain.databases.socket.getaddrinfo",
        lambda host, port: [(0, 0, 0, "", ("93.184.216.34", 0)), (0, 0, 0, "", ("10.0.0.1", 0))],
    )
    with pytest.raises(ValueError, match="not a public address"):
        fetch_parquet("https://split.test/orders.parquet")


def test_a_private_host_is_allowed_when_the_deployment_says_so(
    monkeypatch: pytest.MonkeyPatch, parquet_file: Path
) -> None:
    """Loading from an internal store is legitimate; it just has to be chosen deliberately."""
    monkeypatch.setattr(
        "hotdata_langchain.databases.socket.getaddrinfo",
        lambda host, port: [(0, 0, 0, "", ("10.0.0.1", 0))],
    )
    opener = FakeOpener(parquet_file.read_bytes(), {})
    monkeypatch.setattr("hotdata_langchain.databases.build_opener", lambda *h: opener)
    path = fetch_parquet("https://minio.internal/orders.parquet", allow_private_hosts=True)
    Path(path).unlink()


def test_a_redirect_to_a_private_address_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checking only the URL as written is defeated by a public URL that 302s inwards."""
    resolved = {"public.test": "93.184.216.34", "internal.test": "169.254.169.254"}
    monkeypatch.setattr(
        "hotdata_langchain.databases.socket.getaddrinfo",
        lambda host, port: [(0, 0, 0, "", (resolved[host], 0))],
    )
    handler = _ValidatingRedirectHandler(allow_private_hosts=False)
    with pytest.raises(ValueError, match="not a public address"):
        handler.redirect_request(
            Request("https://public.test/orders.parquet"),
            None,
            302,
            "Found",
            {},
            "https://internal.test/latest/meta-data/",
        )


def test_an_unresolvable_host_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(host: str, port: int) -> NoReturn:
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr("hotdata_langchain.databases.socket.getaddrinfo", explode)
    with pytest.raises(ValueError, match="could not resolve"):
        fetch_parquet("https://nowhere.test/orders.parquet")


def test_an_oversized_download_is_refused_from_the_declared_length(
    serve: Callable[..., FakeOpener],
) -> None:
    """Declared up front, so the transfer never starts."""
    opener = serve(b"PAR1" + b"x" * 100, {"Content-Length": str(10 * 1024**3)})
    with pytest.raises(ValueError, match=r"over the \d+-byte limit"):
        fetch_parquet("https://example.test/huge.parquet")
    assert opener.requests, "the request was made, and the body was never read"


def test_an_oversized_download_is_refused_while_streaming(serve: Callable[..., FakeOpener]) -> None:
    """Content-Length is optional and can lie, so the bytes are counted regardless."""
    serve(b"PAR1" + b"x" * 5000)
    with pytest.raises(ValueError, match="over the 1024-byte limit"):
        fetch_parquet("https://example.test/huge.parquet", max_bytes=1024)


def test_an_oversized_download_leaves_no_temporary_file(
    serve: Callable[..., FakeOpener], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the cap is the disk, so a refusal that kept the bytes would defeat it."""
    created: list[str] = []
    real_named_temp = tempfile.NamedTemporaryFile

    def record(*args: Any, **kwargs: Any) -> Any:
        handle = real_named_temp(*args, **kwargs)
        created.append(handle.name)
        return handle

    monkeypatch.setattr("hotdata_langchain.databases.tempfile.NamedTemporaryFile", record)
    serve(b"PAR1" + b"x" * 5000)
    with pytest.raises(ValueError, match="over the"):
        fetch_parquet("https://example.test/huge.parquet", max_bytes=1024)
    assert created and not any(Path(name).exists() for name in created)


def test_a_file_inside_the_cap_is_kept(
    serve: Callable[..., FakeOpener], parquet_file: Path
) -> None:
    payload = parquet_file.read_bytes()
    serve(payload, {"Content-Length": str(len(payload))})
    path = fetch_parquet("https://example.test/orders.parquet", max_bytes=len(payload))
    assert Path(path).read_bytes() == payload
    Path(path).unlink()


def test_make_hotdata_tools(
    mock_client: MagicMock,
    sample_result: QueryResult,
    managed_db: ManagedDatabase,
    databases_api: MagicMock,
    parquet_file: Path,
) -> None:
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


def test_every_tool_name_is_an_exported_constant(mock_client: MagicMock) -> None:
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


def test_management_tools_can_be_left_out(mock_client: MagicMock) -> None:
    """An agent reading one fixed database cannot use them, and can only misuse them."""
    tools = make_hotdata_tools(mock_client, management_tools=False)
    assert {tool.name for tool in tools} == {
        DEFAULT_SQL_TOOL_NAME,
        DEFAULT_DESCRIBE_TOOL_NAME,
    }


def test_management_tools_are_on_by_default(mock_client: MagicMock) -> None:
    tools = {tool.name for tool in make_hotdata_tools(mock_client)}
    assert DEFAULT_CREATE_DATABASE_TOOL_NAME in tools
    assert DEFAULT_LOAD_TABLE_TOOL_NAME in tools


def test_the_sql_tool_stays_first_whichever_tools_are_included(mock_client: MagicMock) -> None:
    """It is the one every agent needs; the ordering the model sees should not shift."""
    kwarg_sets: tuple[dict[str, Any], ...] = (
        {},
        {"management_tools": False},
        {"describe_tables": False},
    )
    for kwargs in kwarg_sets:
        tools = make_hotdata_tools(mock_client, **kwargs)
        assert tools[0].name == DEFAULT_SQL_TOOL_NAME


# --- Results that succeeded without doing what they said ----------------------------


def test_sql_result_warns_about_an_uninterpreted_format_pattern(mock_client: MagicMock) -> None:
    """The measured failure: correct numbers labelled with the literal text 'YYYY-MM-DD'."""
    payload = json.loads(
        execute_sql_json(mock_client, "SELECT to_char(start_time, 'YYYY-MM-DD') AS day FROM spans")
    )
    assert "'YYYY-MM-DD'" in payload["metadata"][CLIENT_WARNING_KEY]


def test_a_correct_query_carries_no_client_warning(mock_client: MagicMock) -> None:
    payload = json.loads(execute_sql_json(mock_client, "SELECT to_char(d, '%Y-%m-%d') FROM t"))
    assert CLIENT_WARNING_KEY not in payload["metadata"]


def test_capped_sql_result_says_how_many_rows_matched(mock_client: MagicMock) -> None:
    payload = json.loads(execute_sql_json(mock_client, "SELECT n FROM t", max_rows=1))
    assert len(payload["rows"]) == 1
    assert "2" in payload["metadata"][CLIENT_WARNING_KEY]


def test_sql_description_states_the_row_cap(mock_client: MagicMock) -> None:
    """A model that inferred the cap itself guessed the boundary and re-read four rows."""
    tools = {tool.name: tool for tool in make_hotdata_tools(mock_client, max_rows=250)}
    description = tools[DEFAULT_SQL_TOOL_NAME].description or ""
    assert "250" in description
    assert "row_count" in description


def test_tool_arguments_carry_descriptions(mock_client: MagicMock) -> None:
    """Until now the schema told the model a parameter's type and nothing else."""
    tools = {tool.name: tool for tool in make_hotdata_tools(mock_client)}
    assert "description" in tools[DEFAULT_SQL_TOOL_NAME].args["sql"]
    assert "description" in tools[DEFAULT_LOAD_TABLE_TOOL_NAME].args["database_id"]
    assert "description" in tools[DEFAULT_CREATE_DATABASE_TOOL_NAME].args["name"]


def test_tool_descriptions_are_unchanged_by_parsing_the_docstrings(mock_client: MagicMock) -> None:
    """The explicit description= is what reaches the model, not the docstring summary."""
    tools = {tool.name: tool for tool in make_hotdata_tools(mock_client)}
    assert (tools[DEFAULT_SQL_TOOL_NAME].description or "").startswith("Run a read-only SQL query")


def test_a_failing_query_still_reports_the_format_pattern(mock_client: MagicMock) -> None:
    """Applying a Postgres template to a column returns only 'an internal server error'."""
    mock_client.execute_sql.side_effect = RuntimeError("An internal server error occurred.")
    sql = "SELECT to_char(to_date(first_review, 'YYYY-MM-DD'), 'YYYY-MM') FROM listings"
    with pytest.raises(HotdataToolError) as raised:
        execute_sql_json(mock_client, sql)
    message = engine_error_message(raised.value)
    assert "'YYYY-MM-DD'" in message
    assert message.startswith("to_char")
    assert "An internal server error occurred." in message


def test_a_failure_with_nothing_to_add_is_raised_untouched(mock_client: MagicMock) -> None:
    """Wrapping every failure would put this package's name on errors it knows nothing about."""
    original = RuntimeError("Bad Request")
    mock_client.execute_sql.side_effect = original
    with pytest.raises(RuntimeError) as raised:
        execute_sql_json(mock_client, "SELECT nope FROM listings")
    assert raised.value is original


def test_the_format_hint_survives_the_error_feedback_wrapper(mock_client: MagicMock) -> None:
    """It is the failure path the model reads, so the hint has to reach the payload."""
    mock_client.execute_sql.side_effect = RuntimeError("An internal server error occurred.")
    tools = {tool.name: tool for tool in make_hotdata_tools(mock_client, handle_errors=True)}
    payload = json.loads(
        tools[DEFAULT_SQL_TOOL_NAME].invoke({"sql": "SELECT to_char(d, 'YYYY-MM-DD') FROM t"})
    )
    assert "Write '%Y-%m-%d' instead." in payload["error"]


# --- Telling two tool sets apart --------------------------------------------------
#
# One client can query many databases, so registering two tool sets in one agent is a
# supported shape. Without a suffix both sets register `hotdata_execute_sql`, and the
# model is handed two tools it cannot address or choose between.


SALES = ManagedDatabase(id="dbid1", description="sales", default_connection_id="conn1")
SUPPORT = ManagedDatabase(id="dbid2", description="support", default_connection_id="conn2")


def _names(client: MagicMock, **kwargs: object) -> list[str]:
    return [t.name for t in make_hotdata_tools(client, **kwargs)]  # type: ignore[arg-type]


def test_two_tool_sets_collide_without_a_suffix(mock_client: MagicMock) -> None:
    """The defect this exists to fix, pinned so the fix cannot be mistaken for unnecessary."""
    both = _names(mock_client, database_id=SALES) + _names(mock_client, database_id=SUPPORT)
    assert len(set(both)) < len(both)


def test_a_suffix_makes_every_name_in_the_set_distinct(mock_client: MagicMock) -> None:
    both = _names(mock_client, database_id=SALES, tool_name_suffix="sales") + _names(
        mock_client, database_id=SUPPORT, tool_name_suffix="support"
    )
    assert len(set(both)) == len(both)
    assert "hotdata_execute_sql_sales" in both
    assert "hotdata_execute_sql_support" in both


def test_an_explicit_search_tool_name_is_used_as_given(mock_client: MagicMock) -> None:
    """Naming that tool is already the caller's decision, so the suffix does not second-guess it."""
    names = _names(
        mock_client,
        database_id=SALES,
        search_table="default.public.listings",
        search_column="description",
        search_tool_name="search_sales",
        tool_name_suffix="sales",
    )
    assert "search_sales" in names
    assert "hotdata_search_text_sales" not in names


@pytest.mark.parametrize(
    "suffix",
    [
        pytest.param("has a space", id="space"),
        pytest.param("dots.not.allowed", id="dot"),
        pytest.param("x" * 60, id="too-long"),
    ],
)
def test_a_suffix_that_no_provider_would_accept_fails_at_build_time(
    mock_client: MagicMock, suffix: str
) -> None:
    """Rejected here rather than at the provider, which sees it only on the first call."""
    with pytest.raises(ValueError):
        make_hotdata_tools(mock_client, database_id=SALES, tool_name_suffix=suffix)


def test_the_length_limit_is_checked_against_the_tools_actually_built(
    mock_client: MagicMock,
) -> None:
    """Otherwise the error names a tool the caller excluded, which reads as a bug.

    `hotdata_create_managed_database` is the longest base name in the set, so a suffix can
    be too long for it and fine for everything else.
    """
    suffix = "x" * 33
    names = _names(mock_client, database_id=SALES, management_tools=False, tool_name_suffix=suffix)
    assert names and all(n.endswith(suffix) for n in names)

    with pytest.raises(ValueError, match="hotdata_create_managed_database"):
        make_hotdata_tools(mock_client, database_id=SALES, tool_name_suffix=suffix)


def test_the_database_scoped_tools_name_their_database(mock_client: MagicMock) -> None:
    tools = {t.name: t for t in make_hotdata_tools(mock_client, database_id=SALES)}
    for name in ("hotdata_execute_sql", "hotdata_describe_tables"):
        assert (tools[name].description or "").startswith("Works on the 'sales' database.")


def test_the_workspace_tools_do_not_claim_a_database(mock_client: MagicMock) -> None:
    """Listing, creating and loading act on the workspace, so naming one would be false."""
    tools = {t.name: t for t in make_hotdata_tools(mock_client, database_id=SALES)}
    for name in (
        "hotdata_list_managed_databases",
        "hotdata_create_managed_database",
        "hotdata_load_managed_table",
    ):
        assert "Works on the" not in (tools[name].description or "")


def test_the_label_defaults_to_the_database_name_and_can_be_overridden(
    mock_client: MagicMock,
) -> None:
    tools = {
        t.name: t
        for t in make_hotdata_tools(mock_client, database_id=SALES, label="EU support desk")
    }
    assert (tools["hotdata_execute_sql"].description or "").startswith(
        "Works on the 'EU support desk' database."
    )


def test_a_database_with_no_name_gets_no_sentence_rather_than_one_naming_its_id(
    mock_client: MagicMock,
) -> None:
    """An id is not a name; presenting one as a name invites passing it where a name goes."""
    unnamed = ManagedDatabase(id="dbid3", description=None, default_connection_id="conn3")
    tools = {t.name: t for t in make_hotdata_tools(mock_client, database_id=unnamed)}
    described = tools["hotdata_execute_sql"].description or ""
    assert "Works on the" not in described
    assert "dbid3" not in described


def test_an_unscoped_tool_set_names_no_database(mock_client: MagicMock) -> None:
    tools = {t.name: t for t in make_hotdata_tools(mock_client)}
    assert "Works on the" not in (tools["hotdata_execute_sql"].description or "")
