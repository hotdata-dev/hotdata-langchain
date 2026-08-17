from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import MagicMock

import pytest
from hotdata_framework import ManagedDatabase, QueryResult

from hotdata_langchain.results import CLIENT_WARNING_KEY
from hotdata_langchain.search import (
    DEFAULT_SEARCH_LIMIT,
    bm25_search_json,
    bm25_search_sql,
    default_search_description,
    make_hotdata_search_tool,
)
from hotdata_langchain.tools import make_hotdata_tools

TABLE = "default.public.listings"
COLUMN = "description"
QUERY = "cozy apartment"

# Matches the four-argument bm25_search(...) call, capturing the trailing limit. The
# fourth argument is what bounds the engine's search; a trailing SQL LIMIT does not
# reach the scan through ORDER BY.
_SEARCH_CALL_RE = re.compile(
    r"bm25_search\('[^']*', '[^']*', '.*', (\d+)\)",
)


def executed_sql(client: MagicMock) -> str:
    sql = client.execute_sql.call_args.args[0]
    assert isinstance(sql, str)
    return sql


# --- SQL construction -------------------------------------------------------------


def test_bm25_search_sql_full_shape() -> None:
    assert bm25_search_sql(table=TABLE, column=COLUMN, query=QUERY, k=5) == (
        "SELECT description, score "
        "FROM bm25_search('default.public.listings', 'description', 'cozy apartment', 5) "
        "ORDER BY score DESC "
        "LIMIT 5"
    )


def test_bm25_search_sql_passes_k_as_explicit_fourth_argument() -> None:
    """The search bound must be an argument, not only a trailing LIMIT."""
    sql = bm25_search_sql(table=TABLE, column=COLUMN, query=QUERY, k=7)
    match = _SEARCH_CALL_RE.search(sql)
    assert match is not None, f"bm25_search call is not in four-argument form: {sql}"
    assert match.group(1) == "7"
    assert sql.endswith("LIMIT 7")


def test_bm25_search_sql_orders_by_score_descending() -> None:
    """The engine returns hits in rowid order, so ranking has to be requested."""
    sql = bm25_search_sql(table=TABLE, column=COLUMN, query=QUERY)
    assert "ORDER BY score DESC" in sql
    assert sql.index("ORDER BY score DESC") < sql.index("LIMIT")


def test_bm25_search_sql_defaults_k_to_search_limit() -> None:
    sql = bm25_search_sql(table=TABLE, column=COLUMN, query=QUERY)
    assert sql.endswith(f"LIMIT {DEFAULT_SEARCH_LIMIT}")


def test_bm25_search_sql_defaults_projection_to_searched_column_and_score() -> None:
    sql = bm25_search_sql(table=TABLE, column=COLUMN, query=QUERY)
    assert sql.startswith("SELECT description, score FROM")


def test_bm25_search_sql_projects_requested_columns_with_score_last() -> None:
    sql = bm25_search_sql(
        table=TABLE,
        column=COLUMN,
        query=QUERY,
        columns=["id", "name", "description"],
    )
    assert sql.startswith("SELECT id, name, description, score FROM")


def test_bm25_search_sql_does_not_duplicate_score_column() -> None:
    sql = bm25_search_sql(table=TABLE, column=COLUMN, query=QUERY, columns=["id", "score", "name"])
    projection = sql[len("SELECT ") : sql.index(" FROM")]
    assert projection == "id, name, score"


def test_bm25_search_sql_escapes_single_quotes_in_query() -> None:
    sql = bm25_search_sql(table=TABLE, column=COLUMN, query="host's place")
    assert "'host''s place'" in sql


def test_bm25_search_sql_neutralises_quote_injection() -> None:
    """LLM-authored search text reaches a SQL literal, so quotes must not terminate it."""
    sql = bm25_search_sql(table=TABLE, column=COLUMN, query="x') OR 1=1 --")
    assert "'x'') OR 1=1 --'" in sql
    assert _SEARCH_CALL_RE.search(sql) is not None
    assert sql.count("bm25_search(") == 1


def test_bm25_search_sql_rejects_null_byte_in_query() -> None:
    with pytest.raises(ValueError, match="null bytes"):
        bm25_search_sql(table=TABLE, column=COLUMN, query="a\x00b")


@pytest.mark.parametrize(
    "table",
    [
        "listings",
        "public.listings",
        "default.public.listings.extra",
        "default.public.'; DROP TABLE x; --",
        "default..listings",
        "",
    ],
)
def test_bm25_search_sql_rejects_bad_table_reference(table: str) -> None:
    with pytest.raises(ValueError, match=r"catalog\.schema\.table"):
        bm25_search_sql(table=table, column=COLUMN, query=QUERY)


@pytest.mark.parametrize("column", ["desc ription", "desc'ription", "1description", ""])
def test_bm25_search_sql_rejects_bad_column(column: str) -> None:
    with pytest.raises(ValueError, match="bare SQL identifier"):
        bm25_search_sql(table=TABLE, column=column, query=QUERY)


def test_bm25_search_sql_rejects_bad_projection_column() -> None:
    with pytest.raises(ValueError, match="bare SQL identifier"):
        bm25_search_sql(table=TABLE, column=COLUMN, query=QUERY, columns=["id; DROP TABLE x"])


@pytest.mark.parametrize("k", [0, -1])
def test_bm25_search_sql_rejects_non_positive_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        bm25_search_sql(table=TABLE, column=COLUMN, query=QUERY, k=k)


def test_bm25_search_sql_rejects_empty_columns() -> None:
    with pytest.raises(ValueError, match="columns must not be empty"):
        bm25_search_sql(table=TABLE, column=COLUMN, query=QUERY, columns=[])


# --- JSON envelope ----------------------------------------------------------------


def test_bm25_search_json_returns_metadata_and_rows(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    mock_client.execute_sql.return_value = search_result
    payload = json.loads(bm25_search_json(mock_client, table=TABLE, column=COLUMN, query=QUERY))
    assert set(payload) == {"metadata", "rows"}
    assert payload["metadata"]["row_count"] == 2
    assert payload["rows"][0] == {"description": "Cozy apartment with a view", "score": 8.5}


def test_bm25_search_json_scopes_query_to_database(
    mock_client: MagicMock, search_result: QueryResult, managed_db: ManagedDatabase
) -> None:
    mock_client.execute_sql.return_value = search_result
    bm25_search_json(mock_client, table=TABLE, column=COLUMN, query=QUERY, database=managed_db)
    assert mock_client.execute_sql.call_args.kwargs == {"database": managed_db}


def test_bm25_search_json_refuses_an_unresolved_database_scope(mock_client: MagicMock) -> None:
    """A bare string would reach the framework's by-name fallback."""
    with pytest.raises(TypeError, match="resolve_database_by_id"):
        bm25_search_json(
            mock_client,
            table=TABLE,
            column=COLUMN,
            query=QUERY,
            database="sf_airbnb",  # type: ignore[arg-type]
        )


def test_bm25_search_json_truncates_rows_to_max_rows(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    mock_client.execute_sql.return_value = search_result
    payload = json.loads(
        bm25_search_json(mock_client, table=TABLE, column=COLUMN, query=QUERY, max_rows=1)
    )
    assert len(payload["rows"]) == 1
    assert payload["metadata"]["row_count"] == 2


# --- Tool surface -----------------------------------------------------------------


def test_search_tool_name_and_arguments(mock_client: MagicMock) -> None:
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN)
    assert tool.name == "hotdata_search_text"
    assert set(tool.args) == {"query", "k"}
    assert tool.args["query"]["type"] == "string"


def test_search_tool_description_grounds_the_agent(mock_client: MagicMock) -> None:
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, key_column=None)
    expected = default_search_description(TABLE, COLUMN, columns=[COLUMN], max_k=100)
    assert tool.description == expected
    assert COLUMN in tool.description
    assert TABLE in tool.description


def test_search_description_steers_away_from_substring_matching_in_sql() -> None:
    """Saying only that LIKE "works" was observed to pull models into ILIKE '%word%'."""
    description = default_search_description(TABLE, COLUMN).lower()
    assert "like and ilike" in description
    assert "not a way to find relevant rows" in description
    assert "score" in description


def test_search_description_never_claims_sql_cannot_rank_text() -> None:
    """It can, via a table-valued function.

    This tool is registered alongside the SQL tool, so both descriptions reach the model in
    one prompt. The SQL description is pinned against the same claim by
    ``test_sql_description_never_claims_sql_cannot_rank_text``; asserting it in only one of
    them leaves the false sentence reachable through the other.
    """
    description = default_search_description(TABLE, COLUMN).lower()
    assert "cannot rank" not in description
    assert "only way" not in description


def test_search_description_sends_aggregates_to_sql_instead_of_literals() -> None:
    """Measured: carrying ids back capped a cohort at this tool's row limit, not at intent."""
    description = default_search_description(TABLE, COLUMN).lower()
    assert "aggregates over the matches" in description
    assert "rank inside sql instead" in description


def test_search_description_names_the_capability_not_the_index() -> None:
    """The contract has to outlive the retrieval strategy behind it."""
    description = default_search_description(TABLE, COLUMN).lower()
    for mechanism in ("bm25", "tantivy", "hnsw", "vector", "embedding"):
        assert mechanism not in description, f"description leaks the mechanism {mechanism!r}"


def test_search_tool_accepts_description_override(mock_client: MagicMock) -> None:
    tool = make_hotdata_search_tool(
        mock_client, table=TABLE, column=COLUMN, description="Search Airbnb blurbs."
    )
    assert tool.description == "Search Airbnb blurbs."


def test_search_tool_invocation_builds_ranked_sql(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    mock_client.execute_sql.return_value = search_result
    tool = make_hotdata_search_tool(
        mock_client, table=TABLE, column=COLUMN, columns=["id", "description"]
    )
    payload = json.loads(tool.invoke({"query": QUERY}))
    assert payload["rows"][0]["score"] == 8.5
    assert executed_sql(mock_client) == (
        "SELECT id, description, score "
        "FROM bm25_search('default.public.listings', 'description', 'cozy apartment', 5) "
        "ORDER BY score DESC "
        "LIMIT 5"
    )


def test_search_tool_uses_constructor_k_when_agent_omits_it(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    mock_client.execute_sql.return_value = search_result
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, k=3)
    tool.invoke({"query": QUERY})
    assert executed_sql(mock_client).endswith("LIMIT 3")


def test_search_tool_lets_agent_override_k(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    mock_client.execute_sql.return_value = search_result
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, k=3)
    tool.invoke({"query": QUERY, "k": 10})
    sql = executed_sql(mock_client)
    match = _SEARCH_CALL_RE.search(sql)
    assert match is not None
    assert match.group(1) == "10"
    assert sql.endswith("LIMIT 10")


def test_search_tool_clamps_a_model_supplied_k_to_max_rows(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    """Above max_rows the engine would rank rows that are discarded before the model sees them."""
    mock_client.execute_sql.return_value = search_result
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, max_rows=10)
    tool.invoke({"query": QUERY, "k": 100_000})
    sql = executed_sql(mock_client)
    match = _SEARCH_CALL_RE.search(sql)
    assert match is not None
    assert match.group(1) == "10"
    assert sql.endswith("LIMIT 10")


def test_search_tool_leaves_a_caller_supplied_k_alone(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    """The caller is trusted; only the model's k is clamped."""
    mock_client.execute_sql.return_value = search_result
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, k=50, max_rows=10)
    tool.invoke({"query": QUERY})
    assert executed_sql(mock_client).endswith("LIMIT 50")


def test_search_tool_keeps_a_clamped_k_positive(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    mock_client.execute_sql.return_value = search_result
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN)
    tool.invoke({"query": QUERY, "k": 0})
    assert executed_sql(mock_client).endswith("LIMIT 1")


def test_clamping_k_says_so_in_the_client_warning(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    """The clamp runs before the query, so nothing else in the result records it."""
    mock_client.execute_sql.return_value = search_result
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, max_rows=10)
    payload = json.loads(tool.invoke({"query": QUERY, "k": 200}))
    warning = payload["metadata"][CLIENT_WARNING_KEY]
    assert "k=200" in warning
    assert "10" in warning
    assert payload["metadata"]["warning"] is None


def test_an_unclamped_k_warns_about_nothing(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    mock_client.execute_sql.return_value = search_result
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, max_rows=10)
    payload = json.loads(tool.invoke({"query": QUERY, "k": 10}))
    assert CLIENT_WARNING_KEY not in payload["metadata"]


def test_a_caller_supplied_k_does_not_warn(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    """Nothing was clamped, so there is nothing to report."""
    mock_client.execute_sql.return_value = search_result
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, k=50, max_rows=10)
    payload = json.loads(tool.invoke({"query": QUERY}))
    assert CLIENT_WARNING_KEY not in payload["metadata"]


def test_search_tool_states_the_k_ceiling_to_the_model(mock_client: MagicMock) -> None:
    """ "Ask for more with k" was an invitation the tool did not honour."""
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, max_rows=25)
    assert "25" in (tool.description or "")


def test_search_tool_describes_its_k_argument(mock_client: MagicMock) -> None:
    """The argument schema is a second model-facing channel; k reached it unexplained."""
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN)
    assert "description" in tool.args["k"]
    assert "description" in tool.args["query"]


def test_search_tool_rejects_a_non_positive_max_rows(mock_client: MagicMock) -> None:
    with pytest.raises(ValueError, match="max_rows must be >= 1"):
        make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, max_rows=0)


# --- The join key in the default projection ---------------------------------------


def columns_result(names: list[str]) -> QueryResult:
    return QueryResult(
        columns=["column_name"],
        rows=[[name] for name in names],
        row_count=len(names),
        result_id="res_cols",
        query_run_id="run_cols",
        execution_time_ms=2,
        warning=None,
        error_message=None,
    )


def test_default_projection_carries_the_key_when_the_table_has_one(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    """A hit with no id joins back to nothing, which disables the whole claim."""
    mock_client.execute_sql.side_effect = [columns_result(["id"]), search_result]
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN)
    tool.invoke({"query": QUERY})
    assert executed_sql(mock_client).startswith("SELECT id, description, score")


def test_key_column_is_dropped_when_the_table_does_not_have_it(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    """It is looked up, not assumed, so a table without one is projected as before."""
    mock_client.execute_sql.side_effect = [columns_result([]), search_result]
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN)
    tool.invoke({"query": QUERY})
    assert executed_sql(mock_client).startswith("SELECT description, score")


def test_a_failed_key_lookup_still_builds_the_tool(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    """A schema query is never the reason tool construction fails."""
    mock_client.execute_sql.side_effect = [
        RuntimeError("information_schema is unavailable"),
        search_result,
    ]
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN)
    tool.invoke({"query": QUERY})
    assert executed_sql(mock_client).startswith("SELECT description, score")


def test_explicit_columns_are_left_exactly_as_given(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    """A caller naming the projection has already chosen; nothing is added to it."""
    mock_client.execute_sql.return_value = search_result
    tool = make_hotdata_search_tool(
        mock_client, table=TABLE, column=COLUMN, columns=["listing_id", COLUMN]
    )
    tool.invoke({"query": QUERY})
    assert executed_sql(mock_client).startswith("SELECT listing_id, description, score")


def test_key_column_can_be_turned_off(mock_client: MagicMock, search_result: QueryResult) -> None:
    mock_client.execute_sql.return_value = search_result
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, key_column=None)
    tool.invoke({"query": QUERY})
    assert executed_sql(mock_client).startswith("SELECT description, score")
    assert mock_client.execute_sql.call_count == 1


def test_key_column_lookup_is_scoped_to_the_searched_table(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    mock_client.execute_sql.side_effect = [columns_result(["id"]), search_result]
    make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, key_column="listing_id")
    lookup = mock_client.execute_sql.call_args_list[0].args[0]
    assert "table_catalog = 'default'" in lookup
    assert "table_schema = 'public'" in lookup
    assert "table_name = 'listings'" in lookup
    assert "column_name = 'listing_id'" in lookup


def test_description_names_the_columns_a_hit_carries(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    """Whether a result can be joined should be readable before calling the tool."""
    mock_client.execute_sql.side_effect = [columns_result(["id"]), search_result]
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN)
    assert "id and description" in (tool.description or "")


def test_make_hotdata_tools_passes_the_key_column_through(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    mock_client.execute_sql.side_effect = [columns_result(["id"]), search_result]
    tools = {
        tool.name: tool
        for tool in make_hotdata_tools(
            mock_client,
            search_table=TABLE,
            search_column=COLUMN,
            describe_tables=False,
            management_tools=False,
            catalog="default",
        )
    }
    tools["hotdata_search_text"].invoke({"query": QUERY})
    assert executed_sql(mock_client).startswith("SELECT id, description, score")


def test_search_tool_validates_corpus_at_construction(mock_client: MagicMock) -> None:
    with pytest.raises(ValueError, match=r"catalog\.schema\.table"):
        make_hotdata_search_tool(mock_client, table="listings", column=COLUMN)


def test_search_tool_validates_k_at_construction(mock_client: MagicMock) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, k=0)


def test_search_tool_validates_columns_at_construction(mock_client: MagicMock) -> None:
    with pytest.raises(ValueError, match="bare SQL identifier"):
        make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, columns=["bad col"])


# --- Wiring into make_hotdata_tools -----------------------------------------------


def tool_names(tools: list[Any]) -> set[str]:
    return {tool.name for tool in tools}


def test_make_hotdata_tools_omits_search_tool_by_default(mock_client: MagicMock) -> None:
    assert "hotdata_search_text" not in tool_names(make_hotdata_tools(mock_client))


def test_make_hotdata_tools_appends_search_tool_when_configured(mock_client: MagicMock) -> None:
    tools = make_hotdata_tools(mock_client, search_table=TABLE, search_column=COLUMN)
    assert tool_names(tools) == {
        "hotdata_execute_sql",
        "hotdata_list_managed_databases",
        "hotdata_create_managed_database",
        "hotdata_load_managed_table",
        "hotdata_describe_tables",
        "hotdata_search_text",
    }


def test_make_hotdata_tools_honours_custom_search_tool_name(mock_client: MagicMock) -> None:
    tools = make_hotdata_tools(
        mock_client,
        search_table=TABLE,
        search_column=COLUMN,
        search_tool_name="search_listings",
    )
    assert "search_listings" in tool_names(tools)


@pytest.mark.parametrize(
    ("table", "column"),
    [(TABLE, None), (None, COLUMN)],
)
def test_make_hotdata_tools_requires_both_search_arguments(
    mock_client: MagicMock, table: str | None, column: str | None
) -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        make_hotdata_tools(mock_client, search_table=table, search_column=column)


def test_make_hotdata_tools_shares_database_scope_with_search(
    mock_client: MagicMock, search_result: QueryResult, managed_db: ManagedDatabase
) -> None:
    mock_client.execute_sql.return_value = search_result
    tools = {
        tool.name: tool
        for tool in make_hotdata_tools(
            mock_client, database_id=managed_db, search_table=TABLE, search_column=COLUMN
        )
    }
    tools["hotdata_search_text"].invoke({"query": QUERY})
    assert mock_client.execute_sql.call_args.kwargs == {"database": managed_db}


def test_make_hotdata_tools_shares_max_rows_with_search(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    mock_client.execute_sql.return_value = search_result
    tools = {
        tool.name: tool
        for tool in make_hotdata_tools(
            mock_client, max_rows=1, search_table=TABLE, search_column=COLUMN
        )
    }
    payload = json.loads(tools["hotdata_search_text"].invoke({"query": QUERY}))
    assert len(payload["rows"]) == 1


def test_a_capped_search_result_does_not_advise_paging(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    """The envelope is shared, but this tool's caller supplies a query string, not SQL."""
    mock_client.execute_sql.return_value = search_result
    payload = json.loads(
        bm25_search_json(mock_client, table=TABLE, column=COLUMN, query=QUERY, max_rows=1)
    )
    warning = payload["metadata"][CLIENT_WARNING_KEY]
    assert "LIMIT/OFFSET" not in warning
    assert "bm25_search inside SQL" in warning


def test_a_caller_k_above_max_rows_still_warns_usefully(
    mock_client: MagicMock, search_result: QueryResult
) -> None:
    """A caller's k is trusted, so the engine ranks more rows than the envelope returns."""
    mock_client.execute_sql.return_value = search_result
    tool = make_hotdata_search_tool(mock_client, table=TABLE, column=COLUMN, k=50, max_rows=1)
    payload = json.loads(tool.invoke({"query": QUERY}))
    assert "LIMIT/OFFSET" not in payload["metadata"][CLIENT_WARNING_KEY]
