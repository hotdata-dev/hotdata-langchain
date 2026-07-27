from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from hotdata_framework import QueryResult


@pytest.fixture
def sample_result() -> QueryResult:
    return QueryResult(
        columns=["n"],
        rows=[[1], [2]],
        row_count=2,
        result_id="res_1",
        query_run_id="run_1",
        execution_time_ms=12,
        warning=None,
        error_message=None,
    )


@pytest.fixture
def search_result() -> QueryResult:
    return QueryResult(
        columns=["description", "score"],
        rows=[["Cozy apartment with a view", 8.5], ["Cozy studio, great light", 4.25]],
        row_count=2,
        result_id="res_bm25",
        query_run_id="run_bm25",
        execution_time_ms=8,
        warning=None,
        error_message=None,
    )


@pytest.fixture
def mock_client(sample_result: QueryResult):
    client = MagicMock()
    client.workspace_id = "ws_test"
    client.execute_sql = MagicMock(return_value=sample_result)
    client.list_managed_databases = MagicMock(return_value=[])
    return client
