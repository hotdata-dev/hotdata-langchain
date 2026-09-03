"""Reading back when an instant database is due to be reaped.

A lifetime is written as a string, either an RFC 3339 timestamp or a relative window such
as ``"24h"``, and the server resolves it to an instant. So the resolved time is only ever
knowable by reading it back, and ``ManagedDatabase`` carries no ``expires_at`` to consult.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hotdata.exceptions import ApiException
from hotdata_framework import ManagedDatabase

from hotdata_langchain.databases import database_expiries, database_expiry

REAPED_AT = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


def summary(db_id: str, expires_at: datetime | None) -> SimpleNamespace:
    return SimpleNamespace(id=db_id, name=db_id, expires_at=expires_at)


def page(*summaries: SimpleNamespace, next_cursor: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        databases=list(summaries),
        next_cursor=next_cursor,
        has_more=next_cursor is not None,
        count=len(summaries),
        limit=100,
    )


@pytest.fixture
def api(managed_db: ManagedDatabase) -> Iterator[MagicMock]:
    with patch("hotdata_langchain.databases.DatabasesApi") as api:
        api.return_value.get_database.return_value = SimpleNamespace(
            id=managed_db.id,
            name=managed_db.description,
            default_connection_id=managed_db.default_connection_id,
            attachments=[],
            expires_at=None,
        )
        api.return_value.list_databases.return_value = page()
        yield api


# --- one database -------------------------------------------------------------------


def test_a_resolved_record_cannot_answer_this_which_is_why_the_helper_exists(
    managed_db: ManagedDatabase,
) -> None:
    assert not hasattr(managed_db, "expires_at")


def test_expiry_reports_the_instant_the_server_resolved(
    mock_client: MagicMock, managed_db: ManagedDatabase, api: MagicMock
) -> None:
    api.return_value.get_database.return_value = SimpleNamespace(
        id=managed_db.id,
        name=None,
        default_connection_id="c",
        attachments=[],
        expires_at=REAPED_AT,
    )
    assert database_expiry(mock_client, managed_db.id) == REAPED_AT


def test_a_database_with_no_ttl_reports_none(
    mock_client: MagicMock, managed_db: ManagedDatabase, api: MagicMock
) -> None:
    assert database_expiry(mock_client, managed_db.id) is None


def test_expiry_raises_keyerror_for_an_unknown_database(
    mock_client: MagicMock, api: MagicMock
) -> None:
    api.return_value.get_database.side_effect = ApiException(status=404, reason="Not Found")
    with pytest.raises(KeyError, match="no instant database"):
        database_expiry(mock_client, "dbid000000000000000000000000x")


# --- the whole workspace ------------------------------------------------------------


def test_expiries_come_from_the_listing_not_one_read_per_database(
    mock_client: MagicMock, api: MagicMock
) -> None:
    """The listing already carries expires_at, so a per-database read is waste."""
    api.return_value.list_databases.return_value = page(
        summary("db1", REAPED_AT), summary("db2", None)
    )

    assert database_expiries(mock_client) == {"db1": REAPED_AT, "db2": None}
    api.return_value.get_database.assert_not_called()


def test_expiries_follow_the_cursor_across_pages(mock_client: MagicMock, api: MagicMock) -> None:
    """One page read as the whole workspace would report a subset as if it were complete."""
    api.return_value.list_databases.side_effect = [
        page(summary("db1", REAPED_AT), next_cursor="c1"),
        page(summary("db2", None)),
    ]

    assert database_expiries(mock_client) == {"db1": REAPED_AT, "db2": None}
    assert api.return_value.list_databases.call_count == 2


def test_the_second_page_is_requested_with_the_cursor_it_was_given(
    mock_client: MagicMock, api: MagicMock
) -> None:
    api.return_value.list_databases.side_effect = [
        page(summary("db1", None), next_cursor="c1"),
        page(summary("db2", None)),
    ]

    database_expiries(mock_client)

    assert api.return_value.list_databases.call_args_list[1].kwargs == {"cursor": "c1"}


def test_a_cursor_pointing_at_an_empty_page_terminates(
    mock_client: MagicMock, api: MagicMock
) -> None:
    """A server that keeps handing back a cursor must not spin the loop forever."""
    api.return_value.list_databases.side_effect = [
        page(summary("db1", None), next_cursor="c1"),
        page(next_cursor="c2"),
    ]

    assert database_expiries(mock_client) == {"db1": None}


def test_an_empty_workspace_reports_nothing_rather_than_failing(
    mock_client: MagicMock, api: MagicMock
) -> None:
    assert database_expiries(mock_client) == {}


def test_a_listing_failure_surfaces_the_api_message(mock_client: MagicMock, api: MagicMock) -> None:
    api.return_value.list_databases.side_effect = ApiException(
        status=403, reason="Forbidden", body="workspace does not permit listing"
    )
    with pytest.raises(RuntimeError, match="workspace does not permit listing"):
        database_expiries(mock_client)
