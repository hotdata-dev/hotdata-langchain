"""Attaching a registered connection into an instant database's query scope.

A managed database cannot see another managed database — the platform refuses its
connection with "scoped to another database". A *registered* source attaches as designed,
which is the route across the boundary.

Both endpoints answer 204 with no body, so a refusal raises and there is no ambiguous
return to read. What the status cannot cover is a 204 that did not do the work, which is
what the read-back guards against and what most of these tests exercise.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hotdata.exceptions import ApiException
from hotdata_framework import ManagedDatabase

from hotdata_langchain.databases import (
    CatalogAttachment,
    attach_catalog,
    database_attachments,
    detach_catalog,
)

CONNECTION = "connpg000000000000000000000001"


def detail_with(managed_db: ManagedDatabase, *attachments: SimpleNamespace) -> SimpleNamespace:
    """A ``GET /databases/{id}`` response carrying the fields ``ManagedDatabase`` drops."""
    return SimpleNamespace(
        id=managed_db.id,
        name=managed_db.description,
        default_connection_id=managed_db.default_connection_id,
        default_catalog="default",
        default_schema="public",
        attachments=list(attachments),
    )


def attachment(connection_id: str = CONNECTION, alias: str | None = "warehouse") -> SimpleNamespace:
    return SimpleNamespace(connection_id=connection_id, alias=alias)


@pytest.fixture
def attached(managed_db: ManagedDatabase) -> Iterator[MagicMock]:
    """Patch the raw databases API, reporting nothing attached until a test says otherwise."""
    with patch("hotdata_langchain.databases.DatabasesApi") as api:
        api.return_value.get_database.return_value = detail_with(managed_db)
        yield api


# --- reading back what the resolved record drops -------------------------------------


def test_attachments_are_readable_even_though_the_resolved_record_drops_them(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    """``ManagedDatabase`` carries no attachments field at all, so this is the only route."""
    assert not hasattr(managed_db, "attachments")
    attached.return_value.get_database.return_value = detail_with(managed_db, attachment())

    assert database_attachments(mock_client, managed_db.id) == [
        CatalogAttachment(connection_id=CONNECTION, alias="warehouse")
    ]


def test_a_database_with_nothing_attached_reports_an_empty_list(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    assert database_attachments(mock_client, managed_db.id) == []


def test_attachments_accepts_an_already_resolved_record(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    database_attachments(mock_client, managed_db)
    attached.return_value.get_database.assert_called_once_with(managed_db.id)


def test_attachments_raises_keyerror_for_an_unknown_database(
    mock_client: MagicMock, attached: MagicMock
) -> None:
    attached.return_value.get_database.side_effect = ApiException(status=404, reason="Not Found")
    with pytest.raises(KeyError, match="not accepted here"):
        database_attachments(mock_client, "dbid000000000000000000000000x")


# --- attaching -----------------------------------------------------------------------


def test_attach_sends_the_connection_and_the_requested_alias(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    attached.return_value.get_database.return_value = detail_with(managed_db, attachment())

    attach_catalog(mock_client, managed_db.id, connection_id=CONNECTION, alias="warehouse")

    database_id, request = attached.return_value.attach_database_catalog.call_args.args
    assert database_id == managed_db.id
    assert request.connection_id == CONNECTION
    assert request.alias == "warehouse"


def test_attach_reports_the_alias_that_landed_not_the_one_requested(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    """Left unset the server names the catalog, and SQL has to address that name."""
    attached.return_value.get_database.return_value = detail_with(
        managed_db, attachment(alias="pg_main")
    )

    landed = attach_catalog(mock_client, managed_db.id, connection_id=CONNECTION)

    assert landed == CatalogAttachment(connection_id=CONNECTION, alias="pg_main")
    assert attached.return_value.attach_database_catalog.call_args.args[1].alias is None


def test_attach_raises_when_the_call_succeeds_but_nothing_was_attached(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    """A 204 says the call was accepted, not that it took effect."""
    attached.return_value.get_database.return_value = detail_with(managed_db)

    with pytest.raises(RuntimeError, match="reports it is not attached"):
        attach_catalog(mock_client, managed_db.id, connection_id=CONNECTION)


def test_attach_ignores_an_unrelated_connection_already_attached(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    attached.return_value.get_database.return_value = detail_with(
        managed_db, attachment(connection_id="connother00000000000000000001", alias="other")
    )

    with pytest.raises(RuntimeError, match="reports it is not attached"):
        attach_catalog(mock_client, managed_db.id, connection_id=CONNECTION)


def test_attach_surfaces_the_platforms_refusal_message(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    """Attaching a managed database's own connection is refused, and the reason matters."""
    attached.return_value.attach_database_catalog.side_effect = ApiException(
        status=400,
        reason="Bad Request",
        body="Connection 'conn123' is scoped to another database and cannot be attached here",
    )

    with pytest.raises(RuntimeError, match="scoped to another database"):
        attach_catalog(mock_client, managed_db.id, connection_id=CONNECTION)


def test_attach_accepts_an_already_resolved_record(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    attached.return_value.get_database.return_value = detail_with(managed_db, attachment())

    attach_catalog(mock_client, managed_db, connection_id=CONNECTION)

    assert attached.return_value.attach_database_catalog.call_args.args[0] == managed_db.id


# --- detaching -----------------------------------------------------------------------


def test_detach_passes_the_database_and_connection_ids(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    detach_catalog(mock_client, managed_db.id, connection_id=CONNECTION)
    attached.return_value.detach_database_catalog.assert_called_once_with(managed_db.id, CONNECTION)


def test_detach_translates_a_404_into_a_keyerror(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    attached.return_value.detach_database_catalog.side_effect = ApiException(
        status=404, reason="Not Found"
    )
    with pytest.raises(KeyError, match="attached to it"):
        detach_catalog(mock_client, managed_db.id, connection_id=CONNECTION)


def test_detach_surfaces_a_non_404_failure_as_a_runtimeerror(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    attached.return_value.detach_database_catalog.side_effect = ApiException(
        status=403, reason="Forbidden", body="workspace does not permit detaching"
    )
    with pytest.raises(RuntimeError, match="workspace does not permit detaching"):
        detach_catalog(mock_client, managed_db.id, connection_id=CONNECTION)


# --- confirmation, and what 204 does not tell you ------------------------------------


def test_attach_skips_the_read_back_when_confirmation_is_off(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    """The only GET here would be the confirmation, so its absence is the assertion."""
    landed = attach_catalog(
        mock_client, managed_db.id, connection_id=CONNECTION, alias="warehouse", confirm=False
    )

    assert landed == CatalogAttachment(connection_id=CONNECTION, alias="warehouse")
    attached.return_value.get_database.assert_not_called()


def test_attach_without_confirmation_reports_the_requested_alias_not_the_landed_one(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    """Unconfirmed, there is nothing to read the server's choice from."""
    attached.return_value.get_database.return_value = detail_with(
        managed_db, attachment(alias="pg_main")
    )

    landed = attach_catalog(mock_client, managed_db.id, connection_id=CONNECTION, confirm=False)

    assert landed.alias is None


def test_attach_treats_an_already_attached_connection_as_a_no_op(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    """Re-running a provisioning step should not fail on the attach it already did."""
    attached.return_value.attach_database_catalog.side_effect = ApiException(
        status=409, reason="Conflict", body="already attached"
    )
    attached.return_value.get_database.return_value = detail_with(
        managed_db, attachment(alias="warehouse")
    )

    landed = attach_catalog(mock_client, managed_db.id, connection_id=CONNECTION)

    assert landed == CatalogAttachment(connection_id=CONNECTION, alias="warehouse")


def test_a_conflict_that_is_not_an_existing_attachment_still_raises(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    """409 is resolved by reading, not assumed to mean 'already attached'."""
    attached.return_value.attach_database_catalog.side_effect = ApiException(
        status=409, reason="Conflict", body="alias 'warehouse' is already in use"
    )
    attached.return_value.get_database.return_value = detail_with(managed_db)

    with pytest.raises(RuntimeError, match="already in use"):
        attach_catalog(mock_client, managed_db.id, connection_id=CONNECTION)


def test_detach_raises_when_the_connection_is_still_attached_afterwards(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    """204 says the call was accepted, not that it took effect."""
    attached.return_value.get_database.return_value = detail_with(managed_db, attachment())

    with pytest.raises(RuntimeError, match="still reports it as attached"):
        detach_catalog(mock_client, managed_db.id, connection_id=CONNECTION)


def test_detach_confirms_by_reading_the_database_back(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    detach_catalog(mock_client, managed_db.id, connection_id=CONNECTION)
    attached.return_value.get_database.assert_called_once_with(managed_db.id)


def test_detach_skips_the_read_back_when_confirmation_is_off(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    attached.return_value.get_database.return_value = detail_with(managed_db, attachment())

    detach_catalog(mock_client, managed_db.id, connection_id=CONNECTION, confirm=False)

    attached.return_value.get_database.assert_not_called()


def test_detach_leaving_an_unrelated_attachment_in_place_succeeds(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    """Confirmation matches on the connection detached, not on the list being empty."""
    attached.return_value.get_database.return_value = detail_with(
        managed_db, attachment(connection_id="connother00000000000000000001", alias="other")
    )

    detach_catalog(mock_client, managed_db.id, connection_id=CONNECTION)


# --- the 404 contract, which attach and detach must state the same way ---------------


def test_attach_translates_a_404_into_a_keyerror(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    """The attach fails before the read-back, so this path never reaches _database_detail."""
    attached.return_value.attach_database_catalog.side_effect = ApiException(
        status=404, reason="Not Found"
    )

    with pytest.raises(KeyError) as excinfo:
        attach_catalog(mock_client, "dbid000000000000000000000000x", connection_id=CONNECTION)

    message = str(excinfo.value)
    assert "no instant database" in message
    assert CONNECTION in message


def test_attach_and_detach_agree_on_which_error_a_404_is(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    """A caller wrapping both in one try block should not need two except clauses."""
    attached.return_value.attach_database_catalog.side_effect = ApiException(
        status=404, reason="Not Found"
    )
    attached.return_value.detach_database_catalog.side_effect = ApiException(
        status=404, reason="Not Found"
    )

    for call in (attach_catalog, detach_catalog):
        with pytest.raises(KeyError):
            call(mock_client, managed_db.id, connection_id=CONNECTION)


def test_the_unlanded_attach_message_names_the_way_out(
    mock_client: MagicMock, managed_db: ManagedDatabase, attached: MagicMock
) -> None:
    """Error text in this module carries its own remedy."""
    attached.return_value.get_database.return_value = detail_with(managed_db)

    with pytest.raises(RuntimeError, match="confirm=False"):
        attach_catalog(mock_client, managed_db.id, connection_id=CONNECTION)
