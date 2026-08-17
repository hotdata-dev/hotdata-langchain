from __future__ import annotations

from hotdata_framework import QueryResult

from hotdata_langchain.results import (
    CLIENT_WARNING_KEY,
    result_payload,
    truncation_warning,
)


def result(row_count: int, rows: list[list[object]], warning: str | None = None) -> QueryResult:
    return QueryResult(
        columns=["n"],
        rows=rows,
        row_count=row_count,
        result_id="res",
        query_run_id="run",
        execution_time_ms=4,
        warning=warning,
        error_message=None,
    )


# --- The channel itself -----------------------------------------------------------


def test_no_client_warning_key_when_there_is_nothing_to_say() -> None:
    """The key's presence is itself a signal, so it is absent rather than null."""
    payload = result_payload(result(2, [[1], [2]]), max_rows=10)
    assert CLIENT_WARNING_KEY not in payload["metadata"]


def test_client_warning_is_separate_from_the_engine_warning() -> None:
    """``warning`` is the engine's field; writing ours into it would overwrite theirs."""
    payload = result_payload(
        result(500, [[1]], warning="engine says the source was stale"),
        max_rows=1,
    )
    metadata = payload["metadata"]
    assert metadata["warning"] == "engine says the source was stale"
    assert "stale" not in metadata[CLIENT_WARNING_KEY]


def test_client_warnings_are_joined_into_one_string() -> None:
    payload = result_payload(result(1, [[1]]), max_rows=10, warnings=["first.", "second."])
    assert payload["metadata"][CLIENT_WARNING_KEY] == "first. second."


def test_empty_warnings_are_dropped() -> None:
    payload = result_payload(result(1, [[1]]), max_rows=10, warnings=["", None])  # type: ignore[list-item]
    assert CLIENT_WARNING_KEY not in payload["metadata"]


# --- Truncation -------------------------------------------------------------------


def test_truncation_warning_states_both_numbers_and_where_to_resume() -> None:
    """An agent that inferred the gap itself guessed the boundary and re-read rows."""
    warning = truncation_warning(returned=100, matched=7535)
    assert warning is not None
    assert "100" in warning
    assert "7535" in warning
    assert "OFFSET" in warning


def test_no_truncation_warning_when_everything_matched_came_back() -> None:
    assert truncation_warning(returned=5, matched=5) is None


def test_capped_result_warns_through_the_envelope() -> None:
    payload = result_payload(result(7535, [[1], [2]]), max_rows=2)
    assert len(payload["rows"]) == 2
    assert "7535" in payload["metadata"][CLIENT_WARNING_KEY]


def test_caller_warnings_come_before_the_truncation_note() -> None:
    payload = result_payload(result(50, [[1]]), max_rows=1, warnings=["mine."])
    assert payload["metadata"][CLIENT_WARNING_KEY].startswith("mine.")


def test_the_remedy_is_the_callers_to_choose() -> None:
    """Shared envelope, different callers: one writes SQL, one supplies a search string."""
    payload = result_payload(result(50, [[1]]), max_rows=1, remedy="do something else")
    assert payload["metadata"][CLIENT_WARNING_KEY].endswith("do something else.")
