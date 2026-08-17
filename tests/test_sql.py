from __future__ import annotations

import pytest

from hotdata_langchain._sql import format_pattern_warnings

# The measured failure: three queries returned per-day counts whose 'day' column was the
# literal text 'YYYY-MM-DD' on every row, and the agent answered "Day 1, Day 2, Day 3".
POSTGRES_PATTERN_SQL = (
    "SELECT to_char(cast(start_time AS DATE), 'YYYY-MM-DD') AS day, COUNT(span_id) "
    "FROM default.public.spans GROUP BY day"
)


def test_postgres_template_is_flagged() -> None:
    (warning,) = format_pattern_warnings(POSTGRES_PATTERN_SQL)
    assert "'YYYY-MM-DD'" in warning
    assert "strftime" in warning


def test_translatable_pattern_carries_the_rewrite() -> None:
    (warning,) = format_pattern_warnings(POSTGRES_PATTERN_SQL)
    assert "'%Y-%m-%d'" in warning


def test_untranslatable_pattern_warns_without_guessing_a_rewrite() -> None:
    """A partial translation would read as a correction while still being wrong."""
    (warning,) = format_pattern_warnings("SELECT to_char(d, 'Quarter Q') FROM t")
    assert "Write" not in warning


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT to_char(d, '%Y-%m-%d') FROM t",
        "SELECT to_date('2026-08-12') FROM t",
        "SELECT to_timestamp('2026-08-12T00:00:00Z') FROM t",
        "SELECT to_char(d, '-') FROM t",
        "SELECT name FROM t WHERE name = 'to_char(x, ''YYYY'')'",
        "SELECT id FROM t",
    ],
)
def test_no_warning(sql: str) -> None:
    """strftime patterns, parsed values, separators, and literals are all left alone."""
    assert format_pattern_warnings(sql) == []


def test_first_argument_is_never_read_as_a_pattern() -> None:
    """``to_date``'s first argument is the value being parsed, not a format."""
    assert format_pattern_warnings("SELECT to_date('2026-08-12', '%Y-%m-%d') FROM t") == []


def test_each_offending_pattern_is_reported_once() -> None:
    sql = "SELECT to_char(a, 'YYYY-MM'), to_char(b, 'YYYY-MM'), to_char(c, 'DD') FROM t"
    warnings = format_pattern_warnings(sql)
    assert len(warnings) == 2


def test_pattern_nested_inside_another_call_is_not_read_as_this_call_s() -> None:
    """Only the call's own arguments are inspected, never a literal one level down."""
    assert format_pattern_warnings("SELECT to_char(coalesce(d, cast('YYYY' AS DATE))) FROM t") == []
