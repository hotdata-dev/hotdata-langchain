"""Shared SQL identifier and literal handling.

These sit between model-authored strings and a SQL literal, so they live in one place
rather than being reimplemented per module where the two copies could drift apart.
"""

from __future__ import annotations

import re

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def is_identifier(value: str) -> bool:
    """Return whether ``value`` can be written into SQL as a bare identifier."""
    return IDENTIFIER_RE.fullmatch(value) is not None


def validate_identifier(value: str, *, label: str) -> str:
    """Return ``value`` if it is a bare SQL identifier, else raise ``ValueError``."""
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{label} must be a bare SQL identifier, got {value!r}")
    return value


def quote_literal(value: str) -> str:
    """Return ``value`` as a single-quoted SQL string literal, with quotes doubled."""
    if "\x00" in value:
        raise ValueError("SQL string literals may not contain null bytes")
    return "'" + value.replace("'", "''") + "'"


#: Functions whose non-first string arguments are date/time format patterns.
FORMAT_FUNCTIONS = ("to_char", "to_date", "to_timestamp", "date_format")

_FORMAT_CALL_RE = re.compile(rf"\b({'|'.join(FORMAT_FUNCTIONS)})\s*\(", re.IGNORECASE)

#: PostgreSQL template tokens, longest first, mapped to their strftime equivalent.
_TEMPLATE_TOKENS = (
    ("YYYY", "%Y"),
    ("MONTH", "%B"),
    ("HH24", "%H"),
    ("MON", "%b"),
    ("DAY", "%A"),
    ("DY", "%a"),
    ("MM", "%m"),
    ("DD", "%d"),
    ("HH", "%I"),
    ("MI", "%M"),
    ("SS", "%S"),
    ("YY", "%y"),
)

_SEPARATORS = set(" -/:.,T_")


def _strftime_equivalent(pattern: str) -> str | None:
    """Return ``pattern`` rewritten as strftime, or ``None`` if any part is unrecognised.

    Only offers a rewrite it can make in full: a partial translation would read as a
    correction while still being wrong.
    """
    out: list[str] = []
    index = 0
    upper = pattern.upper()
    while index < len(pattern):
        for token, replacement in _TEMPLATE_TOKENS:
            if upper.startswith(token, index):
                out.append(replacement)
                index += len(token)
                break
        else:
            if pattern[index] not in _SEPARATORS:
                return None
            out.append(pattern[index])
            index += 1
    return "".join(out)


def _format_arguments(sql: str, start: int) -> list[str]:
    """Return the string literals passed after the first argument of a call at ``start``.

    ``start`` is the index of the call's opening parenthesis. Only literals that are
    arguments of that call are returned, never ones nested inside another call, and never
    the first argument — for ``to_date`` and ``to_timestamp`` that one is the value being
    parsed rather than a pattern.
    """
    literals: list[str] = []
    depth = 0
    argument = 0
    index = start
    while index < len(sql):
        char = sql[index]
        if char == "'":
            end = index + 1
            while end < len(sql):
                if sql[end] == "'":
                    if end + 1 < len(sql) and sql[end + 1] == "'":
                        end += 2
                        continue
                    break
                end += 1
            if depth == 1 and argument > 0:
                literals.append(sql[index + 1 : end].replace("''", "'"))
            index = end + 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                break
        elif char == "," and depth == 1:
            argument += 1
        index += 1
    return literals


def _mask_literals(sql: str) -> str:
    """Return ``sql`` with the contents of string literals blanked, length preserved.

    Call detection runs over the masked text so a function name appearing inside a string
    literal is not read as a call; extraction still runs over the original, which the
    equal length keeps in step.
    """
    out = list(sql)
    index = 0
    while index < len(sql):
        if sql[index] != "'":
            index += 1
            continue
        end = index + 1
        while end < len(sql):
            if sql[end] == "'":
                if end + 1 < len(sql) and sql[end + 1] == "'":
                    end += 2
                    continue
                break
            end += 1
        for position in range(index + 1, min(end, len(sql))):
            out[position] = " "
        index = end + 1
    return "".join(out)


def format_pattern_warnings(sql: str) -> list[str]:
    """Return warnings for date/time format patterns that will not be interpreted.

    The engine's format patterns are strftime, so a PostgreSQL template such as
    ``'YYYY-MM-DD'`` is not a pattern at all: ``to_char`` returns the template text
    itself on every row and ``to_date`` rejects it. A pattern containing no ``%`` is the
    signal, which is cheap, engine-independent, and catches a hand-written query as
    readily as a generated one.

    Literals with no letters are left alone, so a separator-only argument does not warn.

    The rule is spelled out once. One query passing the same template to ``to_date`` and
    ``to_char`` is the ordinary shape of this mistake, and repeating the explanation per
    pattern doubles the text the model has to read to find the two names in it.
    """
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    for match in _FORMAT_CALL_RE.finditer(_mask_literals(sql)):
        function = match.group(1)
        for literal in _format_arguments(sql, match.end() - 1):
            if "%" in literal or not any(char.isalpha() for char in literal):
                continue
            if (function.lower(), literal) in seen:
                continue
            seen.add((function.lower(), literal))
            suggestion = _strftime_equivalent(literal)
            fix = f" Write '{suggestion}' instead." if suggestion else ""
            if warnings:
                warnings.append(f"{function} was given '{literal}', with the same problem.{fix}")
                continue
            warnings.append(
                f"{function} was given the format pattern '{literal}', which contains no "
                f"'%'. Format patterns here are strftime, not PostgreSQL templates, so a "
                f"pattern like this is not interpreted: to_char returns the pattern text "
                f"itself on every row instead of a formatted value, and to_date rejects "
                f"it.{fix}"
            )
    return warnings
