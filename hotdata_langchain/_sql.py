"""Shared SQL identifier and literal handling.

These sit between model-authored strings and a SQL literal, so they live in one place
rather than being reimplemented per module where the two copies could drift apart.
"""

from __future__ import annotations

import re

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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
