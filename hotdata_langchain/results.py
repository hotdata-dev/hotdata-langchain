"""The JSON envelope every Hotdata tool returns, and its client-side warning channel.

One helper builds the envelope for both the SQL and the search paths, so an agent sees
the same shape and the same warning key whichever tool it called.

``metadata.warning`` is the engine's field: the SDK populates it from the query
response and this package only passes it through. Warnings raised here — a result
capped, a format pattern that will not do what it says — go in ``metadata.client_warning``
instead, so a consumer can tell which side noticed and neither source can overwrite the
other.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from hotdata_framework import QueryResult

#: Envelope key carrying warnings raised by this package rather than by the engine.
CLIENT_WARNING_KEY = "client_warning"


def truncation_warning(*, returned: int, matched: int) -> str | None:
    """Return the warning for a result capped below what the query matched, or ``None``.

    States where the cap fell rather than leaving it to be inferred: an agent that
    spotted the gap itself was measured guessing the boundary and re-reading rows it
    already had.
    """
    if returned >= matched:
        return None
    return (
        f"Returned the first {returned} rows of the {matched} this query matched. "
        f"row_count is the total before that cap, so the rows here are a prefix, not "
        f"the whole answer: aggregate in SQL, narrow the query, or page with "
        f"LIMIT/OFFSET starting at {returned}."
    )


def result_payload(
    result: QueryResult,
    *,
    max_rows: int,
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """Return the ``{"metadata": ..., "rows": [...]}`` envelope for one query result.

    ``warnings`` are client-side notes to join into ``metadata.client_warning``; the
    truncation warning is added here, since every path returning rows can hit the cap.
    The key is absent when there is nothing to say, so its presence is itself a signal.
    """
    rows = result.to_records(max_rows=max_rows)
    metadata = result.metadata_dict()
    notes = [note for note in warnings if note]
    capped = truncation_warning(returned=len(rows), matched=result.row_count)
    if capped is not None:
        notes.append(capped)
    if notes:
        metadata[CLIENT_WARNING_KEY] = " ".join(notes)
    return {"metadata": metadata, "rows": rows}


def result_json(
    result: QueryResult,
    *,
    max_rows: int,
    warnings: Sequence[str] = (),
) -> str:
    """Return :func:`result_payload` serialised the way the tools return it."""
    return json.dumps(result_payload(result, max_rows=max_rows, warnings=warnings), indent=2)
