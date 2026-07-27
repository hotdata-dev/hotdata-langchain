"""The tool descriptions are the contract the model plans against.

Each claim asserted here was verified against a live engine; a description that drifts
from the engine's real behaviour misleads the model silently, so the wording that
encodes a real constraint is pinned rather than left free.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from hotdata_langchain.tools import make_hotdata_tools, sql_tool_description

TABLE = "default.public.listings"
COLUMN = "description"


def descriptions(**kwargs: object) -> dict[str, str]:
    client = MagicMock()
    return {t.name: t.description or "" for t in make_hotdata_tools(client, **kwargs)}  # type: ignore[arg-type]


def test_every_tool_is_described() -> None:
    for name, description in descriptions().items():
        assert len(description) > 40, f"{name} has a description too thin to plan against"


def test_sql_description_states_the_dialect() -> None:
    assert "postgresql dialect" in sql_tool_description().lower()


def test_sql_description_warns_that_aggregates_need_a_column() -> None:
    """Verified live: ungrouped COUNT(*)/COUNT(1) are rejected, COUNT(<column>) works."""
    description = sql_tool_description()
    assert "COUNT(*)" in description
    assert "COUNT(<column>)" in description
    assert "GROUP BY" in description


def test_sql_description_points_at_the_search_tool_when_one_exists() -> None:
    """Verified live: an unguided agent reaches for to_tsvector, which the engine rejects."""
    description = descriptions(search_table=TABLE, search_column=COLUMN)["hotdata_execute_sql"]
    assert "hotdata_search_text" in description
    assert "LIKE" in description


def test_sql_description_frames_like_as_a_filter_not_a_search() -> None:
    """Saying only that LIKE "works" was observed to pull the model away from searching."""
    description = descriptions(search_table=TABLE, search_column=COLUMN)["hotdata_execute_sql"]
    assert "not a substitute for searching" in description
    # The relevance route must be stated before LIKE is mentioned at all.
    assert description.index("hotdata_search_text") < description.index("LIKE")


def test_sql_description_omits_the_search_tool_when_none_is_registered() -> None:
    description = descriptions()["hotdata_execute_sql"]
    assert "hotdata_search_text" not in description
    assert "LIKE" in description


def test_sql_description_follows_a_custom_search_tool_name() -> None:
    description = descriptions(
        search_table=TABLE, search_column=COLUMN, search_tool_name="search_listings"
    )["hotdata_execute_sql"]
    assert "search_listings" in description
    assert "hotdata_search_text" not in description


def test_database_descriptions_steer_towards_ids() -> None:
    """Database names are display labels and are not unique; ids are the safe handle."""
    described = descriptions()
    assert "id" in described["hotdata_list_managed_databases"]
    assert "not unique" in described["hotdata_list_managed_databases"]
    assert "database id" in described["hotdata_load_managed_table"].lower()


def test_load_description_states_the_accepted_input() -> None:
    """Verified: the tool takes a local parquet path, not a URL."""
    description = descriptions()["hotdata_load_managed_table"].lower()
    assert "parquet" in description
    assert "url" in description
