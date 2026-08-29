"""What a column is searchable by, grouped for the callers that report it.

The vocabulary is the capability, never the mechanism: these assert that the words which
reach a model say "text relevance" and "meaning" rather than "bm25" and "vector".
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hotdata_langchain.indexes import (
    CAPABILITY_PHRASES,
    SEARCH_NOUNS,
    SEMANTIC,
    TEXT,
    SearchableColumn,
    SearchIndex,
    capabilities_by_column,
    search_nouns_by_column,
    verify_searchable_columns,
)


def text_index(column: str = "description") -> SearchIndex:
    return SearchIndex(column=column, kind=TEXT, index_type="bm25", ready=True)


def vector(column: str = "embedding") -> SearchIndex:
    return SearchIndex(column=column, kind=SEMANTIC, index_type="vector", ready=True)


def test_the_phrases_are_built_from_the_nouns() -> None:
    """One source of truth, so a payload and a sentence cannot name a capability differently."""
    assert CAPABILITY_PHRASES.keys() == SEARCH_NOUNS.keys()
    for kind, noun in SEARCH_NOUNS.items():
        assert CAPABILITY_PHRASES[kind] == f"searchable by {noun}"


def test_no_capability_word_names_an_index_type() -> None:
    words = " ".join([*SEARCH_NOUNS.values(), *CAPABILITY_PHRASES.values()]).lower()
    for mechanism in ("bm25", "vector", "hnsw", "usearch", "index"):
        assert mechanism not in words


def test_capabilities_and_nouns_group_by_column() -> None:
    indexes = [text_index("description"), vector("embedding")]
    assert capabilities_by_column(indexes) == {
        "description": ["searchable by text relevance"],
        "embedding": ["searchable by meaning"],
    }
    assert search_nouns_by_column(indexes) == {
        "description": ["text relevance"],
        "embedding": ["meaning"],
    }


def test_two_indexes_of_one_kind_on_a_column_are_named_once() -> None:
    """The engine allows more than one index to land on a column; the capability is still one."""
    assert search_nouns_by_column([text_index(), text_index()]) == {
        "description": ["text relevance"]
    }


def test_both_kinds_on_one_column_are_both_named() -> None:
    assert search_nouns_by_column([text_index("body"), vector("body")]) == {
        "body": ["text relevance", "meaning"]
    }


def test_no_indexes_means_no_columns_rather_than_an_error() -> None:
    assert capabilities_by_column([]) == {}
    assert search_nouns_by_column([]) == {}


def _listing(**per_table: list[SimpleNamespace]) -> MagicMock:
    """Patch the indexes API so each table reports its own listing."""
    api = patch("hotdata_langchain.indexes.IndexesApi").start()
    api.return_value.list_indexes.side_effect = lambda _conn, _schema, table: SimpleNamespace(
        indexes=per_table.get(table, [])
    )
    return api


def _bm25(column: str = "description") -> SimpleNamespace:
    return SimpleNamespace(
        index_name=f"{column}_bm25",
        index_type="bm25",
        columns=[column],
        metric=None,
        status="ready",
        source_column=None,
    )


def test_a_declared_column_is_named_only_when_an_index_covers_it() -> None:
    """BM25 has no brute-force fallback, so naming an unindexed column is a hard error
    the model reaches only after committing to the route."""
    api = _listing(listings=[_bm25("description")], notes=[])
    try:
        found = verify_searchable_columns(
            MagicMock(),
            columns=[("d.public.listings", "description"), ("d.public.notes", "body")],
            database=MagicMock(),
        )
    finally:
        patch.stopall()
    assert [(one.table, one.column) for one in found] == [("d.public.listings", "description")]
    assert api.return_value.list_indexes.call_count == 2


def test_declared_order_is_preserved_and_one_table_is_listed_once() -> None:
    """Order carries into the description: the leading call is the one a model was
    measured reaching for most, so the caller's ordering must survive."""
    api = _listing(listings=[_bm25("description"), _bm25("name")])
    try:
        found = verify_searchable_columns(
            MagicMock(),
            columns=[
                ("d.public.listings", "name"),
                ("d.public.listings", "description"),
                ("d.public.listings", "name"),
            ],
            database=MagicMock(),
        )
    finally:
        patch.stopall()
    assert [one.column for one in found] == ["name", "description"]
    assert api.return_value.list_indexes.call_count == 1


def test_a_table_reference_that_is_not_three_parts_is_refused() -> None:
    """The engine's index-lookup rewrite matches on the reference as written, so a
    two-part form silently forfeits the index it was named to reach."""
    with pytest.raises(ValueError, match=r"catalog\.schema\.table"):
        verify_searchable_columns(
            MagicMock(), columns=[("public.listings", "description")], database=MagicMock()
        )


def test_nothing_is_named_without_a_database_to_confirm_against() -> None:
    assert verify_searchable_columns(MagicMock(), columns=[("d.p.t", "c")], database=None) == []


def test_only_a_column_the_engine_can_be_asked_in_sql_is_composable() -> None:
    """A plain vector index needs a query vector, which an agent writing SQL cannot
    produce; a provider-backed one takes text."""
    plain = SearchableColumn("d.p.t", vector("embedding"))
    backed = SearchableColumn(
        "d.p.t",
        SearchIndex(
            column="content",
            kind=SEMANTIC,
            index_type="vector",
            ready=True,
            embeds_query=True,
        ),
    )
    text = SearchableColumn("d.p.t", text_index())
    assert (plain.composable, backed.composable, text.composable) == (False, True, True)
    assert (plain.function, backed.function, text.function) == (
        "vector_search",
        "vector_search",
        "bm25_search",
    )
