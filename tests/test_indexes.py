"""What a column is searchable by, grouped for the callers that report it.

The vocabulary is the capability, never the mechanism: these assert that the words which
reach a model say "text relevance" and "meaning" rather than "bm25" and "vector".
"""

from __future__ import annotations

from hotdata_langchain.indexes import (
    CAPABILITY_PHRASES,
    SEARCH_NOUNS,
    SEMANTIC,
    TEXT,
    SearchIndex,
    capabilities_by_column,
    search_nouns_by_column,
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
