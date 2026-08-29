"""The tool descriptions are the contract the model plans against.

Each claim asserted here was verified against a live engine; a description that drifts
from the engine's real behaviour misleads the model silently, so the wording that
encodes a real constraint is pinned rather than left free.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hotdata_framework import ManagedDatabase
from langchain_core.tools import StructuredTool

from hotdata_langchain.indexes import SEMANTIC, SearchableColumn, SearchIndex
from hotdata_langchain.search import SearchRoute
from hotdata_langchain.tools import make_hotdata_tools, sql_tool_description

TABLE = "default.public.listings"
COLUMN = "description"


def descriptions(**kwargs: object) -> dict[str, str]:
    client = MagicMock()
    return {t.name: t.description or "" for t in make_hotdata_tools(client, **kwargs)}  # type: ignore[arg-type]


def test_every_tool_is_described() -> None:
    for name, description in descriptions().items():
        assert len(description) > 40, f"{name} has a description too thin to plan against"


def test_sql_description_names_datafusion_as_the_dialect() -> None:
    """Calling it PostgreSQL reinforced the prior behind the one silent-wrong-value failure.

    Verified: ``SELECT version()`` reports ``Apache DataFusion 54.0.0``. Naming the engine
    gives a prior that covers divergences not yet measured; PostgreSQL plus a list of
    exceptions only covers the ones already found.
    """
    description = sql_tool_description()
    assert "DataFusion" in description
    assert "PostgreSQL" in description
    assert "postgresql dialect" not in description.lower()


def test_sql_description_does_not_assert_the_format_echo_as_a_rule() -> None:
    """The echo is a defect the engine may fix (#37); asserting it would then be false.

    Behaviour being corrected upstream is stated as an absence of a guarantee, so the
    wording survives the fix.
    """
    description = sql_tool_description()
    assert "does not reliably raise" in description
    assert "returned verbatim as text on every row" not in description


def test_sql_description_prefers_counting_a_named_column() -> None:
    """Verified live: some tables reject COUNT(*), COUNT(1) and even SELECT 1.

    It is not an aggregate rule and not universal — ``SELECT 1 FROM t LIMIT 1`` fails on
    the same tables, while COUNT(*) succeeds on most. Naming a column always works.
    """
    description = sql_tool_description()
    assert "COUNT(*)" in description
    assert "COUNT(<column>)" in description
    assert "Some tables reject" in description


def test_sql_description_does_not_claim_ungrouped_counts_are_always_rejected() -> None:
    """Verified live: COUNT(*) succeeds on most tables, including a 6M-row one."""
    description = sql_tool_description().lower()
    assert "are rejected on their own" not in description
    assert "must reference at least one column" not in description


def test_sql_description_asks_for_fully_qualified_table_references() -> None:
    """A two-part reference resolves but can forfeit an index, silently and with no warning."""
    description = sql_tool_description()
    assert "catalog.schema.table" in description
    assert "all three parts" in description


def test_sql_description_names_the_catalog_when_it_is_known() -> None:
    """The builder resolves the database, so it can state the catalog rather than a rule."""
    assert "the catalog is 'default'" in sql_tool_description(catalogs=["default"])
    assert "the catalog is 'f1'" in sql_tool_description(catalogs=["f1"])


def test_sql_description_never_claims_the_catalog_is_always_default() -> None:
    """Verified live: an attached source's tables answer to 'f1', not 'default'.

    The database record reports ``default_catalog = 'default'`` for both kinds, so the old
    unconditional claim failed every query against an attached database.
    """
    for description in (sql_tool_description(), sql_tool_description(catalogs=["f1"])):
        assert "always 'default'" not in description


def test_sql_description_falls_back_to_information_schema_for_the_catalog() -> None:
    description = sql_tool_description()
    assert "table_catalog from information_schema.tables" in description


def test_sql_description_lists_every_catalog_when_a_database_has_several() -> None:
    description = sql_tool_description(catalogs=["default", "f1"])
    assert "default, f1" in description
    assert "table_catalog from information_schema.tables" in description


def test_sql_description_states_the_date_dialect() -> None:
    """Verified live: to_char with a PostgreSQL pattern returns the pattern, and never raises."""
    description = sql_tool_description()
    assert "strftime" in description
    assert "'%Y-%m-%d'" in description
    assert "'YYYY-MM-DD'" in description
    assert "date_sub" in description
    assert "INTERVAL '6 days'" in description


def test_sql_description_warns_that_a_bad_date_pattern_does_not_raise() -> None:
    """The silence is the whole danger: the run continues on destroyed values."""
    description = sql_tool_description()
    assert "never assume a bad pattern will announce itself" in description


def test_sql_description_warns_against_quoting_identifiers_for_case() -> None:
    """Verified live: r."driverId" fails while r.driverId resolves; storage is lowercased."""
    description = sql_tool_description()
    assert "lowercased" in description
    assert '"driverId"' in description


def test_sql_description_does_not_call_the_short_table_form_invalid() -> None:
    """It resolves correctly; only the acceleration is at stake, and that is being fixed."""
    description = sql_tool_description()
    assert "resolves to the same rows" in description
    assert "invalid" not in description.lower()


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


def test_sql_description_names_bm25_search_as_callable_from_sql() -> None:
    """Verified live: bm25_search is a TVF whose results join and aggregate like a table.

    Without this an agent calls the search tool and pastes the returned ids back as SQL
    literals, which caps the cohort at the tool's row limit.
    """
    description = descriptions(search_table=TABLE, search_column=COLUMN)["hotdata_execute_sql"]
    assert "bm25_search" in description
    assert "table-valued function" in description


def test_sql_description_names_the_indexed_table_and_column_when_known() -> None:
    """BM25 has no brute-force fallback, so the model must not guess at the column."""
    description = descriptions(search_table=TABLE, search_column=COLUMN)["hotdata_execute_sql"]
    assert f"bm25_search('{TABLE}', '{COLUMN}'" in description


def test_sql_description_prefers_the_composed_form_for_aggregates() -> None:
    description = descriptions(search_table=TABLE, search_column=COLUMN)["hotdata_execute_sql"]
    assert "aggregates over the matches" in description
    # The composable route must be stated before the row-returning tool.
    assert description.index("bm25_search") < description.index("hotdata_search_text")


def test_sql_description_never_claims_sql_cannot_rank_text() -> None:
    """It can, via bm25_search — the old wording was false and steered agents away."""
    for description in (
        sql_tool_description(),
        descriptions(search_table=TABLE, search_column=COLUMN)["hotdata_execute_sql"],
    ):
        assert "cannot rank" not in description.lower()


def test_sql_description_offers_bm25_search_without_a_search_tool() -> None:
    description = descriptions()["hotdata_execute_sql"]
    assert "bm25_search" in description


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


def test_catalog_argument_reaches_the_sql_description() -> None:
    description = descriptions(catalog="f1")["hotdata_execute_sql"]
    assert "the catalog is 'f1'" in description


def test_unscoped_tools_never_probe_for_a_catalog() -> None:
    """No database means no scope to read information_schema in, so no query is issued."""
    client = MagicMock()
    make_hotdata_tools(client)
    client.execute_sql.assert_not_called()


def test_load_description_states_both_accepted_inputs() -> None:
    """A deployed agent has no filesystem to write to, so the URL form is the reachable one."""
    description = descriptions()["hotdata_load_managed_table"].lower()
    assert "parquet" in description
    assert "local filesystem" in description
    assert "https:// url" in description


def test_load_description_no_longer_rules_out_urls() -> None:
    """The old wording was true when written and is now the opposite of the behaviour."""
    description = descriptions()["hotdata_load_managed_table"].lower()
    assert "not urls" not in description
    assert "only local parquet paths" not in description


def test_load_description_states_that_a_url_must_be_public() -> None:
    """Otherwise the model spends a turn discovering the rule from a rejection."""
    description = descriptions()["hotdata_load_managed_table"]
    assert "not an internal address" in description


def test_load_description_drops_the_public_url_rule_when_it_does_not_apply() -> None:
    """A deployment loading from an internal store would be told the opposite of the truth."""
    description = descriptions(allow_private_hosts=True)["hotdata_load_managed_table"]
    assert "not an internal address" not in description


# --- the SQL description follows the retrieval route ------------------------------


def _semantic_route(*, embeds_query: bool = True) -> SearchRoute:
    return SearchRoute(
        SEMANTIC,
        SearchIndex(
            column=COLUMN,
            kind=SEMANTIC,
            index_type="vector",
            ready=True,
            metric="cosine",
            vector_column=f"{COLUMN}_embedding",
            embeds_query=embeds_query,
        ),
    )


def test_sql_description_never_names_bm25_on_a_semantic_column() -> None:
    """Both descriptions reach the model in one prompt and must agree.

    Naming bm25_search beside a tool that ranks by meaning tells the model to call a
    function that has no index on the column it was just handed.
    """
    description = sql_tool_description(
        "hotdata_search_semantic",
        search_table=TABLE,
        search_column=COLUMN,
        search_route=_semantic_route(),
    )
    assert "bm25_search" not in description
    assert f"vector_search('{TABLE}', '{COLUMN}'" in description


def test_sql_description_carries_the_sort_semantic_search_needs() -> None:
    """vector_search returns rows unsorted; a trailing LIMIT then takes arbitrary ones."""
    description = sql_tool_description(
        "hotdata_search_semantic",
        search_table=TABLE,
        search_column=COLUMN,
        search_route=_semantic_route(),
    )
    assert "ORDER BY _distance ASC" in description


def test_sql_description_offers_no_composed_form_it_cannot_write() -> None:
    """A plain vector index needs a query vector, which SQL cannot express."""
    description = sql_tool_description(
        "hotdata_search_semantic",
        search_table=TABLE,
        search_column=COLUMN,
        search_route=_semantic_route(embeds_query=False),
    )
    assert "vector_search(" not in description
    # Scoped to the registered column. "not available in SQL here" would be a claim about
    # the database inferred from one tool's registration, which is the defect this
    # function was corrected for — another table may carry a composable index.
    assert f"Ranking '{COLUMN}' on {TABLE} by meaning needs the query as a vector" in description
    assert "not available in SQL here" not in description


def test_sql_description_keeps_bm25_wording_without_a_route() -> None:
    """The text route is the default, and nothing about it changed."""
    description = sql_tool_description(
        "hotdata_search_text", search_table=TABLE, search_column=COLUMN
    )
    assert f"bm25_search('{TABLE}', '{COLUMN}'" in description
    assert "vector_search" not in description


def _tool_builds() -> list[tuple[str, dict[str, object]]]:
    """The configurations whose descriptions differ, not just the default one.

    The search variants matter most: the tool name woven into the SQL description is
    computed from the resolved route, so it is the one cross-reference that is not a
    literal and the one a rename is most likely to leave behind.
    """
    return [
        ("default", {"management_tools": True}),
        ("no describe tool", {"management_tools": True, "describe_tables": False}),
        ("text search", {"search_table": TABLE, "search_column": COLUMN}),
        (
            "semantic search",
            {"search_table": TABLE, "search_column": COLUMN, "search_strategy": "semantic"},
        ),
        (
            "suffixed names",
            {
                "management_tools": True,
                "search_table": TABLE,
                "search_column": COLUMN,
                "tool_name_suffix": "sales",
            },
        ),
        (
            "suffix with a digit",
            {
                "management_tools": True,
                "search_table": TABLE,
                "search_column": COLUMN,
                "tool_name_suffix": "f1",
            },
        ),
        (
            "suffix with an uppercase letter",
            {
                "management_tools": True,
                "search_table": TABLE,
                "search_column": COLUMN,
                "tool_name_suffix": "Sales",
            },
        ),
        (
            "suffixed semantic search",
            {
                "search_table": TABLE,
                "search_column": COLUMN,
                "search_strategy": "semantic",
                "tool_name_suffix": "sales",
            },
        ),
    ]


def _model_facing(tool: object) -> str:
    """Everything of a tool that reaches the model: its description and its arg schema."""
    description = getattr(tool, "description", "") or ""
    args = getattr(tool, "args", None) or {}
    return description + " " + " ".join(str(a.get("description", "")) for a in args.values())


def _built(name: str, kwargs: dict[str, object]) -> list[StructuredTool]:
    """Build one tool set, resolving a semantic route through introspection.

    `make_hotdata_tools` takes no route: it reads one from the control plane, which is the
    behaviour under test — the tool name woven into the SQL description follows whatever
    that resolves to. So the semantic build patches the index listing rather than handing
    the answer in.
    """
    client = MagicMock()
    if "semantic" not in name:
        return make_hotdata_tools(client, **kwargs)  # type: ignore[arg-type]

    database = ManagedDatabase(
        id="dbidsf000000000000000000000001",
        description="sf_airbnb",
        default_connection_id="connsf00000000000000000000001",
    )
    listed = SimpleNamespace(
        indexes=[
            SimpleNamespace(
                index_name="listings_description_vector",
                index_type="vector",
                columns=["description_embedding"],
                metric="cosine",
                status="ready",
                source_column=COLUMN,
            )
        ]
    )
    with patch("hotdata_langchain.indexes.IndexesApi") as indexes:
        indexes.return_value.list_indexes.return_value = listed
        return make_hotdata_tools(client, database_id=database, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("label,kwargs", _tool_builds())
def test_the_tools_use_one_name_for_a_database(label: str, kwargs: dict[str, object]) -> None:
    """Every description reaches the model in one prompt, so two names for one thing is a
    contradiction it has to resolve. The rename to 'instant database' missed a string split
    across two source lines, which left the load tool describing what the other two called
    something else. Case-insensitive, and over the arg schema too: a sentence-initial
    'Managed database' is the same defect, and an argument description is read just as the
    tool description is."""
    said = {
        # Not lowercased: the pattern is a literal, so folding the case would throw away
        # the one thing the failure has to show — how it was actually written.
        tool.name: sorted(set(re.findall(r"managed database", _model_facing(tool), re.I)))
        for tool in _built(label, kwargs)
    }
    mixed = {name: terms for name, terms in said.items() if terms}
    assert not mixed, (
        f"[{label}] these call a database 'managed' while others say 'instant': {mixed}"
    )


@pytest.mark.parametrize("label,kwargs", _tool_builds())
def test_no_description_points_at_a_tool_that_is_not_registered(
    label: str, kwargs: dict[str, object]
) -> None:
    """A description naming another tool is an instruction to call it, so a name that has
    drifted sends the model somewhere that does not exist. These cross-references are what
    make renaming a tool breaking rather than cosmetic, and a rename is coming.

    Parametrised because the reference woven into the SQL description is *computed* from the
    resolved search route rather than written as a literal, so the default build — which
    registers no search tool — never exercises the case most likely to break."""
    tools = _built(label, kwargs)
    registered = {tool.name for tool in tools}
    dangling = {
        (tool.name, ref)
        for tool in tools
        # The class has to be everything TOOL_NAME_PATTERN permits. A narrower one
        # truncates a suffixed name at the first character it excludes and reports the
        # prefix as dangling, failing a correct build: `[a-z_]+` did that to
        # `hotdata_describe_tables_f1`, and `[a-z0-9_-]+` would to `..._Sales`.
        for ref in re.findall(r"hotdata_[A-Za-z0-9_-]+", _model_facing(tool))
        if ref not in registered
    }
    assert not dangling, f"[{label}] descriptions name unregistered tools: {sorted(dangling)}"


@pytest.mark.parametrize("label,kwargs", _tool_builds())
def test_the_model_is_never_shown_the_phrase_managed_table(
    label: str, kwargs: dict[str, object]
) -> None:
    """`managed table` survives in the Python API and in prose written for people, and a
    table inside an instant database arguably wants a new name too — but that is a product
    decision. Until it is made, the term stays out of what a model reads, so the prompt
    carries one vocabulary rather than two."""
    leaked = {
        tool.name
        for tool in _built(label, kwargs)
        if "managed table" in _model_facing(tool).lower()
    }
    assert not leaked, f"[{label}] these show the model 'managed table': {sorted(leaked)}"


def test_the_database_tools_quote_only_keys_their_payload_carries() -> None:
    """A description naming a key the JSON lacks sends the model hunting for it.

    Both tools return a fixed shape, so every quoted lower-case token in their wording
    should be a key that shape actually has. The framework record still calls a database's
    name ``description``; these tools emit ``name``, and the wording has to follow.
    """
    quoted = re.compile(r"'([a-z_]+)'")
    client = MagicMock()
    client.list_managed_databases.return_value = [
        ManagedDatabase(id="dbid1", description="sales", default_connection_id="conn1")
    ]
    client.create_managed_database.return_value = ManagedDatabase(
        id="dbid1", description="sales", default_connection_id="conn1"
    )
    tools = {t.name: t for t in make_hotdata_tools(client)}

    payloads = {
        "hotdata_list_managed_databases": json.loads(
            tools["hotdata_list_managed_databases"].invoke({})
        )[0],
        "hotdata_create_managed_database": json.loads(
            tools["hotdata_create_managed_database"].invoke({"name": "sales"})
        ),
    }
    for name, payload in payloads.items():
        described = set(quoted.findall(tools[name].description or ""))
        missing = described - set(payload) - set(tools[name].args)
        assert not missing, f"{name} quotes {sorted(missing)}, but returns {sorted(payload)}"


def _searchable(table: str, column: str, kind: str = "text") -> SearchableColumn:
    return SearchableColumn(
        table,
        SearchIndex(
            column=column,
            kind=kind,  # type: ignore[arg-type]
            index_type="bm25" if kind == "text" else "vector",
            ready=True,
            embeds_query=kind == SEMANTIC,
        ),
    )


def test_sql_description_does_not_claim_the_registered_column_is_the_only_indexed_one() -> None:
    """`search_table` is what the search tool ranks over, which is not a statement about
    what else the database indexes. The exclusive claim was measured being followed in
    preference to what hotdata_describe_tables reports, which makes it a wrong answer the
    model cannot recover from."""
    description = descriptions(search_table=TABLE, search_column=COLUMN)["hotdata_execute_sql"]
    assert f"'{COLUMN}' on {TABLE} has a BM25 index" in description
    assert "the BM25-indexed column is" not in description


def test_each_searchable_column_gets_its_own_worked_call() -> None:
    """A model was measured writing whichever call it is shown and ignoring columns it is
    merely told about: naming a second column moved the table it searched in 2 runs of 12,
    giving that column its own call moved it in 8."""
    description = sql_tool_description(
        search_table=TABLE,
        search_column=COLUMN,
        also_searchable=[_searchable("default.public.corpus", "content")],
    )
    assert "bm25_search('default.public.corpus', 'content'" in description
    assert f"bm25_search('{TABLE}', '{COLUMN}'" in description


def test_the_first_declared_column_leads_the_examples() -> None:
    """Which column a model picks tracks which call it is shown first, so the caller's
    ordering is what decides it and has to survive into the text."""
    description = sql_tool_description(
        search_table=TABLE,
        search_column=COLUMN,
        also_searchable=[
            _searchable("default.public.corpus", "content"),
            _searchable("default.public.reviews", "body"),
        ],
    )
    order = [
        description.index(f"'{t}'")
        for t in ("default.public.corpus", "default.public.reviews", TABLE)
    ]
    assert order == sorted(order)


def test_a_column_named_twice_is_worked_once() -> None:
    description = sql_tool_description(
        search_table=TABLE, search_column=COLUMN, also_searchable=[_searchable(TABLE, COLUMN)]
    )
    assert description.count(f"bm25_search('{TABLE}', '{COLUMN}'") == 1


def test_a_semantic_column_is_never_offered_to_bm25_search() -> None:
    """The two functions read different indexes, and BM25 has no fallback: offering a
    vector column to bm25_search is a hard error at the point the model has committed."""
    description = sql_tool_description(
        search_table=TABLE,
        search_column=COLUMN,
        also_searchable=[_searchable("default.public.corpus", "embedding", SEMANTIC)],
    )
    assert "bm25_search('default.public.corpus'" not in description


def test_a_plain_vector_column_is_not_offered_as_composable() -> None:
    """Composing one needs a query vector, which an agent writing SQL cannot produce."""
    plain = SearchableColumn(
        "default.public.corpus",
        SearchIndex(column="embedding", kind=SEMANTIC, index_type="vector", ready=True),
    )
    description = sql_tool_description(
        search_table=TABLE,
        search_column="content",
        search_route=SearchRoute(
            SEMANTIC,
            SearchIndex(
                column="content",
                kind=SEMANTIC,
                index_type="vector",
                ready=True,
                embeds_query=True,
            ),
        ),
        also_searchable=[plain],
    )
    assert "vector_search('default.public.corpus'" not in description


def test_a_declared_text_column_survives_a_semantic_registered_route() -> None:
    """A confirmed column the caller paid a control-plane call for should not vanish
    because the search tool happens to be registered over the other kind of index."""
    description = sql_tool_description(
        "hotdata_search_semantic",
        search_table=TABLE,
        search_column=COLUMN,
        search_route=_semantic_route(embeds_query=True),
        also_searchable=[_searchable("default.public.corpus", "content")],
    )
    assert "bm25_search('default.public.corpus', 'content'" in description
    assert f"vector_search('{TABLE}', '{COLUMN}'" in description


def test_a_plain_vector_route_still_offers_the_columns_that_do_compose() -> None:
    """The registered column needing a query vector says nothing about another table's
    BM25 index, and dropping every declared column for it repeats the defect this
    function was corrected for."""
    description = sql_tool_description(
        "hotdata_search_semantic",
        search_table=TABLE,
        search_column=COLUMN,
        search_route=_semantic_route(embeds_query=False),
        also_searchable=[_searchable("default.public.corpus", "content")],
    )
    assert "bm25_search('default.public.corpus', 'content'" in description
    assert f"Ranking '{COLUMN}' on {TABLE} by meaning needs the query as a vector" in description
    assert f"vector_search('{TABLE}'" not in description


def test_a_declared_plain_vector_column_is_never_offered_as_composable() -> None:
    """Writing that search needs a query vector, which an agent writing SQL cannot make."""
    plain = SearchableColumn(
        "default.public.corpus",
        SearchIndex(column="embedding", kind=SEMANTIC, index_type="vector", ready=True),
    )
    description = sql_tool_description(
        "hotdata_search_text", search_table=TABLE, search_column=COLUMN, also_searchable=[plain]
    )
    assert "vector_search" not in description
    assert f"bm25_search('{TABLE}', '{COLUMN}'" in description


def test_searchable_columns_reaches_the_sql_description(mock_client: MagicMock) -> None:
    """The wiring between the verifier and the description is what a caller touches, and
    it is the part that would silently stop threading through."""
    confirmed = [_searchable("default.public.corpus", "content")]
    with patch("hotdata_langchain.tools.verify_searchable_columns", return_value=confirmed):
        built = {
            tool.name: tool.description or ""
            for tool in make_hotdata_tools(
                mock_client,
                search_table=TABLE,
                search_column=COLUMN,
                searchable_columns=[("default.public.corpus", "content")],
            )
        }
    assert "bm25_search('default.public.corpus', 'content'" in built["hotdata_execute_sql"]
