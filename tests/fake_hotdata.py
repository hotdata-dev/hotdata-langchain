"""An in-memory stand-in for ``HotdataClient`` that really stores and really ranks.

Loads go through ``pyarrow.parquet`` for real, so the ``list<float32>`` embedding column
round-trips rather than being mocked away, and ``execute_sql`` parses the SQL
``HotdataVectorStore`` actually emits and evaluates it. That is what lets LangChain's own
conformance suite run against the store without a live workspace: the suite asserts on
result *ordering*, which a canned return value cannot exercise.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from hotdata_framework import (
    DEFAULT_SCHEMA,
    CreateIndexResult,
    LoadManagedTableResult,
    ManagedDatabase,
    ManagedTable,
    QueryResult,
)

_SEARCH_RE = re.compile(
    r"^SELECT (?P<projection>.+?), (?P<fn>\w+)\(embedding, ARRAY\[(?P<vector>[^\]]*)\]\) "
    r"AS dist FROM (?P<table>\S+)"
    r"(?: WHERE (?P<where>.+?))? "
    r"ORDER BY dist ASC LIMIT (?P<k>\d+)$"
)

_GET_RE = re.compile(
    r"^SELECT (?P<projection>.+?) FROM (?P<table>\S+) WHERE id IN \((?P<ids>.*)\)$"
)

_LITERAL_RE = re.compile(r"'((?:[^']|'')*)'")


def _cosine_distance(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 if norm == 0 else 1.0 - sum(x * y for x, y in zip(a, b, strict=True)) / norm


def _l2_distance(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))


def _negative_dot_product(a: list[float], b: list[float]) -> float:
    return -sum(x * y for x, y in zip(a, b, strict=True))


_DISTANCES = {
    "cosine_distance": _cosine_distance,
    "l2_distance": _l2_distance,
    "negative_dot_product": _negative_dot_product,
}


def _parse_literal(text: str) -> Any:
    text = text.strip()
    if text.startswith("'"):
        return text[1:-1].replace("''", "'")
    if text in ("true", "false"):
        return text == "true"
    return float(text) if "." in text else int(text)


def _matches(row: dict[str, Any], where: str | None) -> bool:
    if not where:
        return True
    for predicate in where.split(" AND "):
        column, _, literal = predicate.partition(" = ")
        if row.get(column.strip()) != _parse_literal(literal):
            return False
    return True


class FakeHotdataClient:
    """Records what it was asked to do, and answers queries from what it stored."""

    workspace_id = "ws_fake"

    #: Stands in for the raw generated API client; the ``databases_api`` fixture patches
    #: ``DatabasesApi`` itself, so this is only ever passed through, never called.
    api = None

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.declared: list[dict[str, Any]] = []
        self.loads: list[str] = []
        self.schemas: list[pa.Schema] = []
        self.queries: list[str] = []
        self.indexes: list[dict[str, Any]] = []

    def create_index(
        self,
        database: ManagedDatabase,
        table: str,
        **kwargs: Any,
    ) -> CreateIndexResult:
        """Record the request, and remember the row count the table held at build time.

        The engine reads a plain vector index's width off stored data, so an index built
        before the first write has nothing to measure. Recording the count is what lets a
        test assert the write happened first.
        """
        self.indexes.append({"table": table, "rows_at_build": len(self.rows), **kwargs})
        return CreateIndexResult(
            full_name=f"{database.id}.{kwargs.get('schema', DEFAULT_SCHEMA)}.{table}",
            schema_name=kwargs.get("schema", DEFAULT_SCHEMA),
            table_name=table,
            index_name=kwargs.get("index_name") or f"{table}_embedding_vector",
            index_type="vector",
            columns=list(kwargs.get("columns") or []),
            metric=kwargs.get("metric"),
            source_column=None,
            status="ready",
            job_id="job_fake",
        )

    def list_managed_tables(
        self,
        database: ManagedDatabase,
        *,
        schema: str | None = None,
    ) -> list[ManagedTable]:
        return [
            ManagedTable(
                full_name=f"{database.id}.{entry['schema']}.{entry['table']}",
                schema=entry["schema"],
                table=entry["table"],
                synced=False,
                last_sync=None,
            )
            for entry in self.declared
            if schema is None or entry["schema"] == schema
        ]

    def add_managed_table(
        self,
        database: ManagedDatabase,
        table: str,
        *,
        schema: str = DEFAULT_SCHEMA,
        key: list[str] | None = None,
    ) -> None:
        self.declared.append({"database": database, "table": table, "schema": schema, "key": key})

    def load_managed_table(
        self,
        database: ManagedDatabase,
        table: str,
        *,
        schema: str = DEFAULT_SCHEMA,
        file: str | None = None,
        mode: str = "replace",
        key: list[str] | None = None,
    ) -> LoadManagedTableResult:
        assert file is not None
        loaded = pq.read_table(file)
        records: list[dict[str, Any]] = loaded.to_pylist()
        self.loads.append(mode)
        self.schemas.append(loaded.schema)
        if mode == "delete":
            for record in records:
                self.rows.pop(record["id"], None)
        else:
            if mode == "replace":
                self.rows.clear()
            for record in records:
                self.rows[record["id"]] = record
        return LoadManagedTableResult(
            connection_id=database.default_connection_id,
            schema_name=schema,
            table_name=table,
            row_count=len(records),
            full_name=f"{database.id}.{schema}.{table}",
        )

    def execute_sql(self, sql: str, *, database: ManagedDatabase | None = None) -> QueryResult:
        self.queries.append(sql)
        search = _SEARCH_RE.match(sql)
        if search is not None:
            return self._search(search)
        lookup = _GET_RE.match(sql)
        if lookup is not None:
            return self._lookup(lookup)
        raise AssertionError(f"unrecognised SQL: {sql}")

    def _search(self, match: re.Match[str]) -> QueryResult:
        distance = _DISTANCES[match["fn"]]
        query_vector = [float(value) for value in match["vector"].split(", ")]
        ranked = sorted(
            (
                (distance(list(row["embedding"]), query_vector), row)
                for row in self.rows.values()
                if _matches(row, match["where"])
            ),
            key=lambda scored: scored[0],
        )[: int(match["k"])]
        columns = [name.strip() for name in match["projection"].split(",")] + ["dist"]
        return self._result(
            columns, [[*(row[name] for name in columns[:-1]), score] for score, row in ranked]
        )

    def _lookup(self, match: re.Match[str]) -> QueryResult:
        wanted = [literal.replace("''", "'") for literal in _LITERAL_RE.findall(match["ids"])]
        columns = [name.strip() for name in match["projection"].split(",")]
        found = [self.rows[row_id] for row_id in wanted if row_id in self.rows]
        return self._result(columns, [[row[name] for name in columns] for row in found])

    @staticmethod
    def _result(columns: list[str], rows: list[list[Any]]) -> QueryResult:
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            result_id="res_fake",
            query_run_id="run_fake",
            execution_time_ms=1,
            warning=None,
            error_message=None,
        )
