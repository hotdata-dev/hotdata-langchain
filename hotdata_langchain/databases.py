"""Managed database helpers for LangChain agents."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from typing import Any
from urllib.request import Request, urlopen

from hotdata.api.databases_api import DatabasesApi
from hotdata.exceptions import ApiException
from hotdata_framework import (
    DEFAULT_SCHEMA,
    HotdataClient,
    LoadManagedTableResult,
    ManagedDatabase,
)
from hotdata_framework.databases import api_error_message, managed_database_from_detail

logger = logging.getLogger(__name__)

CATALOG_QUERY = (
    "SELECT DISTINCT table_catalog FROM information_schema.tables "
    "WHERE table_schema <> 'information_schema'"
)

URL_SCHEMES = ("http://", "https://")
PARQUET_MAGIC = b"PAR1"
FETCH_TIMEOUT_SECONDS = 30.0
FETCH_USER_AGENT = "hotdata-langchain"


def resolve_database_by_id(
    client: HotdataClient,
    database_id: str | ManagedDatabase,
) -> ManagedDatabase:
    """Fetch a managed database record by id (``GET /databases/{id}``).

    Addresses the database by id only. A Hotdata database name is a display label and is
    not unique, so there is deliberately no by-name fallback: a name that collides with
    another database's label would otherwise resolve to the wrong database, and every
    query, load and drop would follow it there. Ids come from
    :func:`list_managed_databases_json` or :func:`create_managed_database`.

    An already-resolved ``ManagedDatabase`` is returned as-is, so a caller holding one
    pays no lookup.

    Raises ``KeyError`` when the workspace has no database with that id.
    """
    if isinstance(database_id, ManagedDatabase):
        return database_id
    try:
        detail = DatabasesApi(client.api).get_database(database_id)
    except ApiException as e:
        if e.status == 404:
            raise KeyError(
                f"no managed database with id {database_id!r} in this workspace. "
                "Ids are listed by hotdata_list_managed_databases; a database name is "
                "not accepted here, because names are not unique."
            ) from e
        raise RuntimeError(api_error_message(e)) from e
    return managed_database_from_detail(detail)


def query_scope(database: ManagedDatabase | None) -> ManagedDatabase | None:
    """Return ``database`` unchanged, rejecting a scope that was never resolved.

    A string reaching ``HotdataClient`` would go through its name-or-id resolver, whose
    by-name fallback matches a non-unique display label. Resolve with
    :func:`resolve_database_by_id` first.
    """
    if database is None or isinstance(database, ManagedDatabase):
        return database
    raise TypeError(
        f"database must be a resolved ManagedDatabase, got {type(database).__name__}; "
        "resolve it with resolve_database_by_id(client, database_id) first"
    )


def query_catalogs(client: HotdataClient, database: ManagedDatabase) -> list[str]:
    """Return the catalogs that hold tables inside ``database``'s query scope.

    Read from ``information_schema`` rather than from the database record. A database
    reports ``default_catalog = 'default'`` whether or not anything answers to that name:
    an attached source's tables answer to the attachment's alias instead, so ``default``
    resolves nothing there.

    Catalogs holding nothing but their own ``information_schema`` are excluded, so a
    scope exposing one empty catalog alongside an attachment still names the attachment.
    The engine has not been observed listing ``information_schema`` in its own output, so
    that filter matches no rows today.

    Returns an empty list if the query fails, so a description that names the catalog is
    never the reason tool construction fails. The caller then describes the catalog
    generically, which is a weaker contract than naming it — hence a warning rather than
    a debug line.
    """
    try:
        result = client.execute_sql(CATALOG_QUERY, database=query_scope(database))
        catalogs = sorted({row[0] for row in result.rows if row and isinstance(row[0], str)})
    except Exception:
        logger.warning(
            "could not read table_catalog for database %s; the SQL tool description will "
            "not name the catalog",
            database.id,
            exc_info=True,
        )
        return []
    return catalogs


def list_managed_databases_json(client: HotdataClient) -> str:
    rows = [{"description": db.description, "id": db.id} for db in client.list_managed_databases()]
    return json.dumps(rows, indent=2)


def create_managed_database(
    client: HotdataClient,
    *,
    name: str,
    schema: str = DEFAULT_SCHEMA,
    tables: list[str] | None = None,
) -> ManagedDatabase:
    """Create a managed database, labelled ``name``.

    ``name`` is a display label only; address the result by its ``id`` from here on.
    """
    return client.create_managed_database(description=name, schema=schema, tables=tables)


def is_url(file: str) -> bool:
    return file.lower().startswith(URL_SCHEMES)


def fetch_parquet(url: str, *, timeout: float = FETCH_TIMEOUT_SECONDS) -> str:
    """Download a parquet file to a temporary path and return that path.

    The caller is responsible for deleting it.

    Fetched here rather than server-side, so a caller-supplied URL is read inside the
    agent's own trust boundary rather than the workspace's — an engine that fetched URLs
    on request would reach whatever its network can see.

    A ``User-Agent`` is set because asset hosts reject urllib's default with 403, which
    the demo hit and worked around by hand. The download is checked for parquet's ``PAR1``
    magic before it goes anywhere: a URL that answers 200 with an HTML error page would
    otherwise be uploaded and fail as a load, several steps from the cause.

    Raises ``ValueError`` for a non-HTTP URL — this is not a general file fetcher, and
    ``urlopen`` would otherwise honour ``file://``.
    """
    if not is_url(url):
        raise ValueError(f"expected an http:// or https:// URL, got {url!r}")

    request = Request(url, headers={"User-Agent": FETCH_USER_AGENT})  # the scheme is checked above
    handle = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)  # noqa: SIM115
    try:
        with urlopen(request, timeout=timeout) as response:  # the scheme is checked above
            shutil.copyfileobj(response, handle)
        handle.close()
        with open(handle.name, "rb") as downloaded:
            if downloaded.read(len(PARQUET_MAGIC)) != PARQUET_MAGIC:
                raise ValueError(
                    f"{url!r} did not return a parquet file: it does not begin with "
                    f"{PARQUET_MAGIC.decode()}. A URL that redirects to a login or error "
                    "page answers 200 with HTML, which looks like a successful download."
                )
    except BaseException:
        handle.close()
        os.unlink(handle.name)
        raise
    return handle.name


def load_managed_table(
    client: HotdataClient,
    *,
    database_id: str | ManagedDatabase,
    table: str,
    file: str,
    schema: str = DEFAULT_SCHEMA,
) -> LoadManagedTableResult:
    """Load a parquet file into a declared table of the database with that id.

    ``file`` is a local path or an ``http(s)`` URL. A URL is downloaded here and uploaded
    from the temporary copy, which is removed afterwards whether or not the load succeeds.
    URLs are what makes this reachable from a deployed agent at all: an Agent Server has
    no filesystem the requesting user can put a file on, so a path-only load can ingest
    nothing the process did not already hold.

    ``database_id`` is resolved by id (see :func:`resolve_database_by_id`) and the
    resolved record is what addresses the load, so a display label never selects the
    target. This load replaces the table's contents, which is why addressing it
    unambiguously matters.
    """
    database = resolve_database_by_id(client, database_id)
    if is_url(file):
        path = fetch_parquet(file)
        try:
            return client.load_managed_table(database, table, schema=schema, file=path)
        finally:
            os.unlink(path)
    if not os.path.isfile(file):
        raise FileNotFoundError(
            f"no file at {file!r}. Pass a path to a local parquet file, or an http:// or "
            "https:// URL to one — other formats are not accepted."
        )
    return client.load_managed_table(database, table, schema=schema, file=file)


def managed_database_summary(db: ManagedDatabase) -> dict[str, str]:
    return {"id": db.id, "description": db.description or db.id}


def load_result_summary(result: LoadManagedTableResult) -> dict[str, Any]:
    return {
        "connection_id": result.connection_id,
        "schema_name": result.schema_name,
        "table_name": result.table_name,
        "row_count": result.row_count,
        "full_name": result.full_name,
    }
