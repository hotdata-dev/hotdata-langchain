"""Instant database helpers for LangChain agents."""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from hotdata.api.databases_api import DatabasesApi
from hotdata.exceptions import ApiException
from hotdata_framework import (
    DEFAULT_SCHEMA,
    HotdataClient,
    LoadManagedTableResult,
    ManagedDatabase,
    TablePartitionKey,
    TableSortKey,
)
from hotdata_framework.databases import api_error_message, managed_database_from_detail

logger = logging.getLogger(__name__)

CATALOG_QUERY = (
    "SELECT DISTINCT table_catalog FROM information_schema.tables "
    "WHERE table_schema <> 'information_schema'"
)

#: How a load treats the rows a table already holds. ``replace`` discards them;
#: ``append`` keeps them; the remaining three match incoming rows against existing ones
#: and so are legal only on a table that was declared with a key.
LoadMode = Literal["replace", "append", "upsert", "update", "delete"]

#: Load modes that match rows by key, and so require one.
KEYED_LOAD_MODES: frozenset[str] = frozenset({"upsert", "update", "delete"})

URL_SCHEMES = ("http://", "https://")
PARQUET_MAGIC = b"PAR1"
FETCH_TIMEOUT_SECONDS = 30.0
FETCH_USER_AGENT = "hotdata-langchain"
MAX_DOWNLOAD_BYTES = 1024**3
DOWNLOAD_CHUNK_BYTES = 1024 * 256


def resolve_database_by_id(
    client: HotdataClient,
    database_id: str | ManagedDatabase,
) -> ManagedDatabase:
    """Fetch an instant database record by id (``GET /databases/{id}``).

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
                f"no instant database with id {database_id!r} in this workspace. "
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


def scoped_description(description: str, label: str | None) -> str:
    """Return ``description`` led by which database the tool works on.

    Two tool sets built over different databases put two identically-worded tools in one
    prompt, and a model has nothing to choose between them on. The scope leads rather than
    trails because it is the one fact that separates them.

    Returns ``description`` unchanged when there is no label, so a single-database tool set
    carries no sentence about a distinction that does not exist.
    """
    if not label:
        return description
    return f"Works on the '{label}' database. {description}"


def database_label(database: ManagedDatabase | None) -> str | None:
    """Return the display name of ``database``, or ``None`` when it has none.

    The id is deliberately not a fallback. It carries no meaning a model can plan against,
    and presenting one as a name invites passing it where a name is expected.
    """
    return database.description if database is not None else None


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
    """List this workspace's instant databases as JSON, each with its ``id`` and ``name``.

    ``name`` is the API's ``name`` field. ``ManagedDatabase`` still holds it under
    ``description``, which this package does not surface.
    """
    rows = [{"id": db.id, "name": db.description} for db in client.list_managed_databases()]
    return json.dumps(rows, indent=2)


def create_managed_database(
    client: HotdataClient,
    *,
    name: str,
    schema: str = DEFAULT_SCHEMA,
    tables: list[str] | None = None,
    keys: dict[str, list[str]] | None = None,
    expires_at: str | None = None,
    partition_by: dict[str, Sequence[TablePartitionKey]] | None = None,
    sorted_by: dict[str, Sequence[TableSortKey]] | None = None,
) -> ManagedDatabase:
    """Create an instant database, labelled ``name``.

    ``name`` is a display label only; address the result by its ``id`` from here on.

    ``keys`` declares each table's natural key, mapping a table name to its key columns.
    A key can only be declared here, at creation: a table created without one is keyless
    for the rest of its life, and every key-matched load mode is rejected against it.

    ``expires_at`` is an RFC 3339 timestamp or a relative window such as ``"24h"`` or
    ``"7d"``, after which the database is reaped. Without it the database lives until
    something deletes it, which makes lifetime a cleanup script's problem rather than a
    property of the thing created.

    ``partition_by`` and ``sorted_by`` set a table's physical layout, and both are
    permanent: the API has no ALTER path, so the only way to change one is to delete the
    table and reload it, which burns the table name in that database. They are reachable
    here and deliberately not offered to a model — see
    :func:`~hotdata_langchain.tools.make_hotdata_tools`.
    """
    return client.create_managed_database(
        description=name,
        schema=schema,
        tables=tables,
        keys=keys,
        expires_at=expires_at,
        partition_by=partition_by,
        sorted_by=sorted_by,
    )


def is_url(file: str) -> bool:
    return file.lower().startswith(URL_SCHEMES)


def reject_unroutable_url(url: str, *, allow_private_hosts: bool = False) -> None:
    """Raise unless ``url`` is an HTTP(S) URL resolving to a publicly routable address.

    The URL reaching :func:`fetch_parquet` is chosen by the model, and a model's inputs
    include whatever text it retrieved — so an instruction planted in a document is enough
    to pick one. Without this check the agent process becomes a fetcher for whatever its
    own network can see, which in a deployment is usually more than the public internet:
    a cloud metadata endpoint on 169.254.169.254, an internal service on a private range.
    Blocked because a load completes the loop: an internal URL serving parquet would be
    uploaded into the workspace and readable from SQL on the next turn.

    Every address the host resolves to is checked, not just the first, and the caller
    re-runs this on each redirect hop — validating only the URL as written is defeated by
    a public URL that 302s to a private one.

    ``allow_private_hosts`` turns the check off, for a deployment whose data genuinely
    sits on an internal host. It is off by default because the safe direction is the one
    that fails loudly.

    This narrows the reachable surface rather than sealing it. The address is resolved
    here and again by ``urlopen``, so a DNS server that answers differently each time can
    still get through; a deployment on a hostile network wants an egress proxy, not this.
    """
    if not is_url(url):
        raise ValueError(f"expected an http:// or https:// URL, got {url!r}")
    if allow_private_hosts:
        return

    host = urlsplit(url).hostname
    if not host:
        raise ValueError(f"no host in {url!r}")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except socket.gaierror as e:
        raise ValueError(f"could not resolve {host!r} from {url!r}: {e}") from e

    for address in sorted(addresses):
        if not ipaddress.ip_address(address).is_global:
            raise ValueError(
                f"{url!r} resolves to {address}, which is not a public address. Loading "
                "from an internal host would let a URL chosen by the model reach a "
                "service only this process can see. Pass allow_private_hosts=True if "
                "that host is genuinely where your data lives."
            )


class _ValidatingRedirectHandler(HTTPRedirectHandler):
    """Re-checks every redirect target, since the first URL is not the one fetched."""

    def __init__(self, *, allow_private_hosts: bool) -> None:
        self._allow_private_hosts = allow_private_hosts

    def redirect_request(  # the signature is urllib's
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        reject_unroutable_url(newurl, allow_private_hosts=self._allow_private_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_parquet(
    url: str,
    *,
    timeout: float = FETCH_TIMEOUT_SECONDS,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    allow_private_hosts: bool = False,
) -> str:
    """Download a parquet file to a temporary path and return that path.

    The caller is responsible for deleting it.

    Fetched here rather than server-side, so the request carries the agent's network
    position rather than the workspace's. That is the narrower blast radius of the two,
    not a safe one — see :func:`reject_unroutable_url`, which is what bounds it, and which
    also runs on every redirect hop.

    ``max_bytes`` caps the download. An agent that follows a planted link, or simply
    mistakes one large file for another, would otherwise fill the disk of a process it
    shares with every other request. ``Content-Length`` is checked first when the server
    sends one, so an oversized file is usually refused before any of it is transferred,
    and the stream is counted regardless because that header is optional and can lie.

    A ``User-Agent`` is set because asset hosts reject urllib's default with 403, which
    the demo hit and worked around by hand. The download is checked for parquet's ``PAR1``
    magic before it goes anywhere: a URL that answers 200 with an HTML error page would
    otherwise be uploaded and fail as a load, several steps from the cause.

    Raises ``ValueError`` for a URL that is not HTTP(S), resolves to a private address, or
    returns something too large or not parquet.
    """
    reject_unroutable_url(url, allow_private_hosts=allow_private_hosts)

    opener = build_opener(_ValidatingRedirectHandler(allow_private_hosts=allow_private_hosts))
    request = Request(url, headers={"User-Agent": FETCH_USER_AGENT})  # the scheme is checked
    handle = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)  # noqa: SIM115
    try:
        with opener.open(request, timeout=timeout) as response:
            _reject_oversized_header(response, url=url, max_bytes=max_bytes)
            _copy_capped(response, handle, url=url, max_bytes=max_bytes)
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
        Path(handle.name).unlink(missing_ok=True)
        raise
    return handle.name


def _reject_oversized_header(response: Any, *, url: str, max_bytes: int) -> None:
    """Refuse before transferring anything, when the server declares the size."""
    declared = response.headers.get("Content-Length")
    if declared is None:
        return
    try:
        length = int(declared)
    except ValueError:
        return
    if length > max_bytes:
        raise ValueError(_too_large(url, length, max_bytes))


def _copy_capped(response: Any, handle: Any, *, url: str, max_bytes: int) -> None:
    written = 0
    while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
        written += len(chunk)
        if written > max_bytes:
            raise ValueError(_too_large(url, None, max_bytes))
        handle.write(chunk)


def _too_large(url: str, length: int | None, max_bytes: int) -> str:
    size = f"{length} bytes" if length is not None else f"more than {max_bytes} bytes"
    return (
        f"{url!r} is {size}, over the {max_bytes}-byte limit. Raise max_bytes if the file "
        "is genuinely this large; the cap is there so one download cannot fill the disk of "
        "a long-running agent process."
    )


def load_managed_table(
    client: HotdataClient,
    *,
    database_id: str | ManagedDatabase,
    table: str,
    file: str,
    schema: str = DEFAULT_SCHEMA,
    mode: LoadMode = "replace",
    key: list[str] | None = None,
    allow_private_hosts: bool = False,
) -> LoadManagedTableResult:
    """Load a parquet file into a declared table of the database with that id.

    ``file`` is a local path or an ``http(s)`` URL. A URL is downloaded here and uploaded
    from the temporary copy, which is removed afterwards whether or not the load succeeds.
    URLs are what makes this reachable from a deployed agent at all: an Agent Server has
    no filesystem the requesting user can put a file on, so a path-only load can ingest
    nothing the process did not already hold. It is fetched under the limits described on
    :func:`fetch_parquet`; ``allow_private_hosts`` lifts the one that would refuse an
    internal host.

    ``database_id`` is resolved by id (see :func:`resolve_database_by_id`) and the
    resolved record is what addresses the load, so a display label never selects the
    target. The default load replaces the table's contents, which is why addressing it
    unambiguously matters.

    ``mode`` chooses what happens to rows already there. ``upsert``, ``update`` and
    ``delete`` match incoming rows against existing ones, so they need ``key``, and they
    are rejected unless the table was declared with one — which can only happen at
    creation. Raises ``ValueError`` for a keyed mode called without ``key``, rather than
    letting the engine reject it after the file has been uploaded.
    """
    if mode in KEYED_LOAD_MODES and not key:
        raise ValueError(
            f"mode={mode!r} matches rows by key, so 'key' is required. Pass the column "
            "names the table was declared with, or use mode='replace' or 'append'."
        )
    database = resolve_database_by_id(client, database_id)
    if is_url(file):
        path = fetch_parquet(file, allow_private_hosts=allow_private_hosts)
        try:
            return client.load_managed_table(
                database, table, schema=schema, file=path, mode=mode, key=key
            )
        finally:
            Path(path).unlink(missing_ok=True)
    if not Path(file).is_file():
        raise FileNotFoundError(
            f"no file at {file!r}. Pass a path to a local parquet file, or an http:// or "
            "https:// URL to one — other formats are not accepted."
        )
    return client.load_managed_table(database, table, schema=schema, file=file, mode=mode, key=key)


def managed_database_summary(db: ManagedDatabase) -> dict[str, str]:
    return {"id": db.id, "name": db.description or db.id}


def load_result_summary(result: LoadManagedTableResult) -> dict[str, Any]:
    return {
        "connection_id": result.connection_id,
        "schema_name": result.schema_name,
        "table_name": result.table_name,
        "row_count": result.row_count,
        "full_name": result.full_name,
    }
