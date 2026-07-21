"""LangChain tools for Hotdata runtime."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hotdata-langchain")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from hotdata_framework import HotdataClient, QueryResult, from_env

from hotdata_langchain.cache import HotdataToolCache, cached
from hotdata_langchain.databases import (
    create_managed_database,
    list_managed_databases_json,
    load_managed_table,
    load_result_summary,
    managed_database_summary,
)
from hotdata_langchain.tools import (
    execute_sql_json,
    make_hotdata_tools,
    result_rows_for_llm,
)

__all__ = [
    "HotdataClient",
    "HotdataToolCache",
    "QueryResult",
    "__version__",
    "cached",
    "create_managed_database",
    "execute_sql_json",
    "from_env",
    "list_managed_databases_json",
    "load_managed_table",
    "load_result_summary",
    "make_hotdata_tools",
    "managed_database_summary",
    "result_rows_for_llm",
]
