"""LangChain tools for Hotdata runtime."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hotdata-langchain")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from hotdata_framework import HotdataClient, ManagedDatabase, QueryResult, from_env

from hotdata_langchain.databases import (
    create_managed_database,
    list_managed_databases_json,
    load_managed_table,
    load_result_summary,
    managed_database_summary,
    resolve_database_by_id,
)
from hotdata_langchain.schema import (
    DEFAULT_DESCRIBE_TOOL_NAME,
    describe_tables_json,
    make_hotdata_describe_tables_tool,
)
from hotdata_langchain.search import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEARCH_TOOL_NAME,
    SCORE_COLUMN,
    bm25_search_json,
    bm25_search_sql,
    make_hotdata_search_tool,
)
from hotdata_langchain.tools import (
    execute_sql_json,
    make_hotdata_tools,
    result_rows_for_llm,
)

__all__ = [
    "DEFAULT_DESCRIBE_TOOL_NAME",
    "DEFAULT_SEARCH_LIMIT",
    "DEFAULT_SEARCH_TOOL_NAME",
    "SCORE_COLUMN",
    "HotdataClient",
    "ManagedDatabase",
    "QueryResult",
    "__version__",
    "bm25_search_json",
    "bm25_search_sql",
    "create_managed_database",
    "describe_tables_json",
    "execute_sql_json",
    "from_env",
    "list_managed_databases_json",
    "load_managed_table",
    "load_result_summary",
    "make_hotdata_describe_tables_tool",
    "make_hotdata_search_tool",
    "make_hotdata_tools",
    "managed_database_summary",
    "resolve_database_by_id",
    "result_rows_for_llm",
]
