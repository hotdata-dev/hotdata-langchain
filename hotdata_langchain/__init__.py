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
from hotdata_langchain.errors import (
    engine_error_message,
    error_feedback,
    with_error_feedback,
)
from hotdata_langchain.results import (
    CLIENT_WARNING_KEY,
    result_json,
    result_payload,
)
from hotdata_langchain.schema import (
    DEFAULT_DESCRIBE_TOOL_NAME,
    describe_tables_json,
    make_hotdata_describe_tables_tool,
)
from hotdata_langchain.search import (
    DEFAULT_KEY_COLUMN,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEARCH_TOOL_NAME,
    SCORE_COLUMN,
    bm25_search_json,
    bm25_search_sql,
    make_hotdata_search_tool,
)
from hotdata_langchain.tools import (
    DEFAULT_CREATE_DATABASE_TOOL_NAME,
    DEFAULT_LIST_DATABASES_TOOL_NAME,
    DEFAULT_LOAD_TABLE_TOOL_NAME,
    DEFAULT_SQL_TOOL_NAME,
    execute_sql_json,
    make_hotdata_tools,
    result_rows_for_llm,
)
from hotdata_langchain.vectorstore import (
    DISTANCE_FUNCTIONS,
    HotdataVectorStore,
)

__all__ = [
    "CLIENT_WARNING_KEY",
    "DEFAULT_CREATE_DATABASE_TOOL_NAME",
    "DEFAULT_DESCRIBE_TOOL_NAME",
    "DEFAULT_KEY_COLUMN",
    "DEFAULT_LIST_DATABASES_TOOL_NAME",
    "DEFAULT_LOAD_TABLE_TOOL_NAME",
    "DEFAULT_SEARCH_LIMIT",
    "DEFAULT_SEARCH_TOOL_NAME",
    "DEFAULT_SQL_TOOL_NAME",
    "DISTANCE_FUNCTIONS",
    "SCORE_COLUMN",
    "HotdataClient",
    "HotdataVectorStore",
    "ManagedDatabase",
    "QueryResult",
    "__version__",
    "bm25_search_json",
    "bm25_search_sql",
    "create_managed_database",
    "describe_tables_json",
    "engine_error_message",
    "error_feedback",
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
    "result_json",
    "result_payload",
    "result_rows_for_llm",
    "with_error_feedback",
]
