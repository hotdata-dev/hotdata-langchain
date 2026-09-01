"""LangChain tools for Hotdata runtime."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hotdata-langchain")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from hotdata_framework import HotdataClient, ManagedDatabase, QueryResult, from_env

from hotdata_langchain._sql import DISTANCE_FUNCTIONS, DistanceMetric
from hotdata_langchain.databases import (
    LoadMode,
    create_managed_database,
    list_managed_databases_json,
    load_managed_table,
    load_result_summary,
    managed_database_summary,
    resolve_database_by_id,
)
from hotdata_langchain.errors import (
    HotdataToolError,
    engine_error_message,
    error_feedback,
    with_error_feedback,
)
from hotdata_langchain.indexes import (
    CAPABILITY_PHRASES,
    SEARCH_NOUNS,
    SEMANTIC,
    TEXT,
    SearchableColumn,
    SearchIndex,
    capabilities_by_column,
    fusable_vector_indexes,
    generated_vector_columns,
    indexes_for_column,
    list_search_indexes,
    search_nouns_by_column,
    verify_searchable_columns,
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
    DEFAULT_SEMANTIC_TOOL_NAME,
    DISTANCE_COLUMN,
    RRF_K,
    SCORE_COLUMN,
    Fusion,
    SearchRoute,
    SearchStrategy,
    bm25_search_json,
    bm25_search_sql,
    hybrid_search_json,
    hybrid_search_sql,
    make_hotdata_search_tool,
    resolve_search_route,
    semantic_search_json,
    vector_distance_sql,
    vector_search_sql,
)
from hotdata_langchain.tools import (
    DEFAULT_CREATE_DATABASE_TOOL_NAME,
    DEFAULT_LIST_DATABASES_TOOL_NAME,
    DEFAULT_LOAD_TABLE_TOOL_NAME,
    DEFAULT_SQL_TOOL_NAME,
    DESTRUCTIVE_TOOL_NAMES,
    execute_sql_json,
    make_hotdata_tools,
    result_rows_for_llm,
    suffixed_tool_name,
)
from hotdata_langchain.vectorstore import HotdataVectorStore

__all__ = [
    "CAPABILITY_PHRASES",
    "CLIENT_WARNING_KEY",
    "DEFAULT_CREATE_DATABASE_TOOL_NAME",
    "DEFAULT_DESCRIBE_TOOL_NAME",
    "DEFAULT_KEY_COLUMN",
    "DEFAULT_LIST_DATABASES_TOOL_NAME",
    "DEFAULT_LOAD_TABLE_TOOL_NAME",
    "DEFAULT_SEARCH_LIMIT",
    "DEFAULT_SEARCH_TOOL_NAME",
    "DEFAULT_SEMANTIC_TOOL_NAME",
    "DEFAULT_SQL_TOOL_NAME",
    "DESTRUCTIVE_TOOL_NAMES",
    "DISTANCE_COLUMN",
    "DISTANCE_FUNCTIONS",
    "RRF_K",
    "SCORE_COLUMN",
    "SEARCH_NOUNS",
    "SEMANTIC",
    "TEXT",
    "DistanceMetric",
    "Fusion",
    "HotdataClient",
    "HotdataToolError",
    "HotdataVectorStore",
    "LoadMode",
    "ManagedDatabase",
    "QueryResult",
    "SearchIndex",
    "SearchRoute",
    "SearchStrategy",
    "SearchableColumn",
    "__version__",
    "bm25_search_json",
    "bm25_search_sql",
    "capabilities_by_column",
    "create_managed_database",
    "describe_tables_json",
    "engine_error_message",
    "error_feedback",
    "execute_sql_json",
    "from_env",
    "fusable_vector_indexes",
    "generated_vector_columns",
    "hybrid_search_json",
    "hybrid_search_sql",
    "indexes_for_column",
    "list_managed_databases_json",
    "list_search_indexes",
    "load_managed_table",
    "load_result_summary",
    "make_hotdata_describe_tables_tool",
    "make_hotdata_search_tool",
    "make_hotdata_tools",
    "managed_database_summary",
    "resolve_database_by_id",
    "resolve_search_route",
    "result_json",
    "result_payload",
    "result_rows_for_llm",
    "search_nouns_by_column",
    "semantic_search_json",
    "suffixed_tool_name",
    "vector_distance_sql",
    "vector_search_sql",
    "verify_searchable_columns",
    "with_error_feedback",
]
