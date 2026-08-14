"""An exception out of a tool aborts the agent run, so the wrapping is the behaviour.

The async path is tested as carefully as the sync one: LangChain calls ``coroutine`` in
preference under async, which is how a deployed Agent Server runs, so a wrapper that only
covers ``func`` fails in exactly the environment that needs it.
"""

from __future__ import annotations

import inspect
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, StructuredTool, create_retriever_tool

from hotdata_langchain.errors import (
    engine_error_message,
    error_feedback,
    with_error_feedback,
)
from hotdata_langchain.schema import DEFAULT_DESCRIBE_TOOL_NAME
from hotdata_langchain.search import DEFAULT_SEARCH_TOOL_NAME
from hotdata_langchain.tools import DEFAULT_SQL_TOOL_NAME, make_hotdata_tools


class FakeApiError(Exception):
    """Shaped like ``hotdata.exceptions.ApiException``: the message lives in ``body``."""

    def __init__(self, body: str | bytes) -> None:
        super().__init__("Bad Request")
        self.body = body


def engine_body(message: str) -> str:
    return json.dumps({"error": {"message": message}})


def framework_error(message: str) -> RuntimeError:
    """Build the exception the framework actually raises: 'Bad Request' over a real body."""
    error = RuntimeError("Bad Request")
    error.__cause__ = FakeApiError(engine_body(message))
    return error


def tool_from(func: Any = None, coroutine: Any = None, name: str = "t") -> StructuredTool:
    return StructuredTool.from_function(
        func=func,
        coroutine=coroutine,
        name=name,
        description="a tool with a description long enough to be plausible",
    )


def test_engine_message_is_recovered_from_the_cause_chain() -> None:
    """The framework raises RuntimeError('Bad Request'); the useful text is underneath."""
    body = engine_body("Invalid function 'date_sub'. Did you mean 'date_bin'?")
    try:
        try:
            raise FakeApiError(body)
        except FakeApiError as e:
            raise RuntimeError("Bad Request") from e
    except RuntimeError as e:
        assert engine_error_message(e) == "Invalid function 'date_sub'. Did you mean 'date_bin'?"


def test_engine_message_follows_an_implicit_context_too() -> None:
    """A re-raise without ``from`` sets __context__, not __cause__, and still carries it."""
    try:
        try:
            raise FakeApiError(engine_body("Table 'listings' not found"))
        except FakeApiError:
            raise RuntimeError("Bad Request")  # noqa: B904 - the point of the test
    except RuntimeError as e:
        assert engine_error_message(e) == "Table 'listings' not found"


def test_engine_message_falls_back_to_an_unparseable_body() -> None:
    """A gateway's HTML error page is still more informative than 'Bad Request'."""
    exc = FakeApiError("<html>502 Bad Gateway</html>")
    assert "502 Bad Gateway" in engine_error_message(exc)


def test_engine_message_decodes_a_bytes_body() -> None:
    exc = FakeApiError(engine_body("no such column: pirce").encode())
    assert engine_error_message(exc) == "no such column: pirce"


def test_engine_message_falls_back_to_the_exception_text() -> None:
    """Not every failure comes from the engine — a local path error must survive too."""
    assert engine_error_message(FileNotFoundError("no such file: /tmp/x.parquet")) == (
        "no such file: /tmp/x.parquet"
    )


def test_engine_message_is_truncated() -> None:
    """It lands in the model's context on every failure, and a long body is not actionable."""
    message = engine_error_message(RuntimeError("x" * 5000))
    assert len(message) == 1001
    assert message.endswith("…")


def test_engine_message_survives_a_cyclic_cause_chain() -> None:
    first = RuntimeError("first")
    second = RuntimeError("second")
    first.__cause__ = second
    second.__cause__ = first
    assert engine_error_message(first) == "first"


def test_a_failing_tool_returns_the_error_instead_of_raising() -> None:
    def boom() -> str:
        raise RuntimeError("Bad Request") from FakeApiError(engine_body("syntax error"))

    wrapped = error_feedback(tool_from(func=boom))
    assert json.loads(wrapped.invoke({})) == {"error": "syntax error"}


async def test_a_failing_async_tool_returns_the_error_instead_of_raising() -> None:
    async def boom() -> str:
        raise RuntimeError("Bad Request") from FakeApiError(engine_body("syntax error"))

    wrapped = error_feedback(tool_from(coroutine=boom))
    assert json.loads(await wrapped.ainvoke({})) == {"error": "syntax error"}


async def test_the_async_path_is_wrapped_rather_than_cleared() -> None:
    """Clearing ``coroutine`` would also work, at the cost of demoting async to a thread.

    The failure this guards is subtler than an unhandled error: a tool that still runs,
    but no longer concurrently, in the one environment agents are deployed in.
    """
    calls: list[str] = []

    def sync() -> str:
        calls.append("sync")
        return "sync"

    async def async_() -> str:
        calls.append("async")
        return "async"

    wrapped = error_feedback(tool_from(func=sync, coroutine=async_))
    assert wrapped.coroutine is not None
    assert await wrapped.ainvoke({}) == "async"
    assert calls == ["async"]


async def test_wrapping_only_func_would_miss_the_async_path() -> None:
    """The bug in the demo's version, pinned so it cannot be reintroduced.

    Under async LangChain calls ``coroutine`` in preference, so a tool whose ``func`` was
    wrapped and whose ``coroutine`` was not still aborts the run.
    """

    async def boom() -> str:
        raise RuntimeError("kaboom")

    wrapped = error_feedback(tool_from(func=lambda: "ok", coroutine=boom))
    assert json.loads(await wrapped.ainvoke({})) == {"error": "kaboom"}


def test_a_succeeding_tool_is_untouched() -> None:
    wrapped = error_feedback(tool_from(func=lambda: "the answer"))
    assert wrapped.invoke({}) == "the answer"


def test_wrapping_preserves_the_tool_contract() -> None:
    """It is a copy, so name, description and argument schema all have to survive."""

    def query(sql: str) -> str:
        return sql

    original = tool_from(func=query, name="hotdata_execute_sql")
    wrapped = error_feedback(original)
    assert wrapped.name == original.name
    assert wrapped.description == original.description
    assert wrapped.args == original.args
    assert wrapped.invoke({"sql": "SELECT 1"}) == "SELECT 1"


def test_wrapping_keeps_the_callbacks_parameter_reaching_a_retriever_tool() -> None:
    """LangChain injects ``callbacks`` by inspecting the signature it finds on the callable.

    ``create_retriever_tool`` builds both callables with that parameter, so a bare
    ``*args, **kwargs`` wrapper would hide it and silently break retriever tracing.
    """

    class Retriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Any]:
            return []

    original = create_retriever_tool(Retriever(), "retrieve", "d" * 50)
    wrapped = error_feedback(original)
    for callable_ in (wrapped.func, wrapped.coroutine):
        assert callable_ is not None
        assert "callbacks" in inspect.signature(callable_).parameters


def test_an_artifact_tool_survives_a_successful_call() -> None:
    """Coercing the result to str breaks the tool type the README says to wrap.

    ``response_format="content_and_artifact"`` returns a ``(content, artifact)`` pair.
    Stringified, LangChain rejects it — so a *successful* call raises, aborting the graph
    through the very path this helper exists to protect.
    """

    class Retriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
            return [Document("hello")]

    tool = create_retriever_tool(
        Retriever(), "search_docs", "d" * 50, response_format="content_and_artifact"
    )
    call = {"name": "search_docs", "args": {"query": "x"}, "id": "1", "type": "tool_call"}
    message = error_feedback(tool).invoke(call)
    assert message.content == "hello"
    assert [d.page_content for d in message.artifact] == ["hello"]


def test_a_non_string_result_is_not_coerced() -> None:
    """The wrapper reports failures; formatting the result is LangChain's job, not ours."""
    wrapped = error_feedback(tool_from(func=lambda: {"rows": [1, 2]}))
    assert wrapped.invoke({}) == {"rows": [1, 2]}


def test_a_graph_interrupt_is_not_reported_as_an_error() -> None:
    """It is a plain Exception, so catching broadly turns a pause into an error message.

    A human-in-the-loop tool would stop interrupting and the graph would run past the
    approval it was waiting on, with only a nonsense error to show for it.
    """
    from langgraph.errors import GraphInterrupt

    def needs_approval() -> str:
        raise GraphInterrupt(("pause here",))

    wrapped = error_feedback(tool_from(func=needs_approval, name="approve"))
    with pytest.raises(GraphInterrupt):
        wrapped.invoke({})


async def test_a_graph_interrupt_is_not_reported_as_an_error_under_async() -> None:
    from langgraph.errors import GraphInterrupt

    async def needs_approval() -> str:
        raise GraphInterrupt(("pause here",))

    wrapped = error_feedback(tool_from(coroutine=needs_approval, name="approve"))
    with pytest.raises(GraphInterrupt):
        await wrapped.ainvoke({})


def test_a_tool_with_nothing_to_wrap_is_rejected() -> None:
    """Returning it unwrapped would reproduce the silent bypass this exists to prevent."""

    class Custom(BaseTool):
        name: str = "custom"
        description: str = "a tool that implements _run directly"

        def _run(self) -> str:
            raise RuntimeError("kaboom")

    with pytest.raises(TypeError, match="neither a 'func' nor a 'coroutine'"):
        error_feedback(Custom())


def test_with_error_feedback_wraps_every_tool() -> None:
    tools = with_error_feedback([tool_from(func=lambda: "a", name="a")])
    assert [t.name for t in tools] == ["a"]


def test_handle_errors_is_off_by_default() -> None:
    """Outside an agent loop a raise is the right behaviour, so it stays opt-in."""
    client = MagicMock()
    client.execute_sql.side_effect = RuntimeError("Bad Request")
    tools = {t.name: t for t in make_hotdata_tools(client)}
    with pytest.raises(RuntimeError):
        tools[DEFAULT_SQL_TOOL_NAME].invoke({"sql": "SELECT 1"})


def test_handle_errors_covers_the_tools_make_hotdata_tools_builds() -> None:
    client = MagicMock()
    client.execute_sql.side_effect = framework_error(
        "Invalid function 'to_tsvector'. Did you mean 'to_char'?"
    )
    tools = {t.name: t for t in make_hotdata_tools(client, handle_errors=True)}
    payload = json.loads(tools[DEFAULT_SQL_TOOL_NAME].invoke({"sql": "SELECT bad"}))
    assert payload == {"error": "Invalid function 'to_tsvector'. Did you mean 'to_char'?"}


def test_handle_errors_reaches_the_search_and_describe_tools_too() -> None:
    """They are appended after the core list, so a wrapper applied too early misses them."""
    client = MagicMock()
    client.execute_sql.side_effect = framework_error("no BM25 index on 'description'")
    tools = {
        t.name: t
        for t in make_hotdata_tools(
            client,
            handle_errors=True,
            search_table="default.public.listings",
            search_column="description",
        )
    }
    search = json.loads(tools[DEFAULT_SEARCH_TOOL_NAME].invoke({"query": "garden"}))
    describe = json.loads(tools[DEFAULT_DESCRIBE_TOOL_NAME].invoke({}))
    assert search == {"error": "no BM25 index on 'description'"}
    assert describe == {"error": "no BM25 index on 'description'"}
