"""Handing engine failures back to the model instead of raising them at the graph."""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Iterable
from typing import Any, TypeVar

from langchain_core.tools import BaseTool

# langgraph is what these tools run under, but it is not a dependency of this package.
# An empty tuple is a legal `except` target that never matches, so this is inert without it.
_GRAPH_CONTROL_FLOW: tuple[type[BaseException], ...]
try:
    from langgraph.errors import GraphBubbleUp

    _GRAPH_CONTROL_FLOW = (GraphBubbleUp,)
except ImportError:  # pragma: no cover - exercised by an install without langgraph
    _GRAPH_CONTROL_FLOW = ()

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 1000

ToolT = TypeVar("ToolT", bound=BaseTool)


class HotdataToolError(RuntimeError):
    """A failure this package has already turned into a message the model can act on.

    Raised where the package knows something about the failure that the engine's own
    message does not carry — a SQL query whose date format pattern will not be
    interpreted, for instance, which the engine can answer with nothing more specific
    than an internal error. :func:`engine_error_message` returns its text verbatim
    instead of walking past it to the response body it was built from.
    """


def engine_error_message(exc: BaseException, *, max_chars: int = MAX_MESSAGE_CHARS) -> str:
    """Return the engine's own message for a failed call.

    The framework raises ``RuntimeError(e.reason)`` — "Bad Request" — while the text that
    says what to do differently ("Invalid function 'date_sub'. Did you mean 'date_bin'?")
    stays in the ``ApiException`` body further down the chain. A model handed only "Bad
    Request" has nothing to correct against; handed the real message it was observed to
    fix an invalid query on the next call.

    Walks ``__cause__`` first and ``__context__`` second, so a message survives whether
    the wrapping ``raise`` used ``from`` or not, and returns the first body it finds. A
    body that does not parse as the engine's error envelope is returned as text rather
    than discarded. Falls back to ``str(exc)`` when nothing in the chain carries a body.

    The result is truncated to ``max_chars``: it goes into the model's context on every
    failure, and a long body is noise the model cannot act on anyway.

    A :class:`HotdataToolError` is returned as it stands. Its message was assembled here
    and already contains whatever the chain below it carries, so walking past it would
    replace a message naming the fix with the engine's own less specific one.
    """
    if isinstance(exc, HotdataToolError):
        return _truncate(str(exc), max_chars)
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        body = getattr(node, "body", None)
        if body:
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="replace")
            try:
                return _truncate(str(json.loads(body)["error"]["message"]), max_chars)
            except (ValueError, KeyError, TypeError):
                return _truncate(str(body), max_chars)
        node = node.__cause__ or node.__context__
    return _truncate(str(exc), max_chars)


def _truncate(message: str, max_chars: int) -> str:
    if len(message) <= max_chars:
        return message
    return message[:max_chars] + "…"


def error_feedback(tool: ToolT) -> ToolT:
    """Return a copy of ``tool`` that returns its failures as JSON instead of raising.

    An exception out of a tool aborts the whole LangGraph run, so a single invalid query
    ends the conversation rather than costing the model one turn. Neither obvious
    LangChain mechanism covers it: ``create_agent`` does not accept a ``ToolNode``, so
    ``handle_tool_errors`` is unreachable, and ``BaseTool.handle_tool_error`` only catches
    ``ToolException`` while these tools raise ``RuntimeError``. The failure is returned as
    ``{"error": "<engine message>"}``, matching the JSON envelope the tools already
    return.

    Both ``func`` and ``coroutine`` are wrapped when set. Wrapping only ``func`` is not
    enough: LangChain calls ``coroutine`` in preference under async, which is how
    ``langgraph dev`` and every deployed Agent Server run, so a sync-only wrapper routes
    around its own error handling in exactly the environment that needs it. Clearing
    ``coroutine`` instead would work but demote the tool to running in a thread.

    Each wrapper carries the wrapped function's signature, because LangChain decides
    whether to inject ``callbacks`` and ``config`` by inspecting it —
    ``create_retriever_tool`` builds both of its callables with a ``callbacks``
    parameter, and a bare ``*args, **kwargs`` wrapper would silently stop that reaching
    the retriever.

    A successful result is returned exactly as the tool produced it, and a failure is
    reported in whatever shape the tool declared. Both matter for
    ``response_format="content_and_artifact"``, which requires a ``(content, artifact)``
    pair on every call: coercing a successful pair to ``str``, or reporting a failure as a
    bare string, makes LangChain raise and abort the graph — the failure this helper exists
    to prevent, arriving through the path meant to protect against it.

    LangGraph's control-flow exceptions are re-raised rather than reported. They are
    ordinary ``Exception`` subclasses, so a tool calling ``interrupt()`` for human
    approval would otherwise have its pause converted into an error message and the graph
    would run straight past it. langgraph is not a dependency here, so this is inert when
    it is absent.

    Raises ``TypeError`` for a tool exposing neither callable, since returning it
    unwrapped would reproduce the silent-bypass this exists to prevent.
    """
    update: dict[str, Any] = {}
    func = getattr(tool, "func", None)
    coroutine = getattr(tool, "coroutine", None)

    artifact = getattr(tool, "response_format", None) == "content_and_artifact"
    if func is not None:
        update["func"] = _wrap_sync(func, tool.name, artifact=artifact)
    if coroutine is not None:
        update["coroutine"] = _wrap_async(coroutine, tool.name, artifact=artifact)
    if not update:
        raise TypeError(
            f"cannot add error feedback to tool {tool.name!r}: it exposes neither a "
            "'func' nor a 'coroutine' to wrap. Tools built by StructuredTool.from_function "
            "or create_retriever_tool do; a BaseTool subclass overriding _run does not, and "
            "should catch its own errors."
        )
    return tool.model_copy(update=update)


def _wrap_sync(func: Any, name: str, *, artifact: bool) -> Any:
    @functools.wraps(func)
    def safe(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except _GRAPH_CONTROL_FLOW:
            raise
        except Exception as e:  # nothing else may escape to the graph
            return _error_payload(e, name, artifact=artifact)

    return safe


def _wrap_async(coroutine: Any, name: str, *, artifact: bool) -> Any:
    @functools.wraps(coroutine)
    async def safe(*args: Any, **kwargs: Any) -> Any:
        try:
            return await coroutine(*args, **kwargs)
        except _GRAPH_CONTROL_FLOW:
            raise
        except Exception as e:  # nothing else may escape to the graph
            return _error_payload(e, name, artifact=artifact)

    return safe


def _error_payload(exc: BaseException, name: str, *, artifact: bool) -> str | tuple[str, None]:
    """Report the failure in the shape the tool declared it returns.

    A tool set to ``content_and_artifact`` is required to return a pair even when it
    failed. Handing it a bare string makes LangChain raise, which would abort the graph on
    the error path after the success path was fixed — the same defect, moved.
    """
    message = engine_error_message(exc)
    logger.warning("tool %s failed; returning the error to the model: %s", name, message)
    payload = json.dumps({"error": message})
    return (payload, None) if artifact else payload


def with_error_feedback(tools: Iterable[ToolT]) -> list[ToolT]:
    """Return copies of ``tools`` that return their failures instead of raising.

    See :func:`error_feedback` for what the wrapping does and why both callables are
    wrapped. Applies to any tool, not only Hotdata's — a retriever tool built alongside
    these has the same graph-aborting failure mode.
    """
    return [error_feedback(tool) for tool in tools]
