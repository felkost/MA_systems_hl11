"""`middleware.py` -- ported from `MA_systems_hl10` with two corrections
(`docs/specs/stage-3.md`): D3.1b's precise hook contract, and D3.7's rename
of the two Supervisor-only classes onto this project's own tool names
(`plan`/`research`/`critique`, not hl10's `delegate_to_*`).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage

import middleware
from tests.fakes import FakeToolCallingModel


def _middleware_classes() -> list[type[AgentMiddleware[Any, Any, Any]]]:
    return [
        obj
        for obj in vars(middleware).values()
        if isinstance(obj, type)
        and issubclass(obj, AgentMiddleware)
        and obj is not AgentMiddleware
    ]


# -- D3.1b: override detected by identity, not inspect.iscoroutinefunction --
# Measured: the *base* AgentMiddleware.awrap_tool_call is itself an async def
# that raises NotImplementedError, so iscoroutinefunction(base.awrap_tool_call)
# is True -- it cannot distinguish "overridden" from "inherited". The correct
# primitive is identity comparison against the base class attribute, the same
# one langchain's own factory uses to detect overrides.


@pytest.mark.parametrize("cls", _middleware_classes(), ids=lambda c: c.__name__)
def test_custom_middleware_define_both_hook_variants(
    cls: type[AgentMiddleware[Any, Any, Any]],
) -> None:
    pairs = [
        ("wrap_model_call", "awrap_model_call"),
        ("wrap_tool_call", "awrap_tool_call"),
    ]
    for sync_name, async_name in pairs:
        sync_overridden = getattr(cls, sync_name) is not getattr(
            AgentMiddleware, sync_name
        )
        async_overridden = getattr(cls, async_name) is not getattr(
            AgentMiddleware, async_name
        )
        assert sync_overridden == async_overridden, (
            f"{cls.__name__} overrides {sync_name} ({sync_overridden}) and "
            f"{async_name} ({async_overridden}) inconsistently -- the "
            "missing variant inherits the base's NotImplementedError, which "
            "ToolErrorMiddleware launders into a plausible-looking failure"
        )


# -- Test 11: CriticVerificationMiddleware retries exactly once --
#
# Unit-tested directly against `wrap_model_call`/`awrap_model_call` rather
# than through a full `create_agent` invocation: routing the retry's forced
# tool call through the real graph would execute the actual `web_search`
# tool (a live DuckDuckGo call) and, with no further scripted response to
# terminate the run, loop until `GraphRecursionError` -- measured. The
# middleware's own contract ("the retried response is returned as-is,
# whatever it contains") is exactly what a direct handler-call check proves,
# without needing a real tool node at all.


def _model_request(**overrides: Any) -> ModelRequest[Any]:
    defaults: dict[str, Any] = {
        "model": FakeToolCallingModel(responses=[AIMessage(content="")]),
        "messages": [HumanMessage("Verify these findings")],
        "state": {"messages": [HumanMessage("Verify these findings")]},
        "model_settings": {},
    }
    return ModelRequest(**{**defaults, **overrides})


def test_critic_verification_middleware_retries_once_when_no_tool_was_called() -> None:
    calls = 0

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return ModelResponse(result=[AIMessage(content="no verification call")])

    result = middleware.CriticVerificationMiddleware().wrap_model_call(
        _model_request(), handler
    )
    assert calls == 2
    assert result.result[0].content == "no verification call"


def test_critic_verification_middleware_does_not_retry_when_a_tool_was_called() -> None:
    calls = 0

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "web_search", "args": {"query": "x"}, "id": "c1"}
                    ],
                )
            ]
        )

    middleware.CriticVerificationMiddleware().wrap_model_call(_model_request(), handler)
    assert calls == 1


def test_critic_verification_middleware_skips_retry_if_already_verified() -> None:
    calls = 0
    prior_ai = AIMessage(
        content="",
        tool_calls=[{"name": "knowledge_search", "args": {"query": "x"}, "id": "c0"}],
    )

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return ModelResponse(result=[AIMessage(content="verdict, no new tool call")])

    request = _model_request(state={"messages": [HumanMessage("q"), prior_ai]})
    middleware.CriticVerificationMiddleware().wrap_model_call(request, handler)
    assert calls == 1


# -- Test 12: agent_middleware() order and retry-on-failure --


def test_agent_middleware_stack_order_and_retry_on_failure() -> None:
    stack = middleware.agent_middleware(tool_call_limit=5)
    kinds = [type(m) for m in stack]
    assert kinds == [
        ModelCallLimitMiddleware,
        ToolCallLimitMiddleware,
        ToolErrorMiddleware,
        ToolRetryMiddleware,
        ModelRetryMiddleware,
    ]
    retry = next(m for m in stack if isinstance(m, ToolRetryMiddleware))
    assert retry.on_failure == "error"


def test_agent_middleware_respects_the_tool_call_limit_argument() -> None:
    stack = middleware.agent_middleware(tool_call_limit=7)
    limiter = next(m for m in stack if isinstance(m, ToolCallLimitMiddleware))
    assert limiter.run_limit == 7


# -- Test 14: D3.7's renamed tool set, vacuous until stage 4 by design --


def test_stability_tools_match_the_supervisor_tool_names() -> None:
    """Pinned against CLAUDE.md's `supervisor.py` row / hl8's `_S1`
    (`plan`, `research`, `critique`, `save_report`), not against
    `supervisor.py` itself -- it does not exist until stage 4. The rule is
    written before the code it constrains, the same shape as
    `test_forbidden_pairs_never_import_each_other`."""
    expected = frozenset({"research", "critique"})
    assert middleware.SUPERVISOR_DELEGATION_TOOLS == expected
