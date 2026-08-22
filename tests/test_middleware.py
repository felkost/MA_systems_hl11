"""`middleware.py` -- ported from `MA_systems_hl10` with two corrections
(`docs/specs/stage-3.md`): D3.1b's precise hook contract, and D3.7's rename
of the two Supervisor-only classes onto this project's own tool names
(`plan`/`research`/`critique`, not hl10's `delegate_to_*`).
"""

from __future__ import annotations

from typing import Any, cast

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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

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


def _tool_call_request(
    *, name: str, call_id: str = "c1", state: dict[str, Any] | None = None
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {}, "id": call_id},
        tool=None,
        state=state if state is not None else {"messages": []},
        runtime=cast(Any, None),
    )


def _ai_with_calls(*names: str, ids: list[str] | None = None) -> AIMessage:
    call_ids = ids or [f"c{i}" for i in range(len(names))]
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": {}, "id": call_id}
            for name, call_id in zip(names, call_ids)
        ],
    )


def _tool_result(call_id: str, content: str, *, is_error: bool = False) -> ToolMessage:
    return ToolMessage(
        content=content,
        tool_call_id=call_id,
        name="critique",
        status="error" if is_error else "success",
    )


# -- Stage-4 spec D4.5: RevisionCapMiddleware --


def test_revision_cap_allows_calls_up_to_max_revisions_plus_one() -> None:
    # max_revisions=1 -> limit is 2 critique calls total; the 2nd (this
    # in-flight call, excluded from the prior count) must still be allowed.
    prior_ai = _ai_with_calls("critique", ids=["c0"])
    request = _tool_call_request(
        name="critique",
        call_id="c1",
        state={"messages": [HumanMessage("q"), prior_ai]},
    )
    guard = middleware.RevisionCapMiddleware(max_revisions=1)
    assert guard.wrap_tool_call(request, lambda r: _tool_result("c1", "ok")) == (
        _tool_result("c1", "ok")
    )


def test_revision_cap_refuses_past_max_revisions_plus_one() -> None:
    prior_ai = _ai_with_calls("critique", "critique", ids=["c0", "c1"])
    request = _tool_call_request(
        name="critique",
        call_id="c2",
        state={"messages": [HumanMessage("q"), prior_ai]},
    )
    guard = middleware.RevisionCapMiddleware(max_revisions=1)
    result = guard.wrap_tool_call(
        request, lambda r: _tool_result("c2", "should not run")
    )
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "revision cap" in str(result.content)


def test_revision_cap_resets_after_a_new_human_message() -> None:
    # Two prior critique calls from a PREVIOUS question, then a new question
    # starts -- the cap must not carry the old count forward.
    old_ai = _ai_with_calls("critique", "critique", ids=["c0", "c1"])
    request = _tool_call_request(
        name="critique",
        call_id="c2",
        state={"messages": [HumanMessage("q1"), old_ai, HumanMessage("q2")]},
    )
    guard = middleware.RevisionCapMiddleware(max_revisions=1)
    result = guard.wrap_tool_call(request, lambda r: _tool_result("c2", "ok"))
    assert result == _tool_result("c2", "ok")


def test_revision_cap_ignores_non_critique_tools() -> None:
    request = _tool_call_request(name="research", state={"messages": []})
    guard = middleware.RevisionCapMiddleware(max_revisions=1)
    result = guard.wrap_tool_call(request, lambda r: _tool_result("c1", "ok"))
    assert result == _tool_result("c1", "ok")


# -- Stage-4 spec D4.18: SaveReportVerdictGuardMiddleware --


def test_verdict_guard_refuses_without_any_completed_critique() -> None:
    request = _tool_call_request(
        name="save_report",
        state={"messages": [HumanMessage("q")], "verdict": "APPROVE"},
    )
    guard = middleware.SaveReportVerdictGuardMiddleware(max_revisions=2)
    result = guard.wrap_tool_call(
        request, lambda r: ToolMessage("saved", tool_call_id="c1")
    )
    assert isinstance(result, ToolMessage)
    assert result.status == "error"


def test_verdict_guard_refuses_a_stale_approve_after_an_errored_critique() -> None:
    """The exact hole a probe-based verification round found: an emitted-only
    gate would let a critique that raised (no Command state write) leave a
    previous question's checkpointed APPROVE unchallenged."""
    critique_ai = _ai_with_calls("critique", ids=["c0"])
    errored = _tool_result("c0", "ERROR: critique failed (RuntimeError)", is_error=True)
    request = _tool_call_request(
        name="save_report",
        call_id="c1",
        state={
            "messages": [HumanMessage("q2"), critique_ai, errored],
            "verdict": "APPROVE",  # stale, from a previous question
        },
    )
    guard = middleware.SaveReportVerdictGuardMiddleware(max_revisions=2)
    result = guard.wrap_tool_call(
        request, lambda r: ToolMessage("saved", tool_call_id="c1")
    )
    assert isinstance(result, ToolMessage)
    assert result.status == "error"


def test_verdict_guard_allows_save_on_approve() -> None:
    critique_ai = _ai_with_calls("critique", ids=["c0"])
    approved = _tool_result("c0", "verdict APPROVE")
    request = _tool_call_request(
        name="save_report",
        call_id="c1",
        state={
            "messages": [HumanMessage("q"), critique_ai, approved],
            "verdict": "APPROVE",
        },
    )
    guard = middleware.SaveReportVerdictGuardMiddleware(max_revisions=2)
    result = guard.wrap_tool_call(
        request, lambda r: ToolMessage("saved", tool_call_id="c1")
    )
    assert result == ToolMessage("saved", tool_call_id="c1")


def test_verdict_guard_refuses_revise_while_rounds_remain() -> None:
    critique_ai = _ai_with_calls("critique", ids=["c0"])
    revised = _tool_result("c0", "verdict REVISE")
    request = _tool_call_request(
        name="save_report",
        call_id="c1",
        state={
            "messages": [HumanMessage("q"), critique_ai, revised],
            "verdict": "REVISE",
        },
    )
    guard = middleware.SaveReportVerdictGuardMiddleware(max_revisions=2)
    result = guard.wrap_tool_call(
        request, lambda r: ToolMessage("saved", tool_call_id="c1")
    )
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "rounds remain" in str(result.content)


def test_verdict_guard_allows_save_once_the_cap_is_exhausted_on_revise() -> None:
    """A cap-exhausted REVISE run must still be able to save (D4.18) --
    otherwise the run deadlocks against `_S1` rule 5's "stopped revising for
    any reason, save now"."""
    critique_ai = _ai_with_calls(
        "critique", "critique", "critique", ids=["c0", "c1", "c2"]
    )
    results = [
        _tool_result("c0", "verdict REVISE"),
        _tool_result("c1", "verdict REVISE"),
        _tool_result("c2", "verdict REVISE"),
    ]
    request = _tool_call_request(
        name="save_report",
        call_id="c3",
        state={
            "messages": [HumanMessage("q"), critique_ai, *results],
            "verdict": "REVISE",
        },
    )
    guard = middleware.SaveReportVerdictGuardMiddleware(max_revisions=2)
    result = guard.wrap_tool_call(
        request, lambda r: ToolMessage("saved", tool_call_id="c3")
    )
    assert result == ToolMessage("saved", tool_call_id="c3")


def test_verdict_guard_ignores_other_tools() -> None:
    request = _tool_call_request(name="research", state={"messages": []})
    guard = middleware.SaveReportVerdictGuardMiddleware(max_revisions=2)
    result = guard.wrap_tool_call(
        request, lambda r: ToolMessage("ok", tool_call_id="c1")
    )
    assert result == ToolMessage("ok", tool_call_id="c1")


# -- Stage-4 spec D4.15: agent_middleware(tool_exit_behavior=...) --


def test_agent_middleware_default_exit_behavior_is_continue() -> None:
    stack = middleware.agent_middleware(tool_call_limit=5)
    limiter = next(m for m in stack if isinstance(m, ToolCallLimitMiddleware))
    assert limiter.exit_behavior == "continue"


def test_agent_middleware_accepts_end_exit_behavior_for_the_supervisor() -> None:
    stack = middleware.agent_middleware(tool_call_limit=5, tool_exit_behavior="end")
    limiter = next(m for m in stack if isinstance(m, ToolCallLimitMiddleware))
    assert limiter.exit_behavior == "end"


def test_stability_tools_match_the_supervisor_tool_names() -> None:
    """Pinned against CLAUDE.md's `supervisor.py` row / hl8's `_S1`
    (`plan`, `research`, `critique`, `save_report`), not against
    `supervisor.py` itself -- it does not exist until stage 4. The rule is
    written before the code it constrains, the same shape as
    `test_forbidden_pairs_never_import_each_other`."""
    expected = frozenset({"research", "critique"})
    assert middleware.SUPERVISOR_DELEGATION_TOOLS == expected
