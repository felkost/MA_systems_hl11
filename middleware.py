"""Middleware for the three sub-agents and the Supervisor.

Ported from `MA_systems_hl10`, with two corrections
(`docs/specs/stage-3.md`):

**D3.1b -- the hook contract, stated precisely.** `wrap_model_call`/
`wrap_tool_call` raise `NotImplementedError` from their default counterpart
when only one side is overridden -- measured against the installed
langchain 1.3.16: the base class's own `awrap_tool_call` is itself an
`async def` that raises, so `inspect.iscoroutinefunction` cannot tell
"overridden" from "inherited"; only identity comparison against the base
class attribute can (`tests/test_middleware.py`, the same primitive
langchain's own factory uses to detect overrides).
`before_agent`/`after_agent`/`before_model`/`after_model` default to a
**silent no-op** instead. A middleware missing one wrap-hook variant
crashes and gets laundered into a plausible-looking tool failure by
`ToolErrorMiddleware`'s untyped `except Exception` -- but that laundered
string is **this project's own** `_tool_error_to_message`, not the
library's; `ToolErrorMiddleware` requires a caller-supplied handler and
raises `ValueError` without one. A middleware missing one before/after-hook
variant just never runs on that path. Both are bugs; only the first
announces itself.

**D3.7 -- the two Supervisor-only classes are rebound to this project's own
tool names.** hl10 keys `RoundStabilityMiddleware`/`SaveReportGuardMiddleware`
on its A2A delegation tools (`delegate_to_researcher`, `delegate_to_critic`).
This project's Supervisor exposes `plan`/`research`/`critique`/`save_report`
(CLAUDE.md's `supervisor.py` row; hl8's `_S1`). Ported unchanged, both
middlewares would watch for tool calls that never happen and silently never
fire -- present, green, and inert. `SUPERVISOR_DELEGATION_TOOLS` carries the
renamed set as a single constant so stage 4's `supervisor.py` binds its
wrappers against it rather than a second copy of the strings.

Both Supervisor-only classes read state keys (`critic_gaps`, `verdict`) that
do not exist until stage 4's `supervisor.py`. They ship this stage anyway --
splitting one module across two stages costs more than it saves, the same
call hl10 made landing `CritiqueResult` a stage before its Critic. Stage 3
tests them as isolated units against hand-built state; stage 4 proves them
wired.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from opentelemetry import trace

from models import compute_cost
from prompts import CRITIC_VERIFICATION_INSTRUCTION

_VERIFICATION_TOOLS = frozenset({"web_search", "read_url", "knowledge_search"})

# D3.7's rename of hl10's `_STABILITY_TOOLS`. This project's Supervisor
# delegates via `research`/`critique` (CLAUDE.md's `supervisor.py` row);
# `plan` and `save_report` are not stability-checked -- a repeated plan or
# save is not the failure shape this guard exists to catch.
SUPERVISOR_DELEGATION_TOOLS = frozenset({"research", "critique"})

# Not a Settings field: nothing varies it yet, and a field with no consumer
# is noise. A generous but bounded backstop against a model that loops
# producing text without tool calls -- ToolCallLimitMiddleware cannot see
# that failure shape at all.
_MODEL_CALL_LIMIT = 20


def _run_tool_call_ids(messages: list[BaseMessage], tool_name: str) -> list[str]:
    """Ids of every call to `tool_name` since the most recent `HumanMessage`.

    A limit scoped to "this run" must reset each turn instead of
    accumulating across a checkpointed thread's whole history -- counting
    from the end of `messages` back to the most recent `HumanMessage` is
    what gives a limit that scope.

    Parameters
    ----------
    messages : list of BaseMessage
        The agent state's message list.
    tool_name : str
        The tool to count calls for.

    Returns
    -------
    list of str
        Tool-call ids, oldest first.
    """
    ids: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            ids = []
        elif isinstance(message, AIMessage):
            ids.extend(
                call["id"]
                for call in message.tool_calls
                if call["name"] == tool_name and call["id"] is not None
            )
    return ids


def _tool_results(messages: list[BaseMessage], call_ids: list[str]) -> list[str]:
    """The `ToolMessage` content for each id in `call_ids`, in that order.

    Ids with no matching `ToolMessage` (e.g. a `Command`-returning tool that
    wrote no visible message this round) are skipped, so the caller sees
    only the results that actually exist to compare.
    """
    by_id = {
        message.tool_call_id: str(message.content)
        for message in messages
        if isinstance(message, ToolMessage)
    }
    return [by_id[call_id] for call_id in call_ids if call_id in by_id]


class ReadUrlCapMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Caps how many `read_url` calls the Researcher may make in one run.

    Without a cap the Researcher can spend its whole tool budget reading
    pages a search already found, instead of running the fresh searches the
    plan actually asks for. `max_calls=None` removes the cap.

    Defines **both** `wrap_tool_call` and `awrap_tool_call` (D3.1b).
    """

    def __init__(self, max_calls: int | None) -> None:
        super().__init__()
        self.max_calls = max_calls

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        refusal = self._refusal(request)
        if refusal is not None:
            return refusal
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        refusal = self._refusal(request)
        if refusal is not None:
            return refusal
        return await handler(request)

    def _refusal(self, request: ToolCallRequest) -> ToolMessage | None:
        if self.max_calls is None or request.tool_call["name"] != "read_url":
            return None

        call_id = request.tool_call["id"]
        prior_calls = [
            prior_id
            for prior_id in _run_tool_call_ids(request.state["messages"], "read_url")
            if prior_id != call_id
        ]
        if len(prior_calls) >= self.max_calls:
            return ToolMessage(
                content=(
                    f"ERROR: read_url call limit ({self.max_calls}) reached for "
                    "this run. Run a new web_search or knowledge_search instead "
                    "of reading another page."
                ),
                tool_call_id=call_id,
                name="read_url",
                status="error",
            )
        return None


class CriticVerificationMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]
):
    """Forces the Critic to verify at least one claim before it verdicts.

    `response_format=CritiqueResult` lets the model end the turn with a
    verdict and no verification call at all. If none of the three research
    tools ran earlier this turn, this middleware re-runs the model call once
    with `CRITIC_VERIFICATION_INSTRUCTION` appended. The retried response is
    returned as-is, whatever it contains -- one-shot, so a model that skips
    verification twice in a row cannot make this middleware loop.

    Defines **both** `wrap_model_call` and `awrap_model_call` (D3.1b).
    """

    def __init__(self, min_verification_calls: int = 1) -> None:
        super().__init__()
        self.min_verification_calls = min_verification_calls

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        response = handler(request)
        if self._calls_a_verification_tool(response) or self._verified_earlier(request):
            return response
        return handler(self._retry_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        response = await handler(request)
        if self._calls_a_verification_tool(response) or self._verified_earlier(request):
            return response
        return await handler(self._retry_request(request))

    @staticmethod
    def _retry_request(request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        # The retry forces a tool call at the API level (ChatOpenAI
        # translates "any" into "required"). Enforcement must ride
        # `model_settings`, not the `tool_choice` field, for the
        # `ProviderStrategy` bind path -- that path builds its bind kwargs
        # from the response format plus `model_settings` only and silently
        # drops `request.tool_choice` (`langchain/agents/factory.py:1391-1404`
        # on the installed 1.3.16). The field is still set too, for the
        # plain-tools bind path, where `model_settings` carrying
        # `tool_choice` would raise a duplicate-kwarg TypeError instead.
        retry_messages = [
            *request.messages,
            HumanMessage(content=CRITIC_VERIFICATION_INSTRUCTION),
        ]
        if isinstance(request.response_format, ProviderStrategy):
            return request.override(
                messages=retry_messages,
                model_settings={**request.model_settings, "tool_choice": "any"},
            )
        return request.override(messages=retry_messages, tool_choice="any")

    def _verified_earlier(self, request: ModelRequest[ContextT]) -> bool:
        # `AgentState["messages"]` is `list[AnyMessage]`, a Union alias;
        # `list` is invariant, so mypy rejects it as a `list[BaseMessage]`
        # argument even though every member of the union is one.
        messages = cast("list[BaseMessage]", request.state["messages"])
        verified_calls = sum(
            len(_run_tool_call_ids(messages, tool_name))
            for tool_name in _VERIFICATION_TOOLS
        )
        return verified_calls >= self.min_verification_calls

    @staticmethod
    def _calls_a_verification_tool(response: ModelResponse[ResponseT]) -> bool:
        return any(
            isinstance(message, AIMessage)
            and any(call["name"] in _VERIFICATION_TOOLS for call in message.tool_calls)
            for message in response.result
        )


class RoundStabilityMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]
):
    """Two deterministic stop signals beyond the Supervisor's iteration
    counter: *signal repetition* -- `critique` refused once
    `state["critic_gaps"]` repeats `state["previous_critic_gaps"]` -- and
    *candidate stability* -- `research` refused once its last two results in
    this run are byte-identical. Both compare the **structured** field or
    the raw tool result, never rendered prose, since `render_critique(...)`'s
    text can vary round to round even when the underlying gap does not.

    **Run-scoped, not thread-scoped.** `critic_gaps`/`previous_critic_gaps`
    can survive across questions in one checkpointed `thread_id`, so a
    comparison only fires once *this run* has produced at least two prior
    calls to the tool in question (`_run_tool_call_ids`, "since the most
    recent `HumanMessage`") -- state left over from an earlier question can
    never trigger a refusal on the first call of a new one.

    Defines **both** `wrap_tool_call` and `awrap_tool_call` (D3.1b). Keyed on
    `SUPERVISOR_DELEGATION_TOOLS` (D3.7), not hl10's `delegate_to_*` names.
    """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        refusal = self._refusal(request)
        if refusal is not None:
            return refusal
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        refusal = self._refusal(request)
        if refusal is not None:
            return refusal
        return await handler(request)

    def _refusal(self, request: ToolCallRequest) -> ToolMessage | None:
        name = request.tool_call["name"]
        if name not in SUPERVISOR_DELEGATION_TOOLS:
            return None

        messages = cast("list[BaseMessage]", request.state["messages"])
        call_id = request.tool_call["id"]
        prior = [
            prior_id
            for prior_id in _run_tool_call_ids(messages, name)
            if prior_id != call_id
        ]
        if len(prior) < 2:
            return None

        if name == "critique":
            gaps = request.state.get("critic_gaps")
            if gaps is None or gaps != request.state.get("previous_critic_gaps"):
                return None
            reason = "the Critic's gaps repeated the previous round's exactly"
        else:
            last_two = _tool_results(messages, prior[-2:])
            if len(last_two) < 2 or last_two[0] != last_two[1]:
                return None
            reason = "the Researcher's findings repeated the previous round's exactly"

        return ToolMessage(
            content=(
                f"ERROR: {name} call refused -- {reason}, so another round "
                "cannot discover anything new. Move on with what you have."
            ),
            tool_call_id=call_id,
            name=name,
            status="error",
        )


_SAVE_REPORT_NUDGE = (
    "The Critic's verdict was APPROVE and no save_report call has happened "
    "yet this run. Compose the final Markdown report and call save_report "
    "with it now."
)


class SaveReportGuardMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]
):
    """One-shot nudge toward `save_report` after an APPROVE the model is
    about to leave unsaved.

    Fires only when **all four** hold on the model's about-to-end response:
    it carries no tool calls, `state["verdict"] == "APPROVE"`, a `critique`
    result exists since the most recent `HumanMessage` (not merely
    `state["verdict"]`, which a checkpointed thread carries across
    questions, the same trap `RoundStabilityMiddleware` guards against), and
    no `save_report` call exists since that same boundary. **A standing
    REVISE is never forced** -- a run that exhausted its revision budget has
    no approved content, and forcing a save there would ship a report its
    own Critic rejected.

    One re-request with `tool_choice="any"`, whatever it returns. The
    Supervisor carries no `response_format`, so it is always on the
    plain-tools bind path -- `request.override(tool_choice=...)` is the
    correct channel here (unlike `CriticVerificationMiddleware`'s
    `ProviderStrategy` path, where the same field is measured to be a no-op
    and `model_settings` must carry it instead).

    Defines **both** `wrap_model_call` and `awrap_model_call` (D3.1b). Keyed
    on `SUPERVISOR_DELEGATION_TOOLS`'s `"critique"` (D3.7), not hl10's
    `delegate_to_critic`.
    """

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        response = handler(request)
        if not self._should_nudge(request, response):
            return response
        return handler(self._retry_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        response = await handler(request)
        if not self._should_nudge(request, response):
            return response
        return await handler(self._retry_request(request))

    def _should_nudge(
        self, request: ModelRequest[ContextT], response: ModelResponse[ResponseT]
    ) -> bool:
        if self._has_tool_calls(response):
            return False
        if request.state.get("verdict") != "APPROVE":
            return False
        messages = cast("list[BaseMessage]", request.state["messages"])
        if not _run_tool_call_ids(messages, "critique"):
            return False
        if _run_tool_call_ids(messages, "save_report"):
            return False
        return True

    @staticmethod
    def _retry_request(request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        retry_messages = [
            *request.messages,
            HumanMessage(content=_SAVE_REPORT_NUDGE),
        ]
        return request.override(messages=retry_messages, tool_choice="any")

    @staticmethod
    def _has_tool_calls(response: ModelResponse[ResponseT]) -> bool:
        return any(
            isinstance(message, AIMessage) and bool(message.tool_calls)
            for message in response.result
        )


class RevisionCapMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]
):
    """The Supervisor's per-question revision cap (stage-4 spec D4.5).

    Refuses a `critique` call once `max_revisions + 1` calls have already
    been emitted since the most recent `HumanMessage`. Counts **emitted**
    calls (`_run_tool_call_ids`), not executed ones -- an earlier draft of
    this design counted `ToolMessage`s instead, on the reasoning that a
    stability-refused `critique` should not consume a slot. That reasoning
    was false (a refusal *is* a `ToolMessage`, `_tool_results` indexes it
    with no status filter) and additionally blind to same-batch calls
    (`ToolCallRequest.state` is fixed at tool-node entry, so two `critique`
    calls in one `AIMessage` are both invisible to an executed count at the
    time either one runs). The emitted count has neither defect and matches
    `RoundStabilityMiddleware`'s own counting exactly.

    Scoped to "since the most recent `HumanMessage`", not to the checkpointed
    thread: neither `ToolCallLimitMiddleware` mode fits here.
    `thread_limit` never resets, starving the second question of a session;
    `run_limit` lives in an `UntrackedValue` channel that resets on every
    `Command(resume=...)`, so a human pause silently refills the budget
    (both measured against the installed langgraph).

    Defines **both** `wrap_tool_call` and `awrap_tool_call` (D3.1b).
    """

    def __init__(self, max_revisions: int) -> None:
        super().__init__()
        self.max_revisions = max_revisions

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        refusal = self._refusal(request)
        if refusal is not None:
            return refusal
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        refusal = self._refusal(request)
        if refusal is not None:
            return refusal
        return await handler(request)

    def _refusal(self, request: ToolCallRequest) -> ToolMessage | None:
        if request.tool_call["name"] != "critique":
            return None

        messages = cast("list[BaseMessage]", request.state["messages"])
        call_id = request.tool_call["id"]
        limit = self.max_revisions + 1
        if emitted_critique_count(messages, exclude_call_id=call_id) < limit:
            return None

        return ToolMessage(
            content=(
                f"ERROR: critique call refused -- the revision cap "
                f"({self.max_revisions} revision(s), {limit} critique "
                "call(s) total) is reached for this question. Compose the "
                "final report with the findings and verdict you already "
                "have and call save_report."
            ),
            tool_call_id=call_id,
            name="critique",
            status="error",
        )


def emitted_critique_count(
    messages: list[BaseMessage], *, exclude_call_id: str | None = None
) -> int:
    """How many `critique` calls have been emitted since the most recent
    `HumanMessage`, optionally excluding one in-flight call id.

    Shared between `RevisionCapMiddleware` (the cap itself) and
    `supervisor.py`'s verdict guard (D4.18), so both read "rounds remain"
    off the same counter -- one lifecycle on the supervisor path, not two.
    """
    ids = _run_tool_call_ids(messages, "critique")
    if exclude_call_id is not None:
        ids = [call_id for call_id in ids if call_id != exclude_call_id]
    return len(ids)


class SaveReportVerdictGuardMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]
):
    """Refuses `save_report` on a non-APPROVE verdict while revision rounds
    remain (stage-4 spec D4.18).

    Refuses when, since the most recent `HumanMessage`: no non-error
    `critique` `ToolMessage` exists at all, **or** the standing
    `state["verdict"]` is not `"APPROVE"` and `emitted_critique_count` is
    below `max_revisions + 1`.

    A **non-error** critique `ToolMessage` is required before the verdict is
    honoured -- an *emitted* gate is not enough: a critique whose execution
    raises (`ToolErrorMiddleware`'s `ERROR:` message, no `Command` state
    write) would leave the gate satisfied while `state["verdict"]` still
    holds a *previous question's* checkpointed `APPROVE`, waving a premature
    save through on a stale verdict -- the exact trap
    `SaveReportGuardMiddleware`'s own docstring names for a different field.

    Once the cap is exhausted (`emitted_critique_count >= max_revisions + 1`)
    a REVISE verdict may still save: `_S1` rule 5 orders a save once the
    model "stopped revising for any reason", and a strict
    `verdict != "APPROVE"` refusal would deadlock against that -- the router
    would send the model back with no report ever produced.

    The refusal `ToolMessage`'s own text is the only nudge back toward
    `critique` while rounds remain; `SaveReportGuardMiddleware` nudges only
    on an APPROVE-unsaved response, not this case.

    Defines **both** `wrap_tool_call` and `awrap_tool_call` (D3.1b).
    """

    def __init__(self, max_revisions: int) -> None:
        super().__init__()
        self.max_revisions = max_revisions

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        refusal = self._refusal(request)
        if refusal is not None:
            return refusal
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        refusal = self._refusal(request)
        if refusal is not None:
            return refusal
        return await handler(request)

    def _refusal(self, request: ToolCallRequest) -> ToolMessage | None:
        if request.tool_call["name"] != "save_report":
            return None

        messages = cast("list[BaseMessage]", request.state["messages"])
        call_id = request.tool_call["id"]

        critique_ids = _run_tool_call_ids(messages, "critique")
        critique_results = _tool_results(messages, critique_ids)
        non_error_verdicted = any(
            not result.startswith("ERROR:") for result in critique_results
        )
        if not non_error_verdicted:
            return self._refusal_message(
                call_id,
                "no completed critique exists yet this question -- call "
                "critique first",
            )

        if request.state.get("verdict") == "APPROVE":
            return None

        limit = self.max_revisions + 1
        if emitted_critique_count(messages) < limit:
            return self._refusal_message(
                call_id,
                "the verdict is not APPROVE and revision rounds remain -- "
                "call critique again before saving",
            )
        return None

    @staticmethod
    def _refusal_message(call_id: str | None, reason: str) -> ToolMessage:
        return ToolMessage(
            content=f"ERROR: save_report call refused -- {reason}.",
            tool_call_id=call_id,
            name="save_report",
            status="error",
        )


def truncate_for_span(text: str, max_length: int) -> str:
    """Cap `text` at `max_length`, marking that it was cut.

    Used by both this module's `TracingMiddleware` (`tool.args`/
    `tool.result`) and `tools.py`'s `knowledge_search` (`retrieval.chunks`,
    `docs/specs/stage-5.md` D5.8) -- both infra, so this lives here rather
    than in `observability.py` (obs), which infra may not import
    (`tests/test_layering.py`'s `MAY_IMPORT[INFRA]` has no `OBS` entry).
    """
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}...[truncated, {len(text)} chars total]"


class TracingMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """One span per model call (`model.<role>`, token usage and cost as
    attributes) and one per tool call (`tool.<name>`, args and result,
    both truncated) -- appended to every sub-agent's and the Supervisor's
    own `agent_middleware()` stack (stage 5, `docs/specs/stage-5.md`,
    D5.10). Reaches OpenTelemetry's ambient global tracer directly
    (`trace.get_tracer(__name__)`), never `observability.py` itself (D5.3)
    -- the span this creates nests under whatever `main.py`'s per-turn
    `repl.question` span or a delegation-site `agent.<role>` span (D5.9)
    currently has open, via the same `contextvars`-backed propagation
    OpenTelemetry already provides with no explicit wiring.

    Defines **both** `wrap_model_call`/`awrap_model_call` and
    `wrap_tool_call`/`awrap_tool_call` (D3.1b).
    """

    _MAX_PAYLOAD_LENGTH = 2000  # Settings.max_span_payload_length's default;
    # this class takes a plain int rather than a Settings object, since a
    # middleware instance is built once at agent-construction time, before
    # any per-run scoping exists -- callers that need the configured value
    # pass it in.

    def __init__(
        self, *, role: str, max_payload_length: int = _MAX_PAYLOAD_LENGTH
    ) -> None:
        super().__init__()
        self.role = role
        self.max_payload_length = max_payload_length

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"model.{self.role}") as span:
            response = handler(request)
            self._record_usage(span, request, response)
            return response

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"model.{self.role}") as span:
            response = await handler(request)
            self._record_usage(span, request, response)
            return response

    def _record_usage(
        self,
        span: Any,
        request: ModelRequest[ContextT],
        response: ModelResponse[ResponseT],
    ) -> None:
        usage = next(
            (
                message.usage_metadata
                for message in response.result
                if isinstance(message, AIMessage) and message.usage_metadata
            ),
            None,
        )
        if usage is None:
            return
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        model_name = getattr(request.model, "model_name", None)
        if model_name is None:
            return
        cost = compute_cost(model_name, input_tokens, output_tokens)
        if cost is not None:
            span.set_attribute("gen_ai.usage.cost_usd", cost)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"tool.{request.tool_call['name']}") as span:
            self._record_args(span, request)
            result = handler(request)
            self._record_result(span, result)
            return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"tool.{request.tool_call['name']}") as span:
            self._record_args(span, request)
            result = await handler(request)
            self._record_result(span, result)
            return result

    def _record_args(self, span: Any, request: ToolCallRequest) -> None:
        # A raw dict is silently dropped by OTel's own attribute validation
        # (measured against the installed opentelemetry-sdk) -- always a
        # JSON string, never the dict itself.
        args_json = json.dumps(request.tool_call.get("args", {}), default=str)
        span.set_attribute(
            "tool.args", truncate_for_span(args_json, self.max_payload_length)
        )

    def _record_result(self, span: Any, result: ToolMessage | Command[Any]) -> None:
        if isinstance(result, ToolMessage):
            span.set_attribute(
                "tool.result",
                truncate_for_span(str(result.content), self.max_payload_length),
            )


def _tool_error_to_message(exc: Exception, request: ToolCallRequest) -> str:
    """Names the exception type rather than echoing its message, keeping
    internal detail out of the model's context, while preserving this
    project's "an error is data" invariant."""
    return f"ERROR: {request.tool_call['name']} failed ({type(exc).__name__})"


def agent_middleware(
    *,
    tool_call_limit: int,
    role: str,
    model_call_limit: int = _MODEL_CALL_LIMIT,
    tool_exit_behavior: Literal["continue", "error", "end"] = "continue",
) -> list[AgentMiddleware[Any, Any, Any]]:
    """The shared middleware stack every sub-agent's caller assembles.

    Parameters
    ----------
    tool_call_limit : int
        Per-run tool-call budget, e.g. `settings.researcher_max_tool_calls`.
    role : str
        Passed straight through to `TracingMiddleware` for its span names
        (`model.<role>`) -- required, no default (stage 5,
        `docs/specs/stage-5.md` D5.10: a `TracingMiddleware` silently
        mislabelled by a missing role is worse than a loud `TypeError` at
        every one of this function's six real call sites).
    model_call_limit : int, default `_MODEL_CALL_LIMIT`
        Per-run model-call budget -- the only bound on a model that loops
        producing text without tool calls, which `ToolCallLimitMiddleware`
        cannot see.
    tool_exit_behavior : "continue" | "error" | "end", default "continue"
        Forwarded to `ToolCallLimitMiddleware`. The three sub-agents keep
        the library default; the Supervisor (stage-4 spec D4.15) passes
        `"end"` -- with `"continue"`, a blocked call stays visible in
        `last_ai_message.tool_calls` and `HumanInTheLoopMiddleware` still
        interrupts for it (measured: ordering the Supervisor's middleware
        list cannot prevent this on its own), asking a human to approve a
        write that cannot happen. `"end"` jumps past the remaining
        `after_model` hooks instead, so an exhausted budget ends the run.

    Returns
    -------
    list of AgentMiddleware
        `ModelCallLimit -> ToolCallLimit -> ToolError -> ToolRetry ->
        ModelRetry -> Tracing`, list order = outermost first for the first
        five (`TracingMiddleware` is appended last, innermost -- it wraps
        the actual model/tool call, so it sees the real request/response,
        not a limit-refused stand-in). Callers append their own
        agent-specific middleware (`ReadUrlCapMiddleware`,
        `CriticVerificationMiddleware`) after this list, since under D3.2
        agent factories no longer assemble any part of their own stack.

    Notes
    -----
    `ToolRetryMiddleware` is constructed with `on_failure="error"`: without
    it the outer `ToolErrorMiddleware`'s handler never fires, because the
    library default (`on_failure="continue"`) never re-raises. First entry
    in the list is outermost for both the model-call chain
    (`langchain/agents/factory.py:263-268`) and the tool-call chain
    (`:658-674`), which is what lets an exhausted retry's exception reach
    `ToolErrorMiddleware`.
    """
    return [
        ModelCallLimitMiddleware(run_limit=model_call_limit),
        ToolCallLimitMiddleware(
            run_limit=tool_call_limit, exit_behavior=tool_exit_behavior
        ),
        ToolErrorMiddleware(_tool_error_to_message),
        ToolRetryMiddleware(on_failure="error", jitter=True),
        ModelRetryMiddleware(),
        TracingMiddleware(role=role),
    ]
