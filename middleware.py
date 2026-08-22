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

from collections.abc import Awaitable, Callable
from typing import Any, cast

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


def _tool_error_to_message(exc: Exception, request: ToolCallRequest) -> str:
    """Names the exception type rather than echoing its message, keeping
    internal detail out of the model's context, while preserving this
    project's "an error is data" invariant."""
    return f"ERROR: {request.tool_call['name']} failed ({type(exc).__name__})"


def agent_middleware(
    *, tool_call_limit: int, model_call_limit: int = _MODEL_CALL_LIMIT
) -> list[AgentMiddleware[Any, Any, Any]]:
    """The shared middleware stack every sub-agent's caller assembles.

    Parameters
    ----------
    tool_call_limit : int
        Per-run tool-call budget, e.g. `settings.researcher_max_tool_calls`.
    model_call_limit : int, default `_MODEL_CALL_LIMIT`
        Per-run model-call budget -- the only bound on a model that loops
        producing text without tool calls, which `ToolCallLimitMiddleware`
        cannot see.

    Returns
    -------
    list of AgentMiddleware
        `ModelCallLimit -> ToolCallLimit -> ToolError -> ToolRetry ->
        ModelRetry`, list order = outermost first. Callers append their own
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
        ToolCallLimitMiddleware(run_limit=tool_call_limit),
        ToolErrorMiddleware(_tool_error_to_message),
        ToolRetryMiddleware(on_failure="error", jitter=True),
        ModelRetryMiddleware(),
    ]
