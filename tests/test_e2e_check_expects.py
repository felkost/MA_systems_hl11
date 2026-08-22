"""`check_expects`/`skip_if_poisoned_chunk_absent` (stage 9a,
`docs/specs/stage-9a.md` D9a.11/D9a.12).

Offline: hand-built `ToolCall` lists and `LLMTestCase`s, matching the shape
`tests/test_tools.py`'s own `_expect` helper already uses -- no live call.
"""

from __future__ import annotations

import pytest
from deepeval.test_case import LLMTestCase, ToolCall

from tests.test_e2e import check_expects, skip_if_poisoned_chunk_absent


def _call(name: str) -> ToolCall:
    return ToolCall(name=name, input_parameters=None)


def _case(tools_called: list[ToolCall], actual_output: str = "") -> LLMTestCase:
    return LLMTestCase(
        input="q", actual_output=actual_output, tools_called=tools_called
    )


def test_must_cite_fails_only_when_no_retrieval_tool_was_called() -> None:
    case = {"id": "x", "expects": {"must_cite": True}}

    failures_without_retrieval = check_expects(case, _case([_call("web_search")] * 0))
    assert any("must_cite" in f for f in failures_without_retrieval)

    failures_with_retrieval = check_expects(case, _case([_call("web_search")]))
    assert not any("must_cite" in f for f in failures_with_retrieval)


def test_out_of_scope_fails_only_when_a_delegation_tool_ran() -> None:
    case = {"id": "x", "expects": {"out_of_scope": True}}

    ok = check_expects(case, _case([_call("web_search")]))
    assert not any("out_of_scope" in f for f in ok)

    bad = check_expects(case, _case([_call("research"), _call("save_report")]))
    assert any("out_of_scope" in f for f in bad)


def test_injection_echo_check_fires_only_for_its_own_case_id() -> None:
    injection_case = {"id": "adversarial-indirect-injection", "expects": {}}
    other_case = {"id": "adversarial-direct-jailbreak", "expects": {}}

    injected = check_expects(injection_case, _case([], actual_output="INJECTED"))
    assert any("refuses_injection" in f for f in injected)

    not_injected = check_expects(
        injection_case, _case([], actual_output="Here is a real summary.")
    )
    assert not any("refuses_injection" in f for f in not_injected)

    unrelated_case_with_injected_text = check_expects(
        other_case, _case([], actual_output="INJECTED")
    )
    assert not any("refuses_injection" in f for f in unrelated_case_with_injected_text)


def test_a_case_with_no_expects_field_returns_no_failures() -> None:
    assert check_expects({"id": "x"}, _case([])) == []


def test_skip_if_poisoned_chunk_absent_skips_when_marker_missing() -> None:
    case = {"id": "adversarial-poisoned-knowledge-base", "needs_poisoned_index": True}
    test_case = LLMTestCase(
        input="q", actual_output="", retrieval_context=["unrelated chunk"]
    )

    with pytest.raises(pytest.skip.Exception):
        skip_if_poisoned_chunk_absent(case, test_case)


def test_skip_if_poisoned_chunk_absent_does_nothing_when_marker_present() -> None:
    case = {"id": "adversarial-poisoned-knowledge-base", "needs_poisoned_index": True}
    test_case = LLMTestCase(
        input="q",
        actual_output="",
        retrieval_context=["... Cascading Relay Consensus ..."],
    )

    skip_if_poisoned_chunk_absent(case, test_case)  # must not raise/skip


def test_skip_if_poisoned_chunk_absent_does_nothing_for_an_ordinary_case() -> None:
    case = {"id": "core-single-vs-multi-agent"}
    test_case = LLMTestCase(input="q", actual_output="", retrieval_context=[])

    skip_if_poisoned_chunk_absent(case, test_case)  # must not raise/skip
