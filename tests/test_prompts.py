"""hl8's registry mechanism carrying hl10's prompt text
(`docs/specs/stage-3.md`, "Why the prompt text must be hl10's").

hl8's own prompt text is wrong for this project on two counts checked here:
it instructs the Researcher to call a tool this project removed
(`graph_search`), and its Supervisor prompt has no out-of-scope path for
D3.3's `in_scope` field to be read by. Both are pinned as standing rules,
not one-time review comments.
"""

from __future__ import annotations

from datetime import date

import pytest

from prompts import (
    build_composer_prompt,
    build_critic_prompt,
    build_planner_prompt,
    build_researcher_prompt,
    build_supervisor_prompt,
)


def test_build_planner_prompt_looks_up_a_registered_version() -> None:
    text = build_planner_prompt("p1")
    assert "Planner" in text


def test_build_planner_prompt_raises_on_unknown_version() -> None:
    with pytest.raises(KeyError, match="p1"):
        build_planner_prompt("p99")


def test_build_researcher_prompt_raises_on_unknown_version() -> None:
    with pytest.raises(KeyError):
        build_researcher_prompt("r99")


def test_build_critic_prompt_raises_on_unknown_version() -> None:
    with pytest.raises(KeyError):
        build_critic_prompt("c99", today=date(2026, 8, 22))


def test_build_supervisor_prompt_raises_on_unknown_version() -> None:
    with pytest.raises(KeyError):
        build_supervisor_prompt("s99")


def test_build_critic_prompt_injects_the_date() -> None:
    text = build_critic_prompt("c2", today=date(2026, 8, 22))
    assert "2026-08-22" in text


def test_researcher_prompt_never_mentions_graph_search() -> None:
    text = build_researcher_prompt("r1")
    assert "graph_search" not in text


def test_supervisor_prompt_has_an_out_of_scope_path() -> None:
    text = build_supervisor_prompt("s1")
    assert "out of scope" in text.lower() or "in_scope" in text.lower()


def test_critic_prompt_c1_and_c2_both_registered() -> None:
    c1 = build_critic_prompt("c1", today=date(2026, 8, 22))
    c2 = build_critic_prompt("c2", today=date(2026, 8, 22))
    assert c1 != c2


def test_build_composer_prompt_raises_on_unknown_version() -> None:
    with pytest.raises(KeyError):
        build_composer_prompt("w99")


def test_composer_prompt_is_a_separate_registry_from_the_supervisor() -> None:
    # D4.4: a composer prompt must not be reachable through
    # `build_supervisor_prompt` -- registering it in `s*` would let
    # `SUPERVISOR_PROMPT_VERSION="w1"` hand the agent-as-tool Supervisor a
    # composer prompt with no delegation rules.
    composer = build_composer_prompt("w1")
    assert "plan, research, critique" not in composer
    with pytest.raises(KeyError):
        build_supervisor_prompt("w1")
