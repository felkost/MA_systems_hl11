"""`Settings.critic_prompt_version` defaults to `"c2"`, the version
whose verdict/booleans coupling the assignment's own Critique Quality
GEval steps measure against.

Every test in this file passes `openrouter_api_key` explicitly:
`Settings.model_validate({})` only constructs on a machine with an untracked
`.env` -- a "green by accident" trap, caught and avoided here.
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr

from config import Settings


def _settings(**overrides: Any) -> Settings:
    return Settings(openrouter_api_key=SecretStr("test-key"), **overrides)


def test_critic_prompt_defaults_to_c2() -> None:
    assert _settings().critic_prompt_version == "c2"


def test_prompt_versions_have_sensible_defaults() -> None:
    settings = _settings()
    assert settings.planner_prompt_version == "p2"
    assert settings.researcher_prompt_version == "r2"
    assert settings.supervisor_prompt_version == "s2"


def test_prompt_versions_are_overridable() -> None:
    settings = _settings(critic_prompt_version="c1")
    assert settings.critic_prompt_version == "c1"
