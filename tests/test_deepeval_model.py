"""Offline tests for `evals.deepeval_model.OpenRouterModel`.

No network call: every test injects an `httpx.MockTransport` through the
wrapper's `transport=` constructor parameter, so this suite runs in the gate
tier. The one live call this module needs (a real OpenRouter round trip) is a
separate, explicitly-run smoke check -- never part of
`pytest -m "not smoke and not eval"`.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from evals.deepeval_model import OpenRouterModel


class _Verdict(BaseModel):
    verdict: str
    score: float


def _model(response_content: str, usage: dict[str, int]) -> OpenRouterModel:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": response_content}}],
                "usage": usage,
            },
        )

    return OpenRouterModel(
        model_name="openai/gpt-4.1-mini",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )


def test_generate_returns_message_content() -> None:
    model = _model(
        "plain text answer",
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    )

    result = model.generate("say something")

    assert result == "plain text answer"


def test_generate_records_usage_because_deepeval_will_not() -> None:
    """DeepEval never calls `_accrue_cost` for a custom (non-native) model --
    `initialize_model` marks it `using_native_model=False` -- so this wrapper
    is the only place that ever sees the provider's real token counts."""
    model = _model(
        "ok", {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168}
    )

    model.generate("anything")

    assert model.last_usage == {
        "prompt_tokens": 123,
        "completion_tokens": 45,
        "total_tokens": 168,
    }


def test_generate_with_schema_returns_validated_instance() -> None:
    payload = json.dumps({"verdict": "PASS", "score": 0.9})
    model = _model(
        payload, {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    )

    result = model.generate_with_schema("judge this", schema=_Verdict)

    assert isinstance(result, _Verdict)
    assert result.verdict == "PASS"
    assert result.score == 0.9


@pytest.mark.asyncio
async def test_a_generate_with_schema_returns_validated_instance() -> None:
    payload = json.dumps({"verdict": "FAIL", "score": 0.1})
    model = _model(
        payload, {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    )

    result = await model.a_generate_with_schema("judge this", schema=_Verdict)

    assert isinstance(result, _Verdict)
    assert result.verdict == "FAIL"


def test_get_model_name_returns_the_configured_id() -> None:
    model = OpenRouterModel(model_name="openai/gpt-4.1-mini", api_key="test-key")

    assert model.get_model_name() == "openai/gpt-4.1-mini"
