"""Shared eval-tier fixtures and helpers (stage 7).

`live_settings` is session-scoped on purpose: `observability.configure_observability`
raises `RuntimeError` on a second call in one process
(`observability.py`'s `_CONFIGURED` sentinel), and every eval-tier test
in `deepeval test run tests/` shares one process. One provider, one
tmp-directory span dump root, reused by every live case in the session.
"""

from __future__ import annotations

import json
from typing import Iterator

import pytest

from config import Settings, load_settings
from evals.deepeval_model import OpenRouterModel
from paths import PROJECT_ROOT
from tests.live_agents import configured_for_eval, eval_settings

_GOLDEN_DATASET_PATH = PROJECT_ROOT / "tests" / "golden_dataset.json"
_INDEX_MANIFEST_PATH = PROJECT_ROOT / "index" / "manifest.json"


@pytest.fixture(scope="session")
def live_settings(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    """One real `Settings`, span dump redirected off the project's real
    `runs/` (`insights.md`'s stage-5 pollution mistake, avoided here the
    same way), with observability configured exactly once for the session.
    """
    settings = eval_settings(
        runs_dir=str(tmp_path_factory.mktemp("eval-runs")),
        settings=load_settings(),
    )
    with configured_for_eval(settings):
        yield settings


def golden_input(case_id: str) -> str:
    """The `input` field of one `tests/golden_dataset.json` case, by id."""
    cases = {
        case["id"]: case
        for case in json.loads(_GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    }
    return str(cases[case_id]["input"])


def judge_model(settings: Settings) -> OpenRouterModel:
    """The judge model, resolved the same way every eval-tier metric does:
    `Settings.judge_model_name` if set, else the shared agent model."""
    return OpenRouterModel(
        settings.judge_model_name or settings.model_name,
        api_key=settings.openrouter_api_key.get_secret_value(),
    )


def skip_without_index() -> None:
    """Skip an eval-tier test that needs a live index this CI job never
    builds (`index/` is gitignored, the `evals` job runs no `ingest.py`
    step -- the same defect stage 6 found for its own freshness test,
    `docs/specs/stage-7.md` D7.9)."""
    if not _INDEX_MANIFEST_PATH.is_file():
        pytest.skip(
            f"{_INDEX_MANIFEST_PATH} does not exist -- run `python ingest.py` "
            "locally before the eval tier"
        )


def fixture_text(name: str) -> str:
    """Read a fixture file under `evals/fixtures/` by name."""
    path = PROJECT_ROOT / "evals" / "fixtures" / name
    return path.read_text(encoding="utf-8")
