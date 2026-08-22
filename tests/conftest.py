"""Shared eval-tier fixtures and helpers (stage 7; extended stage 9a,
`docs/specs/stage-9a.md`).

`live_settings` is session-scoped on purpose: `observability.configure_observability`
raises `RuntimeError` on a second call in one process
(`observability.py`'s `_CONFIGURED` sentinel), and every eval-tier test
in `deepeval test run tests/` shares one process. One provider, one
tmp-directory span dump root, reused by every live case in the session.

`eval_run_id`/`e2e_judge_model`/`fixture_base_url`/`record_case_cost` are
stage 9a's own additions, session-scoped for the same reason: one
`OpenRouterModel` instance shared across all 15 golden-dataset cases' 3
metrics is what lets its `usage_log` accumulate a real, measurable judge-cost
total (D9a.7), and one fixture HTTP server (D9a.1) is enough for the one
case that needs it.
"""

from __future__ import annotations

import json
from typing import Any, Iterator
from uuid import uuid4

import pytest

import paths
from config import Settings, load_settings
from evals.deepeval_model import OpenRouterModel
from paths import PROJECT_ROOT
from tests.fixture_server import run_fixture_server
from tests.live_agents import configured_for_eval, eval_settings

_GOLDEN_DATASET_PATH = PROJECT_ROOT / "tests" / "golden_dataset.json"
_INDEX_MANIFEST_PATH = PROJECT_ROOT / "index" / "manifest.json"
_FIXTURES_DIR = PROJECT_ROOT / "evals" / "fixtures"


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


def _golden_cases_by_id() -> dict[str, dict[str, Any]]:
    return {
        case["id"]: case
        for case in json.loads(_GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    }


def golden_input(case_id: str) -> str:
    """The `input` field of one `tests/golden_dataset.json` case, by id."""
    return str(_golden_cases_by_id()[case_id]["input"])


def all_case_ids() -> list[str]:
    """Every case id in `tests/golden_dataset.json`, in file order -- the
    parametrize source for `tests/test_e2e.py::test_golden_dataset` (stage
    9a, R4b's "усі 15")."""
    return list(_golden_cases_by_id().keys())


def golden_case(case_id: str) -> dict[str, Any]:
    """The full row of one `tests/golden_dataset.json` case, by id (stage
    9a) -- `golden_input` only ever needed `input`; `resolve_case_input`
    (below) and stage 9a's own `check_expects`/`skip_if_poisoned_chunk_absent`
    (`tests/test_e2e.py`) need `category`, `expected_output`, `fixture`,
    `needs_poisoned_index` and `expects` too.

    Raises
    ------
    KeyError
        No case with this id exists.
    """
    return _golden_cases_by_id()[case_id]


def resolve_case_input(case: dict[str, Any], fixture_base_url: str) -> str:
    """`case["input"]`, with `{fixture_url}` substituted for the one case
    that carries an HTTP-fetchable `fixture` (stage 9a, D9a.3).

    `needs_poisoned_index` is excluded even though that case also carries a
    `fixture` key: its fixture is for the knowledge index, never fetched
    over HTTP, and its own `input` string has no `{fixture_url}` placeholder
    to begin with (verified against `tests/golden_dataset.json` directly).
    The guard exists so a future case that adds both a placeholder and
    `needs_poisoned_index` is not silently mis-resolved.
    """
    fixture_name = case.get("fixture")
    if fixture_name is None or case.get("needs_poisoned_index"):
        return str(case["input"])
    return str(case["input"]).format(fixture_url=f"{fixture_base_url}/{fixture_name}")


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


@pytest.fixture(scope="session")
def fixture_base_url() -> Iterator[str]:
    """One local HTTP server over `evals/fixtures/`, for the whole session
    (stage 9a, D9a.1). Only `adversarial-indirect-injection` uses it
    (`resolve_case_input`), but it is session-scoped rather than
    function-scoped for the same reason `live_settings` is: cheap to start
    once, and nothing about it is per-case state."""
    with run_fixture_server(_FIXTURES_DIR) as url:
        yield url


@pytest.fixture(scope="session")
def eval_run_id() -> str:
    """One id for this whole e2e evaluation session (stage 9a, D9a.8) --
    a different namespace from each case's own Supervisor `run_id` (D8.9).
    Groups this stage's own artefacts under `runs/<eval_run_id>/`."""
    return str(uuid4())


@pytest.fixture(scope="session")
def e2e_judge_model(live_settings: Settings) -> OpenRouterModel:
    """One judge model instance, shared across all 15 golden-dataset cases'
    3 metrics, with a real `usage_log` (stage 9a, D9a.7) -- sharing one
    instance is what lets the log accumulate a real total; a fresh instance
    per call (`judge_model()`, above, kept for stages 7/8's own per-test-body
    reads) would reset nothing on `usage_log` itself, but `test_e2e.py`
    needs the *same* list back after 45 measurements, not 45 separate ones.
    """
    return OpenRouterModel(
        live_settings.judge_model_name or live_settings.model_name,
        api_key=live_settings.openrouter_api_key.get_secret_value(),
        usage_log=[],
    )


def record_case_cost(
    eval_run_id: str,
    *,
    case_id: str,
    run_id: str,
    agent_cost_usd: float,
    judge_cost_usd: float,
) -> None:
    """Append one case's measured cost to `runs/<eval_run_id>/case-costs.jsonl`
    (stage 9a, D9a.8).

    Written incrementally, one line per case, while the data is still in
    memory in the process that produced it -- `evals/summarize_e2e.py` runs
    later, in a separate process, and reads this file rather than trying to
    reconstruct cost from state that will not have survived (the mistake
    this stage's own adversarial review caught in an earlier draft).
    """
    run_dir = paths.run_dir(eval_run_id)
    record = {
        "case_id": case_id,
        "run_id": run_id,
        "agent_cost_usd": agent_cost_usd,
        "judge_cost_usd": judge_cost_usd,
    }
    with (run_dir / "case-costs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_case_costs(eval_run_id: str) -> list[dict[str, Any]]:
    """Read back every record `record_case_cost` wrote for one eval run.

    Does not create `runs/<eval_run_id>/` as a side effect (unlike
    `paths.run_dir`) -- a loader for a run that never wrote anything must
    not leave an empty directory behind.
    """
    path = paths.resolve("runs") / eval_run_id / "case-costs.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
