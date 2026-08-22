"""`record_case_cost`/`load_case_costs` round trip (stage 9a, D9a.8).

Offline: writes into a real `runs/<eval_run_id>/` (via `paths.run_dir`),
using a throwaway uuid so it never collides with a real eval run, and
cleans up after itself -- `runs/` is gitignored, but a stray directory left
behind by a test is exactly the "runs/ pollution" mistake insights.md
already records once (stage 5).
"""

from __future__ import annotations

import shutil
from uuid import uuid4

import paths
from tests.conftest import load_case_costs, record_case_cost


def test_records_round_trip_in_write_order() -> None:
    eval_run_id = str(uuid4())
    try:
        record_case_cost(
            eval_run_id,
            case_id="core-a",
            run_id="run-1",
            agent_cost_usd=0.01,
            judge_cost_usd=0.02,
        )
        record_case_cost(
            eval_run_id,
            case_id="core-b",
            run_id="run-2",
            agent_cost_usd=0.03,
            judge_cost_usd=0.04,
        )

        records = load_case_costs(eval_run_id)

        assert records == [
            {
                "case_id": "core-a",
                "run_id": "run-1",
                "agent_cost_usd": 0.01,
                "judge_cost_usd": 0.02,
            },
            {
                "case_id": "core-b",
                "run_id": "run-2",
                "agent_cost_usd": 0.03,
                "judge_cost_usd": 0.04,
            },
        ]
    finally:
        shutil.rmtree(paths.run_dir(eval_run_id), ignore_errors=True)


def test_load_case_costs_returns_empty_list_for_an_unknown_run() -> None:
    assert load_case_costs(str(uuid4())) == []
