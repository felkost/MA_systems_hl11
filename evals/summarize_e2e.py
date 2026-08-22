"""Turn one `deepeval test run` invocation's own persisted results, plus
this project's own persisted per-case cost log, into `runs/<id>/summary.md`
and `runs/<id>/eval-results.json` (stage 9a, `docs/specs/stage-9a.md` D9a.8).

Run manually, **after** a `deepeval test run` invocation that included
`tests/test_e2e.py` -- `wrap_up_test_run()`, which (re)writes DeepEval's own
`LATEST_FULL_TEST_RUN_FILE_PATH`, is called from exactly one code path this
project's `assert_test`-based usage ever reaches: the `deepeval test run`
CLI command itself, in-process, right after `pytest.main()` returns (stage
9a's own SDK measurement, N6). A bare `pytest tests/test_e2e.py` never
writes it.

Two data sources, written by two different processes, at two different
times:

- DeepEval's own file -- every case's per-metric score/threshold/success.
- `runs/<eval_run_id>/case-costs.jsonl` -- this project's own per-case
  agent/judge cost, written incrementally by `tests/test_e2e.py` itself
  *during* the pytest session, while the data was still in memory (D9a.7's
  judge-cost accumulator does not survive past the pytest process on its
  own; this file is what carries its result across that boundary).

Neither source is ever written by this module -- it only reads and joins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepeval.test_run.test_run import LATEST_FULL_TEST_RUN_FILE_PATH

import paths

_E2E_METRIC_NAMES = frozenset(
    {"Answer Relevancy", "Correctness [GEval]", "Citation Presence [GEval]"}
)
_GOLDEN_DATASET_PATH = paths.PROJECT_ROOT / "tests" / "golden_dataset.json"


class UnknownCaseError(RuntimeError):
    """An e2e-scoped test case's name has no match in the golden dataset --
    fails loudly rather than silently dropping it from the summary (D8.5's
    "fail loudly on an unrecognised mapping" discipline)."""


def load_latest_run() -> dict[str, Any]:
    """Read and JSON-parse DeepEval's own persisted full test-run file.

    Raises
    ------
    FileNotFoundError
        No such file -- named with the exact command needed, not a bare
        "file not found".
    """
    path = paths.resolve(LATEST_FULL_TEST_RUN_FILE_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist -- run `deepeval test run tests/` (with "
            "tests/test_e2e.py included) before summarize_e2e"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _category_by_id() -> dict[str, str]:
    cases = json.loads(_GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    return {case["id"]: case["category"] for case in cases}


def filter_e2e_cases(
    run: dict[str, Any], *, category_by_id: dict[str, str]
) -> list[dict[str, Any]]:
    """Keep only `testCases` entries scored by this stage's own three
    metrics, tagged with their golden-dataset category.

    Filters on metric identity (the three names this stage itself owns),
    not on the case name's own shape -- robust regardless of how many other
    eval-tier files (stage 7/8's component and tool-correctness tests) were
    collected in the same `deepeval test run` invocation.

    Raises
    ------
    UnknownCaseError
        An e2e-scoped case's `name` has no entry in `category_by_id`.
    """
    filtered: list[dict[str, Any]] = []
    for case in run.get("testCases", []):
        metric_names = {metric["name"] for metric in case.get("metricsData", [])}
        if metric_names != _E2E_METRIC_NAMES:
            continue
        category = category_by_id.get(case["name"])
        if category is None:
            raise UnknownCaseError(
                f"e2e test case {case['name']!r} has no matching id in "
                "tests/golden_dataset.json -- the mapping, not the run, is "
                "what failed here"
            )
        filtered.append({**case, "category": category})
    return filtered


def _case_passed(case: dict[str, Any]) -> bool:
    return all(metric["success"] for metric in case["metricsData"])


def render_summary_markdown(
    cases: list[dict[str, Any]],
    *,
    case_costs: list[dict[str, Any]],
    total_cases: int,
) -> str:
    """The brief's own mockup shape, all three of I4's elements.

    Parameters
    ----------
    cases : list of dict
        `filter_e2e_cases`'s own output.
    case_costs : list of dict
        `record_case_cost`'s own persisted records
        (`case_id`/`run_id`/`agent_cost_usd`/`judge_cost_usd`).
    total_cases : int
        The denominator for `Overall: N/<total_cases> passed` -- ordinarily
        `len(cases)`, passed explicitly rather than re-derived so a caller
        that already knows how many cases were *scored* (as opposed to
        skipped, e.g. the poisoned-knowledge-base case, D9a.12) states it
        plainly instead of this function guessing.
    """
    lines: list[str] = ["tests/test_e2e.py"]
    for case in sorted(cases, key=lambda c: str(c["name"])):
        icon = "✅" if _case_passed(case) else "❌"
        lines.append(f"  {icon} test_golden_dataset[{case['name']}]")
        for metric in case["metricsData"]:
            # A metric that errored (e.g. a judge-model timeout) carries no
            # "score" at all -- DeepEval's own file dumps with
            # `exclude_none=True`, so the key is absent, not `null`
            # (measured against a real run this stage's own live run
            # produced, not assumed offline).
            score = metric.get("score")
            score_text = f"{score:.2f}" if score is not None else "ERRORED"
            error = metric.get("error")
            suffix = f" -- {error}" if error else ""
            lines.append(
                f"     {metric['name']}: {score_text} "
                f"(threshold {metric['threshold']:.2f}){suffix}"
            )
    lines.append("")

    lines.append("Aggregates:")
    for metric_name in sorted(_E2E_METRIC_NAMES):
        all_metrics = [
            metric
            for case in cases
            for metric in case["metricsData"]
            if metric["name"] == metric_name
        ]
        scores = [m["score"] for m in all_metrics if m.get("score") is not None]
        errored = sum(1 for m in all_metrics if m.get("score") is None)
        if scores:
            errored_note = f", {errored} errored" if errored else ""
            lines.append(
                f"  {metric_name}: avg {sum(scores) / len(scores):.2f}, "
                f"min {min(scores):.2f}, max {max(scores):.2f}{errored_note}"
            )
    lines.append("")

    lines.append("Category breakdown (absolute counts):")
    categories = sorted({case["category"] for case in cases})
    for category in categories:
        in_category = [case for case in cases if case["category"] == category]
        passed = sum(1 for case in in_category if _case_passed(case))
        lines.append(f"  {category}: {passed}/{len(in_category)}")
    lines.append("")

    agent_total = sum(record["agent_cost_usd"] for record in case_costs)
    judge_total = sum(record["judge_cost_usd"] for record in case_costs)
    lines.append(
        f"Cost (both measured, neither estimated): agent ${agent_total:.4f}, "
        f"judge ${judge_total:.4f}"
    )
    lines.append("")

    passed_overall = sum(1 for case in cases if _case_passed(case))
    pass_rate = (passed_overall / total_cases * 100) if total_cases else 0.0
    lines.append(f"Overall: {passed_overall}/{total_cases} passed ({pass_rate:.1f}%)")

    return "\n".join(lines) + "\n"


def load_case_costs(eval_run_id: str) -> list[dict[str, Any]]:
    """Read back `runs/<eval_run_id>/case-costs.jsonl`
    (`tests.conftest.record_case_cost`'s own output)."""
    path = paths.resolve("runs") / eval_run_id / "case-costs.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _discover_latest_eval_run_id() -> str:
    runs_dir = paths.resolve("runs")
    candidates = sorted(
        runs_dir.glob("*/case-costs.jsonl"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(
            f"no runs/*/case-costs.jsonl under {runs_dir} -- run `deepeval "
            "test run tests/test_e2e.py` first, or pass eval_run_id explicitly"
        )
    return candidates[-1].parent.name


def main(eval_run_id: str | None = None) -> Path:
    """Read both sources, join them, write `eval-results.json` and
    `summary.md` under `runs/<eval_run_id>/`, and return that directory.

    Parameters
    ----------
    eval_run_id : str, optional
        Defaults to the most recently modified `runs/*/case-costs.jsonl`'s
        own directory.
    """
    if eval_run_id is None:
        eval_run_id = _discover_latest_eval_run_id()

    run = load_latest_run()
    cases = filter_e2e_cases(run, category_by_id=_category_by_id())
    case_costs = load_case_costs(eval_run_id)

    run_dir = paths.run_dir(eval_run_id)
    (run_dir / "eval-results.json").write_text(
        json.dumps({"cases": cases, "case_costs": case_costs}, indent=2),
        encoding="utf-8",
    )
    summary = render_summary_markdown(
        cases, case_costs=case_costs, total_cases=len(cases)
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return run_dir


if __name__ == "__main__":
    import sys

    written_to = main(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"wrote {written_to / 'summary.md'}")
