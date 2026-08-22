"""One system run -> a validated span record list (stage 5 slice, D5.1).

`load_run` reads and schema-validates `runs/<run_id>/spans.json` -- the
stage-5/stage-8 contract this project's `docs/specs/stage-5.md` splits:
this module ships the schema and the reader now; `build_llm_test_case`,
turning a `RunSpans` into a DeepEval `LLMTestCase` (`actual_output`,
`retrieval_context` from `retrieval.chunks` attributes, `tools_called` from
`tool.<name>` spans' `tool.args`), ships at stage 8, against this schema --
not designed here, per the rolling-wave rule that only the stage actually
needing a piece of design commits to its exact shape.

A malformed dump raises immediately rather than returning a partial
result: an evaluation stage reading spans that don't match this schema is a
defect to surface loudly, not silently work around.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import paths

_REQUIRED_SPAN_FIELDS = (
    "trace_id",
    "span_id",
    "parent_span_id",
    "name",
    "start_time",
    "end_time",
    "status",
    "attributes",
)


@dataclass(frozen=True)
class RunSpans:
    """One run's validated span records, as written by
    `observability.SpanJsonExporter`."""

    run_id: str
    spans: list[dict[str, Any]]


def load_run(run_id: str, runs_dir: str | Path = "runs") -> RunSpans:
    """Read and schema-validate `runs/<run_id>/spans.json`.

    Parameters
    ----------
    run_id : str
    runs_dir : str or Path, default "runs"

    Returns
    -------
    RunSpans

    Raises
    ------
    FileNotFoundError
        No dump exists for `run_id`.
    ValueError
        A span record is missing a required field.
    """
    dump_path = paths.span_dump_path(run_id, runs_dir)
    if not dump_path.exists():
        raise FileNotFoundError(f"no span dump for run_id {run_id!r} at {dump_path}")

    spans = json.loads(dump_path.read_text(encoding="utf-8"))
    for index, span in enumerate(spans):
        missing = [field for field in _REQUIRED_SPAN_FIELDS if field not in span]
        if missing:
            raise ValueError(
                f"span {index} in {dump_path} is missing required field(s): "
                f"{missing}"
            )
    return RunSpans(run_id=run_id, spans=spans)
