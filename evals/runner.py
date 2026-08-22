"""One system run -> a validated span record list (stage 5 slice, D5.1;
extended stage 7, D7.1).

`load_run` reads and schema-validates `runs/<run_id>/spans.json` -- the
stage-5/stage-8 contract this project's `docs/specs/stage-5.md` splits:
this module ships the schema and the reader at stage 5. `retrieval_context`
extraction shipped early, at stage 7 (`retrieval_context_for_agent`,
`docs/specs/stage-7.md` D7.1), because R2c (Groundedness) needs it a stage
before `build_llm_test_case` was originally due -- the plan's own audit A4
names this as a hard prerequisite, not a nice-to-have. `tools_called`
extraction and the full `build_llm_test_case` (turning a `RunSpans` into a
complete DeepEval `LLMTestCase`, including `tools_called` from `tool.<name>`
spans' `tool.args`) still ship at stage 8, against this schema -- not
designed here, per the rolling-wave rule that only the stage actually
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


def retrieval_context_for_agent(run: RunSpans, agent_span_name: str) -> list[str]:
    """Ancestor-scoped `retrieval.chunks` extraction for one agent.

    `knowledge_search` is on the Planner's, Researcher's and Critic's tool
    allowlist alike, so a flat filter on `tool.knowledge_search` mixes
    chunks none of them actually saw as their own retrieval context --
    measured on two real runs (`docs/specs/stage-7.md`, D7.2): one where a
    `knowledge_search` call belongs to `agent.planner` alongside the
    Researcher's own, another where a third belongs to `agent.critic`.

    Walks each `tool.knowledge_search` span's `parent_span_id` chain and
    keeps it only if `agent_span_name` appears somewhere in that chain. A
    span whose walk hits a `parent_span_id` absent from `run.spans` is
    **excluded**, never assumed in scope -- an incomplete dump must not
    silently widen what counts as this agent's own retrieval.

    Parameters
    ----------
    run : RunSpans
    agent_span_name : str
        E.g. `"agent.researcher"`.

    Returns
    -------
    list of str
        Every chunk string found, in span start order, with exact
        duplicates collapsed while preserving first-seen order.
    """
    by_id = {span["span_id"]: span for span in run.spans}

    def _has_ancestor(span: dict[str, Any]) -> bool:
        current: dict[str, Any] | None = span
        while current is not None:
            if current["name"] == agent_span_name:
                return True
            parent_id = current.get("parent_span_id")
            if parent_id is None:
                return False
            current = by_id.get(parent_id)
            if current is None:
                # parent_id was set but not present in this dump -- an
                # incomplete ancestor chain, treated as out of scope.
                return False
        return False

    seen: set[str] = set()
    chunks: list[str] = []
    for span in run.spans:
        if span["name"] != "tool.knowledge_search":
            continue
        if not _has_ancestor(span):
            continue
        for chunk in span.get("attributes", {}).get("retrieval.chunks", []):
            if chunk not in seen:
                seen.add(chunk)
                chunks.append(chunk)
    return chunks
