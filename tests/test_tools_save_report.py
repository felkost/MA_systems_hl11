"""`save_report` confines a model-supplied filename to `output/`.

The refusal is policy-class: `resolve_report_path` raises `ReportPathError`,
a typed exception, rather than returning a wrong path -- so the `@tool`
wrapper's caught-and-converted `"ERROR: ..."` string is distinguishable, in a
span, from "the disk was full" (`docs/specs/stage-2.md`, "`tools.py` --
guardrails ship with the tools, not after them"). `save_report` also never
overwrites a file that already exists.

Tests call `resolve_report_path` directly, not the `@tool`-wrapped
`save_report`: LangChain's `@tool` decorator turns the function into a
`BaseTool` whose `.invoke()` path adds argument-schema machinery this test
has no reason to go through to check one thing -- path confinement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import ReportPathError, resolve_report_path


def test_relative_traversal_is_refused(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with pytest.raises(ReportPathError):
        resolve_report_path("../escape.md", output_dir)


def test_absolute_path_is_refused(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    outside = tmp_path / "elsewhere" / "report.md"

    with pytest.raises(ReportPathError):
        resolve_report_path(str(outside), output_dir)


def test_symlink_target_outside_output_dir_is_refused(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    outside_dir = tmp_path / "outside"
    output_dir.mkdir()
    outside_dir.mkdir()

    link = output_dir / "escape.md"
    try:
        link.symlink_to(outside_dir / "target.md")
    except OSError:
        pytest.skip("symlink creation requires elevated privileges on this machine")

    with pytest.raises(ReportPathError):
        resolve_report_path("escape.md", output_dir)


def test_plain_filename_resolves_inside_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    resolved = resolve_report_path("my-report.md", output_dir)

    assert resolved.parent == output_dir.resolve()
    assert resolved.name == "my-report.md"


def test_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing = output_dir / "already-there.md"
    existing.write_text("original", encoding="utf-8")

    with pytest.raises(ReportPathError):
        resolve_report_path("already-there.md", output_dir, must_not_exist=True)
