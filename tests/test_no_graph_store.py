"""No tracked Python module mentions Neo4j or `graph_search`.

CLAUDE.md's Forbidden list bans a graph store outright (see "Three deliberate
removals" -- a graph store is a second source of truth this assignment does
not measure). Scoped to `.py` files, not every tracked file: CLAUDE.md and
`insights.md` legitimately *discuss* the removed Neo4j/`graph_search` tool as
history, which is prose about a decision, not a place either could resurface
as code.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIS_FILE = Path(__file__).resolve()

_FORBIDDEN_TERMS = ("neo4j", "graph_search")


def _tracked_python_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [PROJECT_ROOT / line for line in result.stdout.splitlines() if line]


def test_no_tracked_python_file_mentions_neo4j_or_graph_search() -> None:
    offenders = []
    for path in _tracked_python_files():
        if path.resolve() == THIS_FILE:
            continue  # this file names the forbidden terms to check for them
        lowered = path.read_text(encoding="utf-8").lower()
        hits = [term for term in _FORBIDDEN_TERMS if term in lowered]
        if hits:
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {hits}")

    assert (
        not offenders
    ), "tracked Python files still reference a graph store: " + "; ".join(offenders)
