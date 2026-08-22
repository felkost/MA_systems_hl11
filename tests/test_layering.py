"""Enforces the layer table in CLAUDE.md by walking every module's imports.

The table assigns a layer per file rather than per directory, because the
assignment prescribes a flat module layout and inventing a package tree would
make the deliverable harder to grade. Without this test the layer assignment
would be a paragraph nobody checks.

Two negative rules carry most of the value here, and both are about keeping
the two coordination paths independent. They are meant to enforce the same
revision cap by different mechanisms -- middleware on one side, state
arithmetic on the other -- so that the comparison between them says something.
An import between them would make a change to one silently change the other,
and a sub-agent importing a coordinator would turn the dependency arrow around
entirely.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KERNEL = "kernel"
DOMAIN = "domain"
INFRA = "infra"
APPLICATION = "application"
INTERFACE = "interface"
OBS = "obs"

# Mirrors the "Architecture" table in CLAUDE.md. A module absent from this map
# has no declared layer, which `test_every_module_has_a_declared_layer` treats
# as a failure rather than as permission to import anything.
LAYER_OF_MODULE: dict[str, str] = {
    # Package markers (`__init__.py`) declare the same layer as the modules
    # they contain -- an empty marker file still needs a home in the table,
    # or a fresh `agents/__init__.py` silently escapes the import-walk.
    "agents": DOMAIN,
    "evals": OBS,
    "main": INTERFACE,
    "supervisor": APPLICATION,
    "orchestrator": APPLICATION,
    "hitl": APPLICATION,
    "agents.planner": DOMAIN,
    "agents.research": DOMAIN,
    "agents.critic": DOMAIN,
    "schemas": DOMAIN,
    "prompts": DOMAIN,
    "tools": INFRA,
    "middleware": INFRA,
    "models": INFRA,
    "retriever": INFRA,
    "ingest": INFRA,
    "config": KERNEL,
    "paths": KERNEL,
    "observability": OBS,
    "evals.deepeval_model": OBS,
    "evals.runner": OBS,
}

#  Stage 2 finding: a layer must be allowed to import itself. The table in
# CLAUDE.md ("May import") never says a layer may reach its own -- an
# omission nobody could see at stage 0, when this dict was written and no
# code existed yet. The first real modules (config.py -> paths.py, both
# kernel; retriever.py -> models.py, both infra; tools.py -> retriever.py,
# both infra) all need a same-layer import, so each set below now includes
# its own layer explicitly.
MAY_IMPORT: dict[str, frozenset[str]] = {
    KERNEL: frozenset({KERNEL}),
    DOMAIN: frozenset({KERNEL, DOMAIN}),
    INFRA: frozenset({KERNEL, DOMAIN, INFRA}),
    APPLICATION: frozenset({KERNEL, DOMAIN, INFRA, APPLICATION}),
    INTERFACE: frozenset({KERNEL, DOMAIN, INFRA, APPLICATION, INTERFACE, OBS}),
    OBS: frozenset({KERNEL, DOMAIN, INFRA, OBS}),
}

# Pairs that must never import each other in either direction, regardless of
# what the layer rules alone would permit.
FORBIDDEN_PAIRS: tuple[tuple[str, str], ...] = (
    ("supervisor", "orchestrator"),
    ("agents.planner", "supervisor"),
    ("agents.planner", "orchestrator"),
    ("agents.research", "supervisor"),
    ("agents.research", "orchestrator"),
    ("agents.critic", "supervisor"),
    ("agents.critic", "orchestrator"),
)

# Directories that hold no project modules: virtualenvs, caches, local-only
# working areas, and the test suite itself.
_SKIPPED_DIRS = frozenset(
    {".venv", ".cache", ".git", "__pycache__", "docs", "tests", "index", "runs"}
)


def _module_name(path: Path) -> str:
    """Dotted name of `path` relative to the project root, without `.py`."""
    relative = path.relative_to(PROJECT_ROOT).with_suffix("")
    parts = [part for part in relative.parts if part != "__init__"]
    return ".".join(parts)


def _project_modules() -> list[Path]:
    """Every project `.py` file, excluding vendored and local-only trees."""
    return sorted(
        path
        for path in PROJECT_ROOT.rglob("*.py")
        if not _SKIPPED_DIRS.intersection(path.relative_to(PROJECT_ROOT).parts)
    )


def _imported_project_modules(path: Path) -> set[str]:
    """Project-local modules that `path` imports, by dotted name.

    Resolves each import against `LAYER_OF_MODULE` and against its package
    prefix, so `from agents.planner import x` and `import agents.planner` both
    resolve to `agents.planner`, while `import json` resolves to nothing.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {name for name in names if name in LAYER_OF_MODULE}


def test_every_module_has_a_declared_layer() -> None:
    """A new module must be added to the table before it may import anything.

    Catching this here rather than at review time is what stops the table in
    CLAUDE.md from quietly falling behind the tree it describes.
    """
    undeclared = [
        _module_name(path)
        for path in _project_modules()
        if _module_name(path) not in LAYER_OF_MODULE
    ]
    assert not undeclared, (
        f"modules missing from LAYER_OF_MODULE (and from CLAUDE.md's "
        f"architecture table): {undeclared}"
    )


@pytest.mark.parametrize("path", _project_modules(), ids=_module_name)
def test_module_imports_respect_its_layer(path: Path) -> None:
    """No module imports a layer its own layer may not reach."""
    importer = _module_name(path)
    importer_layer = LAYER_OF_MODULE[importer]
    allowed = MAY_IMPORT[importer_layer]

    violations = [
        f"{importer} ({importer_layer}) imports {imported} "
        f"({LAYER_OF_MODULE[imported]})"
        for imported in sorted(_imported_project_modules(path))
        if imported != importer and LAYER_OF_MODULE[imported] not in allowed
    ]
    assert not violations, "; ".join(violations)


@pytest.mark.parametrize("left,right", FORBIDDEN_PAIRS, ids=lambda v: v)
def test_forbidden_pairs_never_import_each_other(left: str, right: str) -> None:
    """The two coordination paths, and the sub-agents, stay independent.

    Passing vacuously while neither module exists is intended: the rule is
    written before the code so that the first version of that code is already
    constrained by it.
    """
    for importer, forbidden in ((left, right), (right, left)):
        path = PROJECT_ROOT / f"{importer.replace('.', '/')}.py"
        if not path.exists():
            continue
        assert forbidden not in _imported_project_modules(path), (
            f"{importer} imports {forbidden}; they must stay independent "
            "(see CLAUDE.md, Invariants)"
        )
