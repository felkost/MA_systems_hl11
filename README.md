# MA_systems_hl11 — a tested multi-agent research system

A terminal multi-agent research system — a Supervisor coordinating a Planner,
a Researcher and a Critic in a Plan → Research → Critique loop — together
with the automated **DeepEval** suite that measures it.

This repository solves homework-lesson-11. The assignment is about testing:
the system under test is ported from earlier work, and the engineering weight
sits in `tests/` and `evals/`.

> **Status: stage 0 of 10 (kickoff).** The repository skeleton, the rules and
> the layering test exist. **No application code has been written yet** — that
> is what kickoff is supposed to look like. Sections marked *(planned)* below
> describe what is coming, not what runs today.

## Architecture

Four agents, all in one process:

- **Supervisor** — coordinates, and owns the only write to disk.
- **Planner** — turns a request into concrete search queries and source
  choices (structured output).
- **Researcher** — executes the plan and produces findings with citations.
- **Critic** — verifies, then returns `APPROVE` or `REVISE` with actionable
  revision requests. The loop is capped.

Two interchangeable coordination paths express the same loop, so that the
revision cap is enforced by two independent mechanisms:

```
python main.py                          # agent-as-tool Supervisor  (planned)
python main.py --orchestration graph    # explicit StateGraph        (planned)
```

Retrieval is **ChromaDB only** — dense vectors plus BM25 over the same chunk
dump, with a cross-encoder rerank. There is no graph database, no MCP server
and no A2A server in this project, and no `docker-compose.yml`.

Every model runs through **OpenRouter**. Embeddings run locally, because
OpenRouter exposes no embeddings endpoint.

## Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

Then fill in `OPENROUTER_API_KEY` in `.env`. `.env.example` documents every
other setting, including which ones are optional.

`deepeval` is in `requirements-dev.txt`, not `requirements.txt`: it is a test
tool, not a runtime dependency. **Installing only `requirements.txt` gives you
tests that cannot run.**

## Run *(planned — stage 2 onwards)*

```bash
python ingest.py     # build the Chroma index from data/
python main.py       # the REPL
```

Reports are written to `output/` and only after you approve them: the single
write in the system is gated by a human-in-the-loop checkpoint.

## Tests

Three tiers, deliberately separate.

```bash
# Offline gate — no network, no API keys, no cost. Runs on every push in CI.
.venv/Scripts/python.exe -m black --check . tests/*.py && \
.venv/Scripts/python.exe -m flake8 . && \
.venv/Scripts/python.exe -m mypy . && \
.venv/Scripts/python.exe -m pytest -q -m "not smoke and not eval"

# Smoke — boots real services. On request.
.venv/Scripts/python.exe -m pytest -q -m smoke

# Evaluation — the deliverable. Calls live models and costs money.
deepeval test run tests/
```

The evaluation tier is excluded from the gate and runs in CI only on manual
dispatch. A test that goes red because a judge model drifted teaches everyone
to ignore a red CI, and a CI nobody trusts stops catching real breakage.

### What is measured *(planned — stages 6 to 9)*

| Metric | Kind | Target |
|---|---|---|
| Plan Quality | GEval | Planner |
| Critique Quality | GEval | Critic |
| Groundedness | GEval, over retrieval context | Researcher |
| Tool Correctness | deterministic | Planner, Researcher, Supervisor |
| Answer Relevancy | built-in | whole system |
| Correctness | GEval, against expected output | whole system |
| Citation Presence | **custom GEval** | whole system |

Against a golden dataset of **15 cases** in `tests/golden_dataset.json`: five
happy-path, five edge-case, five failure-case.

### How to read the numbers

Some tests are expected to fail. The goal is a baseline that improvement can
be measured against, not a green wall.

Three limits are stated here because they apply to every number this project
publishes:

- **Every score comes from a single run.** There are no confidence intervals.
- **The judge is not validated against human labels.** A GEval score is a
  signal for where to look, not measured truth.
- **A difference between two runs is a change, not a proven improvement.**

Thresholds start from a measured baseline and move **up** when the baseline
supports it. A threshold is never lowered to turn a red test green, and a
golden case is never edited to make the system pass.

## Layout

Flat, matching the assignment's own file tree. Layering is a property assigned
per file and enforced by `tests/test_layering.py`, an AST import walk — not by
a directory tree. Its two load-bearing rules are negative: the two
coordination paths never import each other, and no sub-agent imports a
coordinator.

## What leaves this machine

OpenRouter receives every prompt and completion. Langfuse Cloud receives
traces — but only when `TRACING_ENABLED=true`, which is off by default, and
nothing in the evaluation pipeline depends on it: metrics are computed from
local span dumps in `runs/`. See `CONTRIBUTING.md`.

## License

Coursework. No license granted.
