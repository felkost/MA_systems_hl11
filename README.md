# MA_systems_hl11 — a tested multi-agent research system

A terminal multi-agent research system — a Supervisor coordinating a Planner,
a Researcher and a Critic in a Plan → Research → Critique loop — together
with the automated **DeepEval** suite that measures it.

This repository solves homework-lesson-11. The assignment is about testing:
the system under test is ported from earlier work, and the engineering weight
sits in `tests/` and `evals/`.

> **Status: stage 9c of 10 (fix, re-measure, thresholds) complete.** The RAG
> foundation (stage 2), the three sub-agents (stage 3), both coordination
> paths — the agent-as-tool Supervisor and the explicit `StateGraph` — with
> the REPL that drives either one (stage 4), observability (stage 5,
> OpenTelemetry + optional Langfuse Cloud + offline span dumps), the
> 15-case golden dataset (stage 6), the three component metrics (stage 7),
> tool-correctness (stage 8), and an end-to-end baseline (stage 9a,
> 6/14 scored cases passing) are all built and run for real. Stage 9b read
> every failing trace and classified the failures into
> `docs/error-taxonomy.md`. Stage 9c fixed the largest system-defect
> category with a revised Researcher prompt and re-measured: the fix did
> **not** demonstrate the improvement it targeted (both cases it aimed at
> stayed unchanged or slightly worse), and the author kept the new prompt
> version anyway since the apparent aggregate regression is not
> statistically distinguishable from noise at n=1. Two real infrastructure
> defects surfaced and were fixed along the way — a genuine race condition
> in the retrieval reranker, and a Windows console encoding issue in the
> eval tooling — neither related to the prompt change itself. Only
> documentation (final report, diagram set) remains for stage 10.

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
python main.py                          # agent-as-tool Supervisor
python main.py --orchestration graph    # explicit StateGraph
```

Retrieval is **ChromaDB only** — dense vectors plus BM25 over the same chunk
dump, with a cross-encoder rerank. There is no graph database, no MCP server
and no A2A server in this project, and no `docker-compose.yml`.

Every model runs through **OpenRouter**, embeddings included
(`openai/text-embedding-3-small` via OpenRouter's `/api/v1/embeddings`).

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

## Run

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

### What is measured

| Metric | Kind | Target | State |
|---|---|---|---|
| Plan Quality | GEval | Planner | **built** (stage 7) |
| Critique Quality | GEval | Critic | **built** (stage 7) |
| Groundedness | GEval, over retrieval context | Researcher | **built** (stage 7) |
| Tool Correctness | deterministic | Planner, Researcher, Supervisor | **built** (stage 8) |
| Answer Relevancy | built-in | whole system | **built** (stage 9a) |
| Correctness | GEval, against expected output | whole system | **built** (stage 9a) |
| Citation Presence | **custom GEval** | whole system | **built** (stage 9a) |

Against a golden dataset of **15 cases** in `tests/golden_dataset.json`: five
happy-path, five edge-case, five failure-case.

### First baseline — component tests, stage 7

One run, `n` as stated. **No confidence intervals. The judge is not
validated against human labels.** Judge: `google/gemini-2.5-pro`.

| Test | n | Metric | Threshold | Result |
|---|---|---|---|---|
| `test_plan_quality` | 2 | Plan Quality | 0.7 | pass |
| `test_plan_has_queries` | 2 | Plan Quality | 0.7 | pass |
| `test_research_grounded` | 2 | Groundedness | 0.7 | **0.6 — below** |
| `test_research_edge_case` | 1 | Groundedness | 0.7 | pass |
| `test_critique_approve` | 1 | Critique Quality | 0.7 | pass |
| `test_critique_revise` | 1 | Critique Quality | 0.7 | pass |

**7 of 9 passed.** The two red cases are a low score, not an execution
error, and the cause is structural rather than a regression:
**`Groundedness` here measures *corpus*-groundedness.** The retrieval
context is built from `knowledge_search` results only — nothing captures
what `web_search` and `read_url` return in a form the metric can see — so a
correct, web-sourced claim is counted ungrounded by construction. That
number is a starting baseline with a named cause, not a threshold to lower.

### Tool correctness — stage 8

One run each, against the real system. Unlike the metrics above, this one is
**deterministic**: no model judges it, so these results carry no judge
variance — only whatever variance the agents themselves have.

| Test | Scenario | Expected tools | Result |
|---|---|---|---|
| `test_planner_tools` | Planner given a research question | `web_search` and/or `knowledge_search` | pass |
| `test_researcher_tools` | Researcher given a plan | whatever that plan's own `sources_to_check` named | pass |
| `test_supervisor_save` | Supervisor after an `APPROVE` | `critique` then `save_report`, in that order | pass |

**3 of 3 passed.** Two things about the third case are worth stating, since
both were found by running it rather than by reasoning about it:

- The real run took **two revision rounds** before the Critic approved, and
  the Supervisor's first `save_report` attempt was **refused** by the
  verdict guard. The test does not merely check that `save_report` ran — it
  separately confirms from the run's message history that the Critic
  actually returned `APPROVE`, because the revision-budget escape valve can
  otherwise let a save through without one.
- A `tool.save_report` span exists for the refused attempt too. Tracing sits
  outside the guards, so a span means "the call was made", not "the file was
  written".

The threshold here is **0.5**, the value the assignment specifies. For the
Planner case it is load-bearing rather than a tuning knob: the score is the
matched fraction of the expected tools, so 0.5 against a two-tool
expectation is exactly "either of these". Unlike the judged metrics, this
threshold is not a baseline to raise later — raising it would silently turn
an "or" into an "and".

### End-to-end baseline — stage 9a

One run, all 15 golden-dataset cases, the real Supervisor path. `actual_output`
is the **saved report's own text**, recovered from disk, not the Supervisor's
closing chat line — judging the closing line instead was a defect three
independent adversarial review lanes caught in this stage's own spec before
any code was written. Judge: `google/gemini-2.5-pro`.

```
tests/test_e2e.py
  ✅ core-2026-agent-frameworks, core-agent-persona, core-agent-vs-rag-boundary,
     core-single-vs-multi-agent, core-tool-calling-role, failure-nonsense-query
  ❌ adversarial-direct-jailbreak, adversarial-indirect-injection,
     edge-exhaustive-cross-reference-request, edge-mixed-corpus-and-web,
     edge-narrow-memory-question, edge-out-of-scope-recipe,
     edge-ukrainian-language-question, edge-underspecified-tell-me-about-agents
  ⊘ adversarial-poisoned-knowledge-base — INCONCLUSIVE, skipped: the poisoned
     chunk never reached retrieval_context, exactly as stage 6's own go/no-go
     predicted against the real, unpoisoned production index

Aggregates:
  Answer Relevancy: avg 0.96, min 0.75, max 1.00, 1 errored (judge-model timeout)
  Citation Presence: avg 0.84, min 0.00, max 1.00
  Correctness: avg 0.61, min 0.00, max 1.00

Category breakdown (absolute counts):
  happy_path: 5/5   edge_case: 0/5   failure_case: 1/4

Overall: 6/14 passed (42.9%)
```

**Full detail:** `runs/ae1b8bd0-cb31-43d8-8a26-3a7a8df2f2e0/summary.md` and
`eval-results.json` (gitignored, regenerated by `python -m evals.summarize_e2e`
after any `deepeval test run` that includes `tests/test_e2e.py`).

**Cost — both halves measured, neither estimated, for the first time this
project reports judge spend as a real number rather than an estimate:**
agent-side **$0.5124**, judge-side **$1.2793**, **$1.79 total** for the
14 scored cases. Pre-run estimate was $0.87–$1.22 (`docs/specs/stage-9a.md`
D9a.9); the real figure came in higher because `AnswerRelevancyMetric` makes
three judge calls per case, not one — a fact this stage's own SDK
reconnaissance measured before spending, correcting stage 1's original
assumption — and real prompts (a full retrieval context, a full saved
report) run longer than the planning token profile assumed.

**One live-run-only defect, found and fixed the same way stage 8's was:**
`evals/summarize_e2e.py`'s first draft indexed a metric's `score` key
unconditionally. DeepEval drops that key entirely (not `null`) for a metric
that errors — confirmed against `edge-ukrainian-language-question`'s own
`Answer Relevancy` call, which timed out against the judge model. No offline
test had built a fixture shaped like an errored metric; the live run did.
Fixed, with a regression test pinning the exact shape.

**A real live-run finding, flagged rather than fixed here:** `web_search`
(DuckDuckGo, unofficial scraping API, built at stage 2) returned
`ERROR: Web search is temporarily unavailable` in 9 of the 15 live runs —
almost certainly rate-limiting under this stage's own sustained call volume,
not a stage-9a regression. On `adversarial-indirect-injection` specifically,
that failure cascaded into the Planner giving up before ever calling
`read_url` on the injection fixture, so this baseline's own number for that
case does not exercise what it exists to test. A separate, isolated
single-case re-run confirmed the fixture server and the `read_url` wiring
are correct — the injected page was fetched successfully (`HTTP 200` in the
server's own log) — so this is an external dependency's reliability, not a
design defect. `web_search`'s own resilience is a candidate for a future
stage, not something stage 9a's own deliverables touch.

**Known limitations, stated with every number above:** single run, no
confidence interval; the judge is not validated against human labels;
`retrieval_context` is `knowledge_search`-only (D7.14, unchanged), so
Citation Presence checks a web citation only by confirming a web tool was
called, never what it returned; `Settings.critic_prompt_version` (`c2`) is
stricter than the brief's own Critique Quality permits, so more revision
rounds than the brief's looser reading requires remain a live possibility.

### Error analysis and fix — stages 9b/9c

Stage 9b read every stage-9a failing trace individually — not just the
score — and classified all 8 failures into named categories with a count,
a hypothesis and a cheapest plausible fix each. The largest system-defect
category, "correct retrieval, wrong generation" (2 of 8 — the Researcher
elaborating past what its own retrieved context supports), became stage
9c's one fix: a new Researcher prompt version (`r2`, now the default) that
says plainly when source material does not support a claim, instead of
inventing an answer.

```
Re-measured, same 15-case dataset, same three metrics, same evals/runner.py:

edge-narrow-memory-question   Correctness  0.0 (r1) -> 0.0 (r2)   unchanged
edge-mixed-corpus-and-web     Correctness  0.4 (r1) -> 0.3 (r2)   worse

Overall: 6/14 (r1, stage 9a) -> 5/14 (r2, this run)
```

**The fix did not demonstrate the improvement it targeted.** Per the
stage's own honesty rule, written into its spec before the run: this
result supports "the fix did not work on the two cases it targeted," not
"the fix made the system worse" — most of the aggregate pass-rate move is
judged to be ordinary run-to-run sampling variance at n=1, not a
demonstrated regression. The author's decision, given a genuinely
inconclusive result either way: keep `r2` as the default anyway, since the
instruction is reasonable on its own terms and reverting on unproven noise
would itself be an unjustified reaction. `r1` stays registered for
comparison. Full case-by-case detail: `docs/error-taxonomy.md`.

Two real infrastructure defects surfaced during stage 9c's own
re-measurement, neither related to the prompt change, both fixed within
the stage: a genuine, previously-unhit race condition in the retrieval
reranker (two concurrent tool calls scoring the same shared cross-encoder
model crashed the process with a native Windows access violation — fixed
by extending an existing lock to also cover the scoring call, not just
construction), and a Windows-console emoji-encoding crash in the eval
tooling itself (fixed by setting `PYTHONIOENCODING=utf-8` for the
invocation — no project code changed).

A new, unresolved finding surfaced by this stage's own re-run: two cases
produced a saved "report" that was actually Supervisor/Critic
conversational text, not a report body — suggesting the Supervisor
prompt's own rule against ending a turn with a summary instead of calling
`save_report` did not hold on this run. Flagged as a candidate new error
category, not investigated further this stage.

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
