"""Configuration read from the environment and `.env`.

Every tunable value lives on `Settings`. There is exactly one chat-model
provider in this project -- OpenRouter -- so, unlike both donor projects,
there is no `LLM_PROVIDER` toggle: a setting with one legal value is noise
that invites a second value later (`docs/specs/stage-2.md`, "`models.py` --
OpenRouter only, no provider toggle for LLMs"). `EMBEDDING_PROVIDER` stays a
real toggle (`openrouter | local`) because it genuinely has two values -- the
local Hugging Face branch is the documented offline escape hatch.

`_CacheSettings` is deliberately separate from `Settings`: it exists only to
put `HF_HOME`/`PIP_CACHE_DIR`/`DEEPEVAL_HOME` into `os.environ` at import
time, before any Hugging Face or DeepEval import can read them (stage 0's
own finding: pydantic-settings parses `.env` into `Settings` and exports
nothing to the process environment, so those variables otherwise do nothing
on their own). Its `settings_customise_sources` drops the ambient-OS-environment
source entirely for these three fields -- the CLAUDE.md invariant "`Settings`
decides, not the ambient environment" applied literally: an already-exported
`HF_HOME` from some other tool on this machine must not decide where this
project's model cache lands.

`DEEPEVAL_HOME` joined the other two at stage 7: `import deepeval` writes an
anonymous telemetry identity file there (measured, stage 7 kickoff --
`.venv/Lib/site-packages/deepeval/telemetry/identity.py`), and its own
default is `Path.home() / ".deepeval"` -- `C:\\Users\\...` on this machine,
the one place CLAUDE.md's Forbidden list bans a cache from landing. With
`DEEPEVAL_TELEMETRY_OPT_OUT=YES` set (it is, in `.env.example` and in CI)
no file is written at all and `DEEPEVAL_HOME` never matters in practice --
it is set anyway so the invariant holds even if telemetry is ever turned
back on, the same reasoning `.env.example` already gives for that variable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

import paths

_ENV_FILE = str(Path(__file__).resolve().parent / ".env")


class Settings(BaseSettings):
    """Validated configuration for one process, shared by every agent.

    Notes
    -----
    Fields are matched to their upper-case environment variable by name
    (pydantic-settings' default).
    """

    ROLES: ClassVar[tuple[str, ...]] = (
        "supervisor",
        "planner",
        "researcher",
        "critic",
    )

    # -- Models. Every role goes through OpenRouter; model ids are always
    # "vendor/slug" there, checked by _model_names_are_openrouter_ids below.
    openrouter_api_key: SecretStr
    model_name: str = "openai/gpt-4.1-mini"
    supervisor_model_name: str | None = None
    planner_model_name: str | None = None
    researcher_model_name: str | None = None
    critic_model_name: str | None = None
    # No default on purpose (.env.example): a judge model is always a
    # deliberate choice, never inherited from whatever the agents run.
    judge_model_name: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    # -- Prompt versions (prompts.py, stage 3). `critic_prompt_version`
    # defaults to "c2", the coupled verdict/booleans version -- matching
    # hl8's own default and what docs/task-hl11.md's Critique Quality GEval
    # measures against (docs/specs/stage-3.md, D3.4).
    planner_prompt_version: str = "p1"
    researcher_prompt_version: str = "r1"
    critic_prompt_version: str = "c2"
    supervisor_prompt_version: str = "s1"
    composer_prompt_version: str = "w1"

    # -- Supervisor / revision loop
    max_revisions: int = Field(default=2, ge=1, le=3)
    recursion_limit: int = Field(default=100, ge=2, le=200)
    researcher_max_tool_calls: int = Field(default=8, ge=1, le=50)
    critic_max_tool_calls: int = Field(default=5, ge=1, le=50)
    max_read_url_per_search: int | None = Field(default=2, ge=1, le=10)
    # No default: `resolved_supervisor_max_tool_calls` derives a headroom-
    # aware value from `max_revisions` when this is left unset (stage-4 spec
    # D4.13) -- a fixed default sized for one `max_revisions` value is wrong
    # for the others, since the worst-case legitimate call count scales with
    # the cap itself.
    supervisor_max_tool_calls: int | None = Field(default=None, ge=1, le=200)

    # -- Tools (tools.py)
    max_search_results: int = Field(default=5, ge=1, le=10)
    max_search_snippet_length: int = Field(default=500, ge=100, le=2000)
    max_url_content_length: int = Field(default=5000, ge=1000, le=10000)
    http_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    output_dir: str = "output"
    max_report_content_length: int = Field(default=200_000, ge=1_000, le=5_000_000)

    # -- Egress guardrail (read_url). Off by default -- flipping it is a
    # deliberate, recorded act, per "Settings decides, not the ambient
    # environment".
    allow_private_network_urls: bool = False
    max_url_redirects: int = Field(default=3, ge=0, le=10)

    # -- RAG / retrieval (ingest.py, retriever.py)
    data_dir: str = "data"
    index_dir: str = "index"
    collection_name: str = "knowledge_base"
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: Literal["auto", "cpu", "cuda"] = "auto"
    retrieval_top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_n: int = Field(default=3, ge=1, le=20)
    ensemble_bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    model_cache_dir: str = ".cache/models"
    rerank_confidence_floor: float = 0.3
    max_knowledge_search_length: int = Field(default=3000, ge=500, le=10000)

    embedding_provider: Literal["openrouter", "local"] = "openrouter"
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dimensions: int | None = Field(default=None, ge=64, le=3072)

    # -- HITL checkpoint: unused. Stage 4 runs `MemorySaver` on the
    # Supervisor and on the compiled orchestrator graph -- never a
    # sub-agent's (CLAUDE.md invariant) -- because
    # `langgraph-checkpoint-sqlite` is not a dependency and an in-process
    # REPL session needs no cross-process durability. The field and
    # `paths.checkpoint_path` wait for a stage that adds the backend.
    checkpoint_db: str = "runtime/checkpoints.sqlite"

    # -- Observability (stage 5), Langfuse Cloud only -- no docker-compose.yml
    # exists or may exist in this project (CLAUDE.md, Forbidden).
    tracing_enabled: bool = False
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    trace_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    span_dump_dir: str | None = None
    max_span_payload_length: int = Field(default=2000, ge=100, le=20_000)

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("tracing_enabled", mode="before")
    @classmethod
    def _blank_tracing_flag_means_off(cls, value: object) -> object:
        """A present-but-empty `TRACING_ENABLED=` line means "off", not a
        `bool_parsing` error -- `.env.example` leaves it blank until stage 5
        gives it a reader, deliberately, and that must still construct
        `Settings` today."""
        return False if value == "" else value

    def resolved_model(self, role: str) -> str:
        """The OpenRouter model id `role` resolves to.

        Parameters
        ----------
        role : str
            One of `Settings.ROLES`.

        Returns
        -------
        str
            `<role>_model_name` if set, else the shared `model_name` -- the
            one place this fallback formula is implemented.
        """
        if role not in self.ROLES:
            raise ValueError(f"unknown role {role!r}, expected one of {self.ROLES}")
        override: str | None = getattr(self, f"{role}_model_name")
        return override or self.model_name

    def resolved_supervisor_max_tool_calls(self) -> int:
        """The Supervisor's blanket tool-call budget.

        Returns
        -------
        int
            `supervisor_max_tool_calls` if set, else a derived value:
            `1 (plan) + 2 * (max_revisions + 1) (research + critique) + 1
            (save_report) + 3` headroom. The bare worst-case count is below
            what it needs to be for a larger `max_revisions` -- a fixed
            default sized at one value starves the others. Headroom
            accounting, measured against the installed
            `HumanInTheLoopMiddleware`/`ToolCallLimitMiddleware`: one HITL
            reject costs one re-emitted `save_report`, one refused extra
            `critique` costs one more, so a headroom of 2 is exactly
            consumed by one of each -- and hl10 measured a live
            double-reject. 3 covers the double-reject plus one refusal
            (`docs/specs/stage-4.md`, D4.13).
        """
        if self.supervisor_max_tool_calls is not None:
            return self.supervisor_max_tool_calls
        return 1 + 2 * (self.max_revisions + 1) + 1 + 3

    @model_validator(mode="after")
    def _model_names_are_openrouter_ids(self) -> Settings:
        """Reject a model id that cannot be an OpenRouter id.

        OpenRouter model ids are always `vendor/slug`; a bare model name
        (e.g. a raw OpenAI id copied from a donor project) is refused at
        startup, not sent to the wrong endpoint.
        """
        candidates = [self.resolved_model(role) for role in self.ROLES]
        if self.judge_model_name is not None:
            candidates.append(self.judge_model_name)
        bad = [model for model in candidates if "/" not in model]
        if bad:
            raise ValueError(
                f"model id(s) {bad} contain no slash -- OpenRouter model ids "
                "are always 'vendor/slug'"
            )
        return self

    @model_validator(mode="after")
    def _tracing_requires_langfuse_keys(self) -> Settings:
        """Refuse to start rather than trace into the void (stage 5)."""
        if self.tracing_enabled and (
            self.langfuse_public_key is None or self.langfuse_secret_key is None
        ):
            raise ValueError(
                "TRACING_ENABLED is true, but LANGFUSE_PUBLIC_KEY/"
                "LANGFUSE_SECRET_KEY are not both set"
            )
        return self

    @model_validator(mode="after")
    def _overlap_fits_in_a_chunk(self) -> Settings:
        """Reject an overlap that is too large for the chunk size.

        `RecursiveCharacterTextSplitter` raises this error only once it
        starts splitting, which happens after the documents are loaded. Both
        values are already known at startup.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    @model_validator(mode="after")
    def _reranker_has_enough_candidates(self) -> Settings:
        """Reject a `rerank_top_n` larger than the candidate pool can hold.

        Each arm returns `retrieval_top_k` documents and the two arms
        overlap, so the merged pool holds between `retrieval_top_k` and
        twice that number. Only above the upper bound can the reranker never
        filter anything.
        """
        pool = 2 * self.retrieval_top_k
        if self.rerank_top_n > pool:
            raise ValueError(
                f"rerank_top_n ({self.rerank_top_n}) exceeds the largest "
                f"possible candidate pool ({pool} = 2 x retrieval_top_k "
                f"{self.retrieval_top_k})"
            )
        return self


def load_settings() -> Settings:
    """Build `Settings` from the environment and `.env`.

    Returns
    -------
    Settings
        Validated configuration.

    Notes
    -----
    `model_validate({})` is used instead of `Settings()` so mypy does not
    report the (in fact required, environment-supplied) `openrouter_api_key`
    as a missing constructor argument.
    """
    return Settings.model_validate({})


class _CacheSettings(BaseSettings):
    """Just the two variables `config.py` must put into `os.environ` itself.

    Kept separate from `Settings` (see module docstring): its
    `settings_customise_sources` drops the ambient-environment source
    entirely, so an `HF_HOME` some other tool already exported on this
    machine can never win over this project's own value.
    """

    hf_home: str = ".cache/hf"
    pip_cache_dir: str = ".cache/pip"
    deepeval_home: str = ".cache/deepeval"

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, dotenv_settings, file_secret_settings)


def _export_cache_env() -> None:
    """Write the cache-location variables into `os.environ`, project value wins.

    Runs once at import time -- before any module in this project can import
    `sentence_transformers`, `huggingface_hub`, or `deepeval`, all of which
    read `os.environ` and nothing else (stage 0's finding, extended to
    `DEEPEVAL_HOME` at stage 7).
    """
    cache = _CacheSettings()
    os.environ["HF_HOME"] = str(paths.resolve(cache.hf_home))
    os.environ["PIP_CACHE_DIR"] = str(paths.resolve(cache.pip_cache_dir))
    os.environ["DEEPEVAL_HOME"] = str(paths.resolve(cache.deepeval_home))


_export_cache_env()
