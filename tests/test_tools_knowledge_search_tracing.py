"""`knowledge_search` records the retrieved chunks' raw text onto the
current span, separately from the formatted string the model sees
(stage 5, `docs/specs/stage-5.md` D5.8)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._once import Once

import tools


@pytest.fixture(autouse=True)
def _isolated_provider() -> Any:
    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    yield
    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()


def test_knowledge_search_records_retrieval_chunks_on_current_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)

    documents = [
        Document(
            page_content="chunk one text", metadata={"source": "a.pdf", "page": 0}
        ),
        Document(
            page_content="chunk two text", metadata={"source": "b.pdf", "page": 1}
        ),
    ]
    monkeypatch.setattr(tools, "get_retriever", lambda: _FakeRetriever(documents))

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("tool.knowledge_search"):
        result = tools.knowledge_search.invoke({"query": "test question"})

    assert "chunk one text" in result

    finished = exporter.get_finished_spans()
    recorded_span = next(s for s in finished if s.name == "tool.knowledge_search")
    # A homogeneous string sequence is a valid OTel attribute type directly
    # -- no JSON encoding needed here, unlike TracingMiddleware's `tool.args`
    # (a dict, which OTel silently drops unless serialized -- D5.10).
    assert recorded_span.attributes is not None
    chunks = recorded_span.attributes["retrieval.chunks"]
    assert chunks == ("chunk one text", "chunk two text") or chunks == [
        "chunk one text",
        "chunk two text",
    ]


class _FakeRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def invoke(self, query: str) -> list[Document]:
        return self._documents
