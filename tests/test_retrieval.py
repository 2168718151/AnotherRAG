from __future__ import annotations

import dataclasses

from llama_index.core.embeddings.mock_embed_model import MockEmbedding

from story_rag.adapters.model_factory import ModelBundle
from story_rag.config import Settings
from story_rag.container import build_container
from story_rag.domain import QueryRequest
from tests.conftest import CHINESE_STORY, ENGLISH_STORY, ScriptedLLM


def test_retrieval_returns_fused_and_reranked_trace(container) -> None:
    container.ingestion.ingest_bytes(CHINESE_STORY.encode("utf-8"), source_name="dragon.txt")
    container.ingestion.ingest_bytes(ENGLISH_STORY.encode("utf-8"), source_name="valley.txt")

    result = container.retrieval.retrieve_trace(QueryRequest(query="白龙把雨水带给谁？"))

    assert result.vector_candidates, "vector retriever returned nothing"
    assert result.bm25_candidates, "bm25 retriever returned nothing"
    assert result.fused_candidates, "fusion retriever returned nothing"
    assert result.nodes, "reranked nodes missing"
    assert all(node.rerank_score is not None for node in result.nodes)
    assert "白龙" in result.nodes[0].text
    assert result.nodes[0].metadata["source"] == "dragon.txt"
    assert result.latency_ms >= 0


def test_retrieval_trace_is_written_to_logs(container, settings: Settings) -> None:
    container.ingestion.ingest_bytes(CHINESE_STORY.encode("utf-8"), source_name="dragon.txt")

    container.retrieval.retrieve_trace(QueryRequest(query="白龙"))

    traces = list((settings.log_dir / "traces").glob("retrieval-*.jsonl"))
    assert traces, "retrieval trace file missing"
    assert "fused_candidates" in traces[0].read_text(encoding="utf-8")


def test_retrieval_without_documents_is_empty(container) -> None:
    result = container.retrieval.retrieve(QueryRequest(query="anything"))

    assert result.nodes == []
    assert result.fused_candidates == []


def test_retrievers_rebuild_after_late_ingestion(container) -> None:
    assert container.retrieval.retrieve(QueryRequest(query="白龙")).nodes == []

    container.ingestion.ingest_bytes(CHINESE_STORY.encode("utf-8"), source_name="dragon.txt")

    result = container.retrieval.retrieve(QueryRequest(query="白龙把雨水带给了谁"))
    assert result.nodes, "retrievers were not rebuilt after ingestion"
    assert any("白龙" in node.text for node in result.nodes)


def test_query_rewriting_uses_llm_when_enabled(settings: Settings, reranker) -> None:
    rewrite_settings = dataclasses.replace(settings, query_fusion_num_queries=3)
    llm = ScriptedLLM(responses=["rain dragon story\nvillage storyteller"])
    rewritten = build_container(
        rewrite_settings,
        models=ModelBundle(llm=llm, embedding=MockEmbedding(embed_dim=8), reranker=reranker),
    )
    rewritten.ingestion.ingest_bytes(ENGLISH_STORY.encode("utf-8"), source_name="valley.txt")

    result = rewritten.retrieval.retrieve(QueryRequest(query="Who carried the rain?"))

    assert llm.prompts, "query fusion did not call the LLM"
    assert result.fused_candidates
    assert any("dragon" in node.text.casefold() for node in result.nodes)
