"""Integration checks against the real local models (LM Studio + Qwen3 reranker).

Run with:  uv run pytest -m integration -q

These tests never call the paid DeepSeek endpoint: generation is driven by the
scripted LLM double, while embedding and reranking use the production components.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from qdrant_client import QdrantClient

from story_rag.adapters.model_factory import ModelBundle, build_embedding_model, build_reranker
from story_rag.config import Settings
from story_rag.container import build_container
from story_rag.domain import QueryRequest
from tests.conftest import CHINESE_STORY, ENGLISH_STORY, ScriptedLLM

pytestmark = pytest.mark.integration

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
RERANKER_PATH = Path(r"D:\text_Project\Agentic RAG\data\models\Qwen3-Reranker-0.6B")


def _lmstudio_ready() -> bool:
    try:
        response = httpx.get(f"{LMSTUDIO_BASE_URL}/models", timeout=3.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


@pytest.fixture
def real_settings(tmp_path, qdrant_url: str) -> Iterator[Settings]:
    api_file = tmp_path / "api.txt"
    api_file.write_text("sk-integration-placeholder\n模型: deepseek-flash\n", encoding="utf-8")
    data_dir = tmp_path / "data"
    configured = Settings(
        project_root=tmp_path,
        api_file=api_file,
        deepseek_base_url="https://api.deepseek.com",
        llm_model="deepseek-flash",
        llm_api_key="sk-integration-placeholder",
        lmstudio_base_url=LMSTUDIO_BASE_URL,
        reranker_model_path=RERANKER_PATH,
        data_dir=data_dir,
        qdrant_url=qdrant_url,
        collection_name=f"i_{uuid.uuid4().hex[:12]}",
        docstore_path=data_dir / "docstore" / "docstore.json",
        log_dir=tmp_path / "logs",
        chunk_size=200,
        chunk_overlap=32,
        rerank_top_n=2,
        query_fusion_num_queries=1,
    )
    yield configured
    with closing(QdrantClient(url=qdrant_url)) as client:
        if client.collection_exists(configured.collection_name):
            client.delete_collection(configured.collection_name)


@pytest.mark.skipif(not RERANKER_PATH.is_dir(), reason="local Qwen3 reranker is missing")
def test_reranker_ranks_relevant_passage_first(real_settings: Settings) -> None:
    reranker = build_reranker(real_settings)

    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    nodes = [
        NodeWithScore(node=TextNode(text="A boy bought a bicycle.", id_="n1"), score=0.0),
        NodeWithScore(
            node=TextNode(text="The white dragon brought rain to the village.", id_="n2"),
            score=0.0,
        ),
    ]
    ranked = reranker.postprocess_nodes(
        nodes,
        query_bundle=QueryBundle("What did the white dragon bring to the village?"),
    )

    assert "rain" in ranked[0].node.get_content()


@pytest.mark.skipif(not _lmstudio_ready(), reason="LM Studio embedding endpoint is unreachable")
def test_embedding_model_returns_vectors(real_settings: Settings) -> None:
    embedding = build_embedding_model(real_settings)

    vector = embedding.get_text_embedding("白龙带来了雨水")

    assert len(vector) > 100
    assert any(value != 0 for value in vector)


@pytest.mark.skipif(not _lmstudio_ready(), reason="LM Studio embedding endpoint is unreachable")
@pytest.mark.skipif(not RERANKER_PATH.is_dir(), reason="local Qwen3 reranker is missing")
def test_end_to_end_ingest_and_retrieve_with_local_models(real_settings: Settings) -> None:
    llm = ScriptedLLM()
    container = build_container(
        real_settings,
        models=ModelBundle(
            llm=llm,
            embedding=build_embedding_model(real_settings),
            reranker=build_reranker(real_settings),
        ),
    )

    container.ingestion.ingest_bytes(CHINESE_STORY.encode("utf-8"), source_name="zh.txt")
    container.ingestion.ingest_bytes(ENGLISH_STORY.encode("utf-8"), source_name="en.txt")
    result = container.retrieval.retrieve_trace(QueryRequest(query="白龙把雨水带给了谁？"))

    assert result.nodes
    assert any("白龙" in node.text for node in result.nodes)
    assert result.bm25_candidates

    llm.responses = ["无"]
    answer = container.generation.answer(QueryRequest(query="白龙把雨水带给了谁？"))
    assert answer.answer == "无"
    assert len(llm.prompts) == 1
