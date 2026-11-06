from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

from story_rag.adapters.model_factory import ModelBundle
from story_rag.api import create_app
from story_rag.config import Settings
from story_rag.container import build_container
from tests.conftest import CHINESE_STORY, ScriptedLLM

CHINESE_ANSWER = "白龙穿过云层，把雨水带给干涸的村庄。"


@pytest.fixture
def client(container) -> TestClient:
    return TestClient(
        create_app(container),
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_reports_models(client: TestClient) -> None:
    payload = client.get("/ready").json()

    assert payload["status"] == "ready"
    assert payload["llm_model"] == "deepseek-flash"
    assert payload["embedding_model"] == "text-embedding-qwen3-embedding-0.6b"
    assert payload["reranker_model"] == "missing-reranker"
    assert payload["debug_enabled"] is True
    assert "api" not in payload
    assert payload["vector_backend"] == "qdrant"
    assert payload["database_ready"] is True


def test_ready_returns_503_when_database_is_unreachable(client, container, monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise ConnectionError("database unavailable")

    monkeypatch.setattr(container.storage.client, "get_collection", unavailable)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["database_ready"] is False
    assert client.get("/health").status_code == 200


def test_ready_returns_503_when_collection_disappears(client, container) -> None:
    container.storage.client.delete_collection(container.settings.collection_name)

    assert client.get("/ready").status_code == 503


def test_ingest_endpoint_accepts_txt(client: TestClient) -> None:
    response = client.post(
        "/v1/stories/ingest",
        files={"file": ("dragon.txt", CHINESE_STORY.encode("utf-8"), "text/plain")},
        data={"title": "白龙传说", "user_instruction": "写一个关于龙的故事"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["duplicate"] is False
    assert payload["title"] == "白龙传说"
    assert payload["node_count"] >= 1


def test_qa_ingest_endpoint_accepts_txt(client: TestClient) -> None:
    response = client.post(
        "/v1/qa/ingest",
        files={
            "file": ("qa.txt", "问题：什么是 RRF？\n答案：倒数排名融合。".encode(), "text/plain")
        },
    )

    assert response.status_code == 200
    assert response.json()["node_count"] == 1


def test_ingest_endpoint_rejects_non_txt(client: TestClient) -> None:
    response = client.post(
        "/v1/stories/ingest",
        files={"file": ("dragon.md", CHINESE_STORY.encode("utf-8"), "text/markdown")},
    )

    assert response.status_code == 415


def test_ingest_endpoint_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/v1/stories/ingest",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 422


def test_query_endpoint_returns_answer(client: TestClient, scripted_llm: ScriptedLLM) -> None:
    client.post(
        "/v1/stories/ingest",
        files={"file": ("dragon.txt", CHINESE_STORY.encode("utf-8"), "text/plain")},
    )
    scripted_llm.responses = [CHINESE_ANSWER, "Yes"]

    response = client.post("/v1/chat/query", json={"query": "白龙把雨水带给了谁？"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == CHINESE_ANSWER
    assert payload["status"] == "answered"
    assert payload["request_id"]
    assert response.headers["x-request-id"]


def test_query_endpoint_returns_no_when_corpus_is_empty(client: TestClient) -> None:
    response = client.post("/v1/chat/query", json={"query": "白龙是谁？"})

    assert response.status_code == 200
    assert response.json()["answer"] == "无"


def test_debug_routes_are_registered_when_enabled(client: TestClient) -> None:
    payload = client.get("/debug/v1/config").json()

    assert payload["llm_model"] == "deepseek-flash"
    assert "api_key" not in payload
    assert payload["vector_backend"] == "qdrant"
    assert "chroma_dir" not in payload


def test_debug_input_does_not_generate(
    client: TestClient, scripted_llm: ScriptedLLM
) -> None:
    client.post(
        "/v1/qa/ingest",
        files={"file": ("qa.txt", "问题：谁带来雨水？\n答案：白龙。".encode(), "text/plain")},
    )

    response = client.post("/debug/v1/generation/input", json={"query": "谁带来雨水？"})

    assert response.status_code == 200
    assert response.json()["would_call_llm"] is True
    assert "答案：白龙。" in response.json()["prompt"]
    assert scripted_llm.prompts == []


def test_debug_input_rejects_query_rewriting(
    settings: Settings, scripted_llm: ScriptedLLM, reranker
) -> None:
    rewriting = build_container(
        dataclasses.replace(settings, query_fusion_num_queries=3),
        models=ModelBundle(
            llm=scripted_llm,
            embedding=MockEmbedding(embed_dim=8),
            reranker=reranker,
        ),
    )

    response = TestClient(create_app(rewriting)).post(
        "/debug/v1/generation/input", json={"query": "什么是 RRF？"}
    )

    assert response.status_code == 409
    assert scripted_llm.prompts == []


def test_debug_routes_are_hidden_when_disabled(
    settings: Settings,
    scripted_llm: ScriptedLLM,
    reranker,
) -> None:
    locked = build_container(
        dataclasses.replace(settings, debug_enabled=False),
        models=ModelBundle(
            llm=scripted_llm,
            embedding=MockEmbedding(embed_dim=8),
            reranker=reranker,
        ),
    )
    locked_client = TestClient(create_app(locked))

    assert locked_client.get("/debug/v1/config").status_code == 404
    assert locked_client.get("/health").status_code == 200
