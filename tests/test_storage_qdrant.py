from __future__ import annotations

import asyncio
import uuid
from contextlib import closing

import pytest
from llama_index.core import QueryBundle
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from qdrant_client import QdrantClient, models

from story_rag.adapters.storage import DENSE_VECTOR_NAME, build_storage
from story_rag.domain import QueryRequest
from tests.conftest import ScriptedLLM


def test_qdrant_schema_uses_actual_dimension_and_euclid(container) -> None:
    info = container.storage.client.get_collection(container.settings.collection_name)
    dense = info.config.params.vectors[DENSE_VECTOR_NAME]

    assert dense.size == 8
    assert dense.distance == models.Distance.EUCLID
    assert container.storage.vector_store.enable_hybrid is False


def test_euclid_scores_preserve_nearest_first_in_native_rrf(container) -> None:
    nodes = [
        TextNode(
            id_=str(uuid.uuid4()), text=f"distance {distance}", embedding=[distance] + [0.0] * 7
        )
        for distance in (0.0, 1.0, 2.0)
    ]
    container.storage.index.insert_nodes(nodes)
    query = QueryBundle(query_str="distance", embedding=[0.0] * 8)
    vector = container.storage.index.as_retriever(similarity_top_k=3)
    synchronous = vector.retrieve(query)
    asynchronous = asyncio.run(vector.aretrieve(query))
    assert [item.node.node_id for item in synchronous] == [node.node_id for node in nodes]
    assert [item.score for item in synchronous] == pytest.approx([1.0, 0.5, 1.0 / 3])
    assert [item.score for item in asynchronous] == pytest.approx([1.0, 0.5, 1.0 / 3])
    fused = QueryFusionRetriever(
        [vector],
        mode="reciprocal_rerank",
        num_queries=1,
        similarity_top_k=3,
        use_async=True,
        llm=ScriptedLLM(),
    ).retrieve(query)
    assert [item.node.node_id for item in fused] == [node.node_id for node in nodes]


def test_sync_and_async_clients_share_points(container) -> None:
    container.ingestion.ingest_bytes(
        "问题：What is RRF？\n答案：倒数排名融合。".encode(), source_name="rrf.txt"
    )
    query = VectorStoreQuery(query_embedding=[0.5] * 8, similarity_top_k=5)
    sync = container.storage.vector_store.query(query)
    async_result = asyncio.run(container.storage.vector_store.aquery(query))
    another_loop = asyncio.run(container.storage.vector_store.aquery(query))

    assert sync.ids == async_result.ids == another_loop.ids
    assert len(sync.ids) == 1
    assert sync.nodes[0].metadata == async_result.nodes[0].metadata


def test_repeated_async_fusion_preserves_uuid_identity(container) -> None:
    container.ingestion.ingest_bytes(
        "问题：What is RRF？\n答案：倒数排名融合。".encode(), source_name="rrf.txt"
    )
    for _ in range(3):
        result = container.retrieval.retrieve_trace(QueryRequest(query="What is RRF？"))
        assert result.vector_candidates[0].node_id == result.bm25_candidates[0].node_id
        assert len(result.fused_candidates) == 1


def test_incompatible_existing_collection_is_not_recreated(settings) -> None:
    with closing(QdrantClient(url=settings.qdrant_url)) as client:
        client.create_collection(
            settings.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(size=8, distance=models.Distance.COSINE)
            },
        )
        with pytest.raises(ValueError, match="Euclid"):
            build_storage(settings, MockEmbedding(embed_dim=8))
        assert (
            client.get_collection(settings.collection_name)
            .config.params.vectors[DENSE_VECTOR_NAME]
            .distance
            == models.Distance.COSINE
        )


def test_dimension_change_requires_reimport(container, settings) -> None:
    with pytest.raises(ValueError, match="dimension"):
        build_storage(settings, MockEmbedding(embed_dim=16))
    assert container.storage.is_ready()


def test_docstore_without_collection_fails_before_creation(container, settings) -> None:
    container.ingestion.ingest_bytes(b"Q: What is RRF?\nA: Rank fusion.", source_name="a.txt")
    container.storage.client.delete_collection(settings.collection_name)

    with pytest.raises(ValueError, match="Document store exists"):
        build_storage(settings, MockEmbedding(embed_dim=8))
    assert not container.storage.client.collection_exists(settings.collection_name)


def test_missing_points_mark_not_ready(container) -> None:
    result = container.ingestion.ingest_bytes(
        b"Q: What is RRF?\nA: Rank fusion.", source_name="a.txt"
    )
    container.storage.vector_store.delete(result.document_id)

    assert container.storage.is_ready() is False


def test_equal_counts_with_wrong_ids_refuse_startup(container, settings) -> None:
    container.ingestion.ingest_bytes(b"Q: What is RRF?\nA: Rank fusion.", source_name="a.txt")
    client = container.storage.client
    points, _ = client.scroll(settings.collection_name, with_vectors=True)
    original = points[0]
    client.delete(settings.collection_name, points_selector=[original.id], wait=True)
    client.upsert(
        settings.collection_name,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()), vector=original.vector, payload=original.payload
            )
        ],
        wait=True,
    )

    with pytest.raises(ValueError, match="point IDs"):
        build_storage(settings, MockEmbedding(embed_dim=8))


def test_delete_document_updates_both_stores_and_persists(container, settings) -> None:
    result = container.ingestion.ingest_bytes(
        b"Q: What is RRF?\nA: Rank fusion.", source_name="a.txt"
    )
    revision = container.storage.revision
    container.storage.delete_document(result.document_id)
    container.storage.persist()

    assert container.storage.revision == revision + 1
    assert container.storage.document_count() == 0
    assert container.storage.node_count() == 0
    assert container.storage.client.count(settings.collection_name, exact=True).count == 0
    assert container.storage.is_ready()
