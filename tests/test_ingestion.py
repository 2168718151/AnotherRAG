from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.vector_stores.types import VectorStoreQuery

from story_rag.config import Settings
from tests.conftest import CHINESE_STORY, ENGLISH_STORY


def test_ingest_creates_one_document_with_chunks(container) -> None:
    result = container.ingestion.ingest_bytes(
        CHINESE_STORY.encode("utf-8"),
        source_name="stories/dragon.txt",
        user_instruction="写一个关于龙的故事",
    )

    assert result.duplicate is False
    assert result.replaced is False
    assert result.node_count >= 2
    assert container.storage.document_count() == 1
    assert container.storage.node_count() == result.node_count

    document = next(iter(container.storage.docstore.docs.values()))
    assert document.metadata["user_instruction"] == "写一个关于龙的故事"
    assert document.metadata["source"] == "stories/dragon.txt"
    assert document.metadata["encoding"] == "utf-8"


def test_ingest_persists_docstore_and_reloads(container, settings: Settings) -> None:
    container.ingestion.ingest_bytes(ENGLISH_STORY.encode("utf-8"), source_name="valley.txt")
    assert settings.docstore_path.is_file()

    from story_rag.adapters.storage import build_storage

    reloaded = build_storage(settings, MockEmbedding(embed_dim=8))

    assert reloaded.document_count() == 1
    assert reloaded.node_count() == container.storage.node_count()
    result = reloaded.index.as_retriever(similarity_top_k=3).retrieve("valley")
    assert result
    assert all(node.node.node_id in reloaded.docstore.docs for node in result)
    asyncio.run(reloaded.aclose())


def test_duplicate_ingest_is_skipped(container) -> None:
    first = container.ingestion.ingest_bytes(CHINESE_STORY.encode("utf-8"), source_name="d.txt")
    second = container.ingestion.ingest_bytes(CHINESE_STORY.encode("utf-8"), source_name="d.txt")

    assert first.duplicate is False
    assert second.duplicate is True
    assert second.content_hash == first.content_hash
    assert second.node_count == first.node_count
    assert container.storage.document_count() == 1
    assert container.storage.node_count() == first.node_count
    assert container.storage.client.count(container.settings.collection_name, exact=True).count == (
        first.node_count
    )


def test_changed_content_replaces_old_nodes(container) -> None:
    container.ingestion.ingest_bytes(
        "旧版本：白龙住在山谷的云雾里。".encode(),
        source_name="dragon.txt",
    )
    replaced = container.ingestion.ingest_bytes(
        "新版本：黑龙搬到了海边，山谷里再没有龙。".encode(),
        source_name="dragon.txt",
    )

    assert replaced.replaced is True
    assert container.storage.document_count() == 1

    stored = container.storage.vector_store.query(
        VectorStoreQuery(query_embedding=[0.0] * 8, similarity_top_k=20)
    )
    texts = [node.get_content() for node in stored.nodes]

    assert texts
    assert all("新版本" in text for text in texts)
    assert all("旧版本" not in text for text in texts)


def test_ingest_directory_reads_txt_only(container, tmp_path) -> None:
    folder = tmp_path / "stories"
    folder.mkdir()
    (folder / "a.txt").write_bytes(CHINESE_STORY.encode("utf-8"))
    (folder / "b.txt").write_bytes(CHINESE_STORY.encode("gb18030"))
    (folder / "skip.md").write_text("not a story", encoding="utf-8")

    results = container.ingestion.ingest_directory(folder)

    assert len(results) == 2
    assert container.storage.document_count() == 2
    assert {Path(item.source_name).name for item in results} == {"a.txt", "b.txt"}
    encodings = {
        Path(item.source_name).name: container.storage.docstore.get_document(
            item.document_id
        ).metadata["encoding"]
        for item in results
    }
    assert encodings["b.txt"] == "gb18030"


def test_debug_port_returns_nodes_without_persisting(container) -> None:
    inspection = container.ingestion.inspect_bytes(
        ENGLISH_STORY.encode("utf-8"),
        source_name="inspect.txt",
    )

    assert inspection["encoding"] == "utf-8"
    assert inspection["node_count"] >= 2
    assert len(inspection["nodes"]) == inspection["node_count"]
    assert container.storage.document_count() == 0


def test_qa_txt_indexes_pairs_and_keeps_original_document(container) -> None:
    text = (
        "标题：面试记录\n正文：未整理的访谈。\n\n技术问题 -> 答案\n\n"
        "问题：什么是向量检索？\n答案：比较文本向量的相似度。\n\n"
        "问题：什么是 BM25？\n答案：按词项统计相关性。"
    )

    result = container.ingestion.ingest_bytes(text.encode("utf-8"), source_name="qa.txt")
    nodes = sorted(container.storage.text_nodes(), key=lambda node: node.metadata["chunk_index"])
    document = container.storage.docstore.get_document(result.document_id)

    assert result.node_count == 2
    assert document.text == text
    assert all(node.metadata["content_type"] == "qa" for node in nodes)
    assert [node.metadata["question"] for node in nodes] == [
        "什么是向量检索？",
        "什么是 BM25？",
    ]
    assert all("未整理的访谈" not in node.text for node in nodes)


def test_chunk_ids_are_stable_uuids(container) -> None:
    text = "问题：What is RRF？\n答案：Reciprocal rank fusion。".encode()
    first = container.ingestion.inspect_bytes(text, source_name="ids.txt")
    second = container.ingestion.inspect_bytes(text, source_name="ids.txt")

    assert first["document_id"] == second["document_id"]
    assert len(first["document_id"]) == 64
    assert [node["node_id"] for node in first["nodes"]] == [
        node["node_id"] for node in second["nodes"]
    ]
    assert all(uuid.UUID(node["node_id"]).version == 5 for node in first["nodes"])
