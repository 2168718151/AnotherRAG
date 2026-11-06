from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

from story_rag.adapters.storage import StorageBundle, StorageUnavailableError, build_storage
from story_rag.api import create_app
from story_rag.domain import QueryRequest
from story_rag.pipelines.documents import DocumentManagementError, DocumentManagementPipeline

QA = "问题：What is RRF？\n答案：倒数排名融合。\n问题：What is BM25？\n答案：词频检索。"


def ingest(container, source="demo.txt", content=QA, **kwargs):
    return container.ingestion.ingest_bytes(content.encode(), source_name=source, **kwargs)


def delete(container, document_id):
    detail = container.documents.get_document(document_id)
    return container.documents.delete_document(
        document_id, database_id=detail.database_id, if_match=detail.etag
    )


@pytest.fixture
def local_client(container):
    with TestClient(
        create_app(container), base_url="http://127.0.0.1:18082", client=("127.0.0.1", 50000)
    ) as client:
        yield client


def test_empty_list(container):
    page = container.documents.list_documents()
    assert page.total == 0
    assert page.items == []
    assert page.database_id == container.storage.database_id
    assert page.collection_name == container.settings.collection_name


def test_lists_original_files_not_qa_nodes_or_custom_titles(container):
    item = ingest(container, r"C:\资料\中英.txt", title="custom title")
    page = container.documents.list_documents()
    assert page.total == 1
    assert page.items[0].filename == "中英.txt"
    assert page.items[0].node_count == item.node_count == 2
    assert page.items[0].status == "indexed"


def test_same_basename_distinct_paths_remain_distinct(container):
    first = ingest(container, r"C:\first\same.txt")
    second = ingest(container, r"C:\second\same.txt")
    assert first.document_id != second.document_id
    delete(container, first.document_id)
    page = container.documents.list_documents()
    assert [item.document_id for item in page.items] == [second.document_id]
    assert len(container.storage.document_point_ids(second.document_id)) == 2


def test_duplicate_and_replacement_stay_single_row(container):
    original = ingest(container)
    assert ingest(container).duplicate
    assert ingest(container, content="Q: Changed?\nA: New answer.").replaced
    assert container.documents.list_documents().total == 1
    assert container.documents.get_document(original.document_id).node_count == 1


def test_search_sort_and_pagination(container):
    ingest(container, "ZETA.txt")
    ingest(container, "alpha.txt")
    ingest(container, r"C:\中文\beta.txt")
    assert container.documents.list_documents(q="zeta").total == 1
    assert container.documents.list_documents(q="中文").items[0].filename == "beta.txt"
    page = container.documents.list_documents(offset=1, limit=1)
    assert page.total == 3
    assert page.items[0].filename == "beta.txt"
    assert not container.documents.list_documents(offset=10).items


def test_content_preview_is_truncated_and_keeps_version_identity(container, local_client):
    content = "问题：什么是 RRF？\n答案：" + ("倒数排名融合。" * 100)
    target = ingest(container, content=content)

    preview = container.documents.preview_document(target.document_id, max_characters=24)
    assert preview.text == content[:24]
    assert preview.total_characters == len(content)
    assert preview.truncated is True
    assert preview.database_id == container.storage.database_id

    response = local_client.get(f"/v1/documents/{target.document_id}/preview?max_characters=24")
    assert response.status_code == 200
    assert response.headers["etag"] == preview.etag
    assert response.json()["text"] == content[:24]
    assert (
        local_client.get(
            f"/v1/documents/{target.document_id}/preview?max_characters=20001"
        ).status_code
        == 422
    )


def test_frontend_is_disconnected_but_business_api_remains(local_client):
    for path in ("/", "/static/app.js", "/static/index.html", "/v1/ui/context"):
        assert local_client.get(path).status_code == 404
    assert "/v1/ui/context" not in local_client.get("/openapi.json").json()["paths"]
    assert local_client.get("/health").status_code == 200
    assert local_client.get("/ready").status_code == 200


def test_delete_cleans_native_index_docstore_vectors_and_preserves_source(container, tmp_path):
    source = tmp_path / "source.txt"
    source.write_text(QA, encoding="utf-8")
    target = container.ingestion.ingest_path(source)
    retained = ingest(container, "retained.txt")
    before = container.storage.revision
    detail = container.documents.delete_preview(target.document_id)
    assert detail.consistent
    result = delete(container, target.document_id)
    assert result.deleted_nodes == 2
    assert result.source_file_deleted is False
    assert source.read_text(encoding="utf-8") == QA
    assert container.storage.revision == before + 1
    assert not container.storage.deletion_journal.exists()
    assert target.document_id not in container.storage.docstore.docs
    assert container.storage.docstore.get_ref_doc_info(target.document_id) is None
    assert not set(detail.node_ids) & set(container.storage.index.index_struct.nodes_dict.values())
    assert container.storage.document_point_ids(retained.document_id)
    assert container.storage.is_ready()


def test_restart_has_no_deleted_records(container, settings):
    target = ingest(container)
    delete(container, target.document_id)
    restored = build_storage(settings, MockEmbedding(embed_dim=8))
    try:
        assert DocumentManagementPipeline(restored).list_documents().total == 0
        assert restored.is_ready()
    finally:
        asyncio.run(restored.aclose())


def test_delete_existing_document_after_restart_restores_native_index(container, settings):
    target = ingest(container)
    restored = build_storage(settings, MockEmbedding(embed_dim=8))
    try:
        manager = DocumentManagementPipeline(restored)
        detail = manager.get_document(target.document_id)
        manager.delete_document(
            target.document_id, database_id=detail.database_id, if_match=detail.etag
        )
        assert not restored.index.index_struct.nodes_dict
        assert manager.list_documents().total == 0
        assert restored.is_ready()
    finally:
        asyncio.run(restored.aclose())


def test_atomic_docstore_replace_failure_leaves_old_file(container, monkeypatch):
    from story_rag.adapters import storage as storage_module

    ingest(container)
    old = container.settings.docstore_path.read_bytes()

    def fail(*args):
        raise OSError("injected replace failure")

    monkeypatch.setattr(storage_module.os, "replace", fail)
    with pytest.raises(OSError):
        container.storage.persist()
    assert container.settings.docstore_path.read_bytes() == old
    assert not list(container.settings.docstore_path.parent.glob(".docstore.json.*"))


def test_cached_bm25_and_vector_rebuild_after_delete(container):
    target = ingest(container)
    retained = ingest(container, "kept.txt", "Q: Retained?\nA: Kept content.")
    assert container.retrieval.retrieve_trace(QueryRequest(query="RRF")).nodes
    old = container.retrieval._retrievers
    delete(container, target.document_id)
    result = container.retrieval.retrieve_trace(QueryRequest(query="RRF"))
    assert container.retrieval._retrievers is not old
    for nodes in [
        result.nodes,
        result.vector_candidates,
        result.bm25_candidates,
        result.fused_candidates,
    ]:
        assert all(node.metadata["document_id"] == retained.document_id for node in nodes)


def test_delete_last_document_returns_empty_retrieval(container):
    target = ingest(container)
    container.retrieval.retrieve(QueryRequest(query="RRF"))
    delete(container, target.document_id)
    assert not container.retrieval.retrieve(QueryRequest(query="RRF")).nodes


@pytest.mark.parametrize("document_id", ["missing", "a" * 64])
def test_unknown_document_not_found(container, document_id):
    with pytest.raises(DocumentManagementError) as caught:
        container.documents.get_document(document_id)
    assert caught.value.status_code == 404


def test_repeat_delete_is_not_found(container):
    target = ingest(container)
    detail = container.documents.get_document(target.document_id)
    delete(container, target.document_id)
    with pytest.raises(DocumentManagementError) as caught:
        container.documents.delete_document(
            target.document_id, database_id=detail.database_id, if_match=detail.etag
        )
    assert caught.value.status_code == 404


@pytest.mark.parametrize("changed", ["database", "version"])
def test_stale_confirmation_never_deletes(container, changed):
    target = ingest(container)
    detail = container.documents.get_document(target.document_id)
    if changed == "version":
        ingest(container, content="Q: Updated?\nA: New content.")
    with pytest.raises(DocumentManagementError) as caught:
        container.documents.delete_document(
            target.document_id,
            database_id="wrong" if changed == "database" else detail.database_id,
            if_match=detail.etag,
        )
    assert caught.value.status_code == (409 if changed == "database" else 412)
    assert container.storage.document_point_ids(target.document_id)


def test_management_does_not_call_models(container, monkeypatch):
    target = ingest(container)

    def forbidden(*args, **kwargs):
        pytest.fail("Management must not call embedding, reranker or LLM")

    monkeypatch.setattr(MockEmbedding, "get_text_embedding", forbidden)
    monkeypatch.setattr(type(container.llm), "complete", forbidden)
    monkeypatch.setattr(type(container.retrieval._reranker), "postprocess_nodes", forbidden)
    container.documents.list_documents()
    container.documents.get_document(target.document_id)
    container.documents.delete_preview(target.document_id)
    delete(container, target.document_id)


@pytest.mark.parametrize("failure", ["qdrant", "persist", "journal_cleanup"])
def test_partial_failure_blocks_operations_and_recovers_before_validation(
    container,
    settings,
    monkeypatch,
    failure,
):
    target = ingest(container)
    retained = ingest(container, "kept.txt")
    original_persisted = settings.docstore_path.read_bytes()
    journal = container.storage.deletion_journal

    def fail(*args, **kwargs):
        raise OSError("injected storage failure")

    original_unlink = Path.unlink

    def fail_journal(path, *args, **kwargs):
        if path == journal:
            raise OSError("injected journal cleanup failure")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        if failure == "qdrant":
            patch.setattr(type(container.storage.vector_store), "delete", fail)
        elif failure == "persist":
            patch.setattr(StorageBundle, "persist", fail)
        else:
            patch.setattr(Path, "unlink", fail_journal)
        with pytest.raises(StorageUnavailableError):
            delete(container, target.document_id)
    assert journal.exists()
    assert not container.storage.is_ready()
    for operation in [
        container.documents.list_documents,
        lambda: container.retrieval.retrieve(QueryRequest(query="RRF")),
        lambda: ingest(container, "blocked.txt"),
    ]:
        with pytest.raises(StorageUnavailableError):
            operation()
    if failure == "persist":
        assert settings.docstore_path.read_bytes() == original_persisted
    restored = build_storage(settings, MockEmbedding(embed_dim=8))
    try:
        page = DocumentManagementPipeline(restored).list_documents()
        assert [item.document_id for item in page.items] == [retained.document_id]
        assert not journal.exists()
        assert restored.is_ready()
    finally:
        asyncio.run(restored.aclose())


def test_corrupt_or_wrong_database_journal_refuses_recovery(container, settings):
    target = ingest(container)
    detail = container.documents.delete_preview(target.document_id)
    container.storage.deletion_journal.write_text(
        json.dumps(
            {
                "version": 1,
                "database_id": "wrong",
                "document_id": target.document_id,
                "node_ids": detail.node_ids,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StorageUnavailableError, match="journal"):
        build_storage(settings, MockEmbedding(embed_dim=8))
    assert container.storage.document_point_ids(target.document_id) == set(detail.node_ids)


def test_qdrant_scope_mismatch_refuses_deletion(container):
    target = ingest(container)
    container.storage.vector_store.delete(target.document_id)
    assert not container.documents.delete_preview(target.document_id).consistent
    with pytest.raises(StorageUnavailableError, match="differ"):
        delete(container, target.document_id)
    assert target.document_id in container.storage.docstore.docs
    assert not container.storage.deletion_journal.exists()


def test_reads_wait_for_delete_and_never_see_partial_state(container, monkeypatch):
    target = ingest(container)
    retained = ingest(container, "kept.txt")
    entered = threading.Event()
    release = threading.Event()
    original = type(container.storage.vector_store).delete

    def delayed(store, *args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(store, *args, **kwargs)

    monkeypatch.setattr(type(container.storage.vector_store), "delete", delayed)
    with ThreadPoolExecutor(max_workers=2) as pool:
        deletion = pool.submit(delete, container, target.document_id)
        assert entered.wait(5)
        read = pool.submit(container.retrieval.retrieve_trace, QueryRequest(query="RRF"))
        try:
            with pytest.raises(TimeoutError):
                read.result(timeout=0.05)
        finally:
            release.set()
        deletion.result(timeout=5)
        result = read.result(timeout=5)
    assert all(node.metadata["document_id"] == retained.document_id for node in result.nodes)


def test_ingestion_and_management_share_mutation_lock(container):
    with ThreadPoolExecutor(max_workers=1) as pool:
        with container.storage.lock:
            work = pool.submit(ingest, container)
            with pytest.raises(TimeoutError):
                work.result(timeout=0.05)
        work.result(timeout=5)
    assert container.documents.list_documents().total == 1


def test_api_list_detail_preview_and_delete(container, local_client):
    target = ingest(container)
    page = local_client.get("/v1/documents").json()
    assert page["total"] == 1
    url = f"/v1/documents/{target.document_id}"
    detail = local_client.get(url)
    assert detail.headers["etag"] == detail.json()["etag"]
    preview = local_client.get(f"/debug/v1/documents/{target.document_id}/delete-preview")
    assert preview.json()["consistent"] is True
    assert container.storage.document_count() == 1
    result = local_client.delete(
        url,
        headers={
            "X-Database-ID": page["database_id"],
            "If-Match": detail.headers["etag"],
        },
    )
    assert result.status_code == 200
    assert result.json()["deleted_nodes"] == 2
    assert (
        local_client.delete(
            url,
            headers={
                "X-Database-ID": page["database_id"],
                "If-Match": detail.headers["etag"],
            },
        ).status_code
        == 404
    )
    assert local_client.get("/v1/documents").json()["total"] == 0


def test_legacy_file_management_routes_are_not_served(local_client):
    for path in [
        "/files",
        "/assets/files",
        "/assets/files/index.html",
        "/assets/files/files.js",
        "/assets/files/files.css",
        "/assets/files/lucide.min.js",
    ]:
        assert local_client.get(path).status_code == 404


@pytest.mark.parametrize(
    "headers,code",
    [
        ({}, 428),
        ({"X-Database-ID": "wrong", "If-Match": '"stale"'}, 409),
        ({"If-Match": '"stale"'}, 428),
    ],
)
def test_api_preconditions(container, local_client, headers, code):
    target = ingest(container)
    response = local_client.delete(f"/v1/documents/{target.document_id}", headers=headers)
    assert response.status_code == code
    assert container.storage.document_count() == 1


def test_api_changed_version_returns_412(container, local_client):
    target = ingest(container)
    detail = container.documents.get_document(target.document_id)
    ingest(container, content="Q: Changed?\nA: Updated.")
    assert (
        local_client.delete(
            f"/v1/documents/{target.document_id}",
            headers={
                "X-Database-ID": detail.database_id,
                "If-Match": detail.etag,
            },
        ).status_code
        == 412
    )


@pytest.mark.parametrize("query", ["limit=0", "limit=201", "offset=-1"])
def test_api_invalid_pagination(local_client, query):
    assert local_client.get(f"/v1/documents?{query}").status_code == 422


@pytest.mark.parametrize("kind", ["remote", "host", "origin"])
def test_management_rejects_remote_rebinding_and_cross_origin(container, kind):
    app = create_app(container)
    with TestClient(
        app,
        base_url="http://127.0.0.1:18082",
        client=("192.168.1.8" if kind == "remote" else "127.0.0.1", 50000),
    ) as client:
        headers = (
            {"host": "attacker.example"}
            if kind == "host"
            else ({"origin": "https://attacker.example"} if kind == "origin" else {})
        )
        assert client.get("/v1/documents", headers=headers).status_code == 403
        assert client.delete("/v1/documents/missing", headers=headers).status_code == 403
        assert client.get("/", headers=headers).status_code == 404
        assert client.get("/static/app.js", headers=headers).status_code == 404
        for path in ("/v1/qa/ingest", "/v1/stories/ingest"):
            assert (
                client.post(
                    path, headers=headers, files={"file": ("blocked.txt", b"blocked")}
                ).status_code
                == 403
            )


def test_debug_document_preview_disabled(container):
    container.settings = replace(container.settings, debug_enabled=False)
    with TestClient(
        create_app(container), base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        assert client.get("/debug/v1/documents/missing/delete-preview").status_code == 404


def test_storage_failure_returns_503_not_false_success(container, local_client, monkeypatch):
    target = ingest(container)
    detail = container.documents.get_document(target.document_id)

    def fail(*args, **kwargs):
        raise OSError("injected")

    monkeypatch.setattr(StorageBundle, "persist", fail)
    result = local_client.delete(
        f"/v1/documents/{target.document_id}",
        headers={
            "X-Database-ID": detail.database_id,
            "If-Match": detail.etag,
        },
    )
    assert result.status_code == 503
    assert local_client.get("/v1/documents").status_code == 503
    assert local_client.get("/ready").status_code == 503
