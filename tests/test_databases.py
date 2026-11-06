from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from llama_index.core.embeddings import MockEmbedding
from qdrant_client import QdrantClient

from story_rag.adapters.database_registry import FileLease, normalize_name
from story_rag.adapters.document_updates import validate_filename
from story_rag.adapters.model_factory import ModelBundle
from story_rag.adapters.storage import StorageBundle, StorageUnavailableError
from story_rag.cli_session import PreviewLLM, QuerySession, split_command
from story_rag.domain import DatabaseError
from story_rag.pipelines.databases import DatabasesPipeline
from story_rag.pipelines.documents import DocumentManagementError, DocumentManagementPipeline
from story_rag.pipelines.ingestion import StoryIngestionPipeline
from tests.conftest import KeywordReranker

QA = "问题：What is amber?\n答案：Amber is a yellow resin."
NEW_QA = "问题：What is cobalt?\n答案：Cobalt is a metal used in batteries."


@pytest.fixture
def databases(settings):
    manager = DatabasesPipeline(settings, embedding_factory=lambda _: MockEmbedding(embed_dim=8))
    yield manager
    try:
        records = manager.registry.read()
    except DatabaseError:
        records = []
    with closing(QdrantClient(url=settings.qdrant_url)) as client:
        for record in records:
            if record.collection_name.startswith("d_") and client.collection_exists(
                record.collection_name
            ):
                client.delete_collection(record.collection_name)


@contextmanager
def opened(databases, record):
    storage = databases.open(record)
    try:
        yield storage
    finally:
        asyncio.run(storage.aclose())


def seed(databases, name="A", content=QA, source="same.txt"):
    record = databases.create(name)
    with opened(databases, record) as storage:
        result = StoryIngestionPipeline(storage.settings, storage).ingest_bytes(
            content.encode(), source_name=source
        )
    return record, result.document_id


def session_for(databases):
    return QuerySession(
        databases,
        models_factory=lambda settings: ModelBundle(
            llm=PreviewLLM(),
            embedding=MockEmbedding(embed_dim=8),
            reranker=KeywordReranker(top_n=settings.rerank_top_n),
        ),
    )


@pytest.mark.parametrize(
    "name", ["", " ", "a" * 65, "a\x00b", "a\nb", "a\tb", "a\u200bb", "a\x7fb"]
)
def test_reject_invalid_database_names(name):
    with pytest.raises(DatabaseError, match="1-64"):
        normalize_name(name)


@pytest.mark.parametrize(
    "name,expected",
    [
        (" 算法面试 ", "算法面试"),
        ("e\u0301", "é"),
        ("Mixed-中文", "Mixed-中文"),
        ("x" * 64, "x" * 64),
    ],
)
def test_normalize_names(name, expected):
    assert normalize_name(name) == expected


def test_read_empty_registry_does_not_create_files(databases):
    assert databases.list_databases() == []
    assert not databases.registry.path.exists()


def test_create_has_independent_short_scope_and_native_schema(databases):
    record = databases.create("算法面试")
    info = databases.info(record)
    assert record.status == "ready" and record.embedding_dimension == 8
    assert info["node_count"] == info["document_count"] == 0
    assert "算法面试" not in record.docstore_path + record.collection_name
    assert len(record.collection_name) == 14
    assert "llm_api_key" not in databases.registry.path.read_text(encoding="utf-8")
    with opened(databases, record) as storage:
        assert storage.is_ready()


def test_duplicate_normalized_name_is_rejected(databases):
    databases.create("算法面试")
    with pytest.raises(DatabaseError) as error:
        databases.create(" 算法面试 ")
    assert error.value.code == "DB_NAME_EXISTS"


def test_duplicate_casefold_name_is_rejected(databases):
    databases.create("Algorithms")
    with pytest.raises(DatabaseError):
        databases.create("ALGORITHMS")


def test_resolve_requires_explicit_selector(databases):
    with pytest.raises(DatabaseError):
        databases.resolve()
    record = databases.create("A")
    with pytest.raises(DatabaseError):
        databases.resolve("A", record.db_id)
    assert databases.resolve(db_id=record.db_id).name == "A"


def test_corrupt_registry_fails_closed(databases):
    databases.registry.path.parent.mkdir(parents=True)
    databases.registry.path.write_text('{"version":99,"databases":[]}', encoding="utf-8")
    with pytest.raises(DatabaseError) as error:
        databases.list_databases()
    assert error.value.code == "STORAGE_INCONSISTENT"


def test_registry_rejects_outside_docstore(databases):
    record = databases.create("A")
    original = databases.registry.path.read_text(encoding="utf-8")
    payload = json.loads(original)
    payload["databases"][0]["docstore_path"] = "D:\\outside\\docstore.json"
    databases.registry.path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        with pytest.raises(DatabaseError):
            databases.list_databases()
    finally:
        databases.registry.path.write_text(original, encoding="utf-8")
    assert databases.resolve(db_id=record.db_id).name == "A"


def test_register_default_preview_and_actual_keep_storage(container, databases):
    container.storage.persist()
    original = container.storage.settings.docstore_path.read_bytes()
    preview = databases.register_default(dry_run=True)
    assert preview["dry_run"] and not databases.registry.path.exists()
    registered = databases.register_default()
    assert registered["name"] == "default" and registered["legacy"]
    assert Path(registered["docstore_path"]).read_bytes() == original
    assert databases.register_default()["status"] == "unchanged"


def test_register_missing_default_refuses_creation(databases):
    with pytest.raises(DatabaseError):
        databases.register_default()
    assert not databases.settings.docstore_path.exists()


def test_database_rename_preserves_identity_and_storage(databases):
    record, _ = seed(databases)
    preview = databases.rename_preview(record, "新名字")
    assert preview["new_name"] == "新名字" and databases.resolve("A").db_id == record.db_id
    changed = databases.rename(record, "新名字", confirmation=record.db_id)
    assert changed["db_id"] == record.db_id and changed["collection_name"] == record.collection_name
    with pytest.raises(DatabaseError):
        databases.resolve("A")
    assert databases.resolve("新名字").db_id == record.db_id


def test_database_rename_rejects_duplicate_and_wrong_confirmation(databases):
    first = databases.create("A")
    databases.create("B")
    with pytest.raises(DatabaseError):
        databases.rename(first, "B", confirmation=first.db_id)
    with pytest.raises(DatabaseError):
        databases.rename(first, "C", confirmation="wrong")
    assert databases.resolve("A").db_id == first.db_id


def test_database_rename_and_drop_respect_occupancy(databases):
    record = databases.create("A")
    with opened(databases, record):
        for operation in (
            lambda: databases.rename(record, "B", confirmation=record.db_id),
            lambda: databases.drop(record, confirmation=record.db_id),
        ):
            with pytest.raises(DatabaseError) as error:
                operation()
            assert error.value.code == "DB_BUSY"


def test_database_drop_only_removes_selected_pair(databases, tmp_path):
    source = tmp_path / "source.txt"
    source.write_text(QA, encoding="utf-8")
    first, _ = seed(databases, "A", source=str(source))
    second, _ = seed(databases, "B", content=NEW_QA)
    assert databases.delete_preview(first)["node_count"] == 1
    databases.drop(first, confirmation=first.db_id)
    assert source.read_text(encoding="utf-8") == QA
    assert not Path(first.docstore_path).exists()
    assert databases.info(second)["document_count"] == 1
    assert len(databases.list_databases()) == 1
    assert len(databases.list_databases(True)) == 2
    with pytest.raises(DatabaseError):
        databases.open(first)


def test_same_name_recreation_does_not_accept_old_confirmation(databases):
    first = databases.create("A")
    databases.drop(first, confirmation=first.db_id)
    second = databases.create("A")
    assert first.db_id != second.db_id and first.collection_name != second.collection_name
    with pytest.raises(DatabaseError):
        databases.drop(second, confirmation=first.db_id)
    assert databases.info(second)["collection_exists"]


@pytest.mark.parametrize("missing", ["collection", "docstore"])
def test_missing_registered_storage_never_recreated(databases, missing):
    record = databases.create("A")
    if missing == "collection":
        with closing(QdrantClient(url=databases.settings.qdrant_url)) as client:
            client.delete_collection(record.collection_name)
    else:
        Path(record.docstore_path).unlink()
    with pytest.raises(StorageUnavailableError):
        databases.open(record)
    assert (
        not databases.info(record)["collection_exists"]
        if missing == "collection"
        else not Path(record.docstore_path).exists()
    )


def test_embedding_configuration_change_blocks_open(databases):
    record = databases.create("A")
    other = DatabasesPipeline(replace(databases.settings, embedding_model="different"))
    with pytest.raises(DatabaseError):
        other.open(record)


def test_interrupted_creation_has_explicit_recovery(databases, monkeypatch):
    original = QdrantClient.create_payload_index
    with monkeypatch.context() as patch:
        patch.setattr(
            QdrantClient,
            "create_payload_index",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected")),
        )
        with pytest.raises(RuntimeError):
            databases.create("A")
    record = databases.resolve("A", ready=False)
    assert record.status == "creating"
    assert databases.list_databases()[0]["status"] == "creating"
    with pytest.raises(DatabaseError):
        databases.open(record)
    preview = databases.recovery_preview(record)
    assert preview["operation"]["kind"] == "create"
    assert QdrantClient.create_payload_index is original
    databases.recover(
        record, confirmation=record.db_id, operation_id=preview["operation"]["operation_id"]
    )
    assert databases.resolve("A").status == "ready"


def test_interrupted_drop_stays_closed_until_recovery(databases, monkeypatch):
    record, _ = seed(databases)
    with monkeypatch.context() as patch:
        patch.setattr(
            QdrantClient,
            "delete_collection",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected")),
        )
        with pytest.raises(RuntimeError):
            databases.drop(record, confirmation=record.db_id)
    pending = databases.resolve("A", ready=False)
    assert pending.status == "deleting" and Path(record.docstore_path).exists()
    operation = databases.recovery_preview(pending)["operation"]
    with pytest.raises(DatabaseError):
        databases.recover(pending, confirmation=record.db_id, operation_id="wrong")
    databases.recover(pending, confirmation=record.db_id, operation_id=operation["operation_id"])
    assert databases.resolve(db_id=record.db_id, ready=False).status == "deleted"


def test_atomic_registry_failure_preserves_previous_name(databases, monkeypatch):
    from story_rag.adapters import database_registry as module

    record = databases.create("A")
    previous = databases.registry.path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("injected")))
        with pytest.raises(OSError):
            databases.rename(record, "B", confirmation=record.db_id)
    assert databases.registry.path.read_bytes() == previous


def test_cross_process_lock_and_release(tmp_path):
    path = tmp_path / "scope.lock"
    code = (
        "from pathlib import Path; from story_rag.adapters.database_registry import FileLease; "
        "from story_rag.domain import DatabaseError; import sys\n"
        "try:\n FileLease(Path(sys.argv[1])).acquire()\nexcept DatabaseError:\n sys.exit(4)"
    )
    with FileLease(path):
        result = subprocess.run(
            [sys.executable, "-c", code, str(path)], capture_output=True, timeout=30
        )
        assert result.returncode == 4, result.stderr
    with FileLease(path):
        pass


@pytest.mark.parametrize(
    "name",
    ["../a.txt", "D:\\a.txt", "CON.txt", "a?.txt", "a.md", "a\x00.txt", ".txt", "a" * 256 + ".txt"],
)
def test_invalid_txt_display_name(name):
    with pytest.raises(DatabaseError):
        validate_filename(name)


def test_txt_rename_changes_metadata_not_vectors(databases, monkeypatch):
    record, document_id = seed(databases)
    with opened(databases, record) as storage:
        manager = DocumentManagementPipeline(storage)
        old = manager.get_document(document_id)
        ids = storage.document_point_ids(document_id)
        vectors = storage.client.retrieve(record.collection_name, ids=list(ids), with_vectors=True)
        monkeypatch.setattr(
            MockEmbedding,
            "_get_text_embedding",
            lambda *a: (_ for _ in ()).throw(AssertionError("no embedding")),
        )
        manager.rename_document(
            document_id, "操作系统.txt", database_id=old.database_id, if_match=old.etag
        )
        detail = manager.get_document(document_id)
        assert detail.filename == "操作系统.txt" and detail.source == old.source
        assert detail.etag != old.etag and storage.document_point_ids(document_id) == ids
        after = storage.client.retrieve(record.collection_name, ids=list(ids), with_vectors=True)
        assert [p.vector for p in vectors] == [p.vector for p in after]
        with pytest.raises(DocumentManagementError):
            manager.delete_document(document_id, database_id=old.database_id, if_match=old.etag)
    with opened(databases, record) as restored:
        assert (
            DocumentManagementPipeline(restored).get_document(document_id).filename
            == "操作系统.txt"
        )


def test_reimport_preserves_user_display_name(databases):
    record, document_id = seed(databases)
    with opened(databases, record) as storage:
        docs = DocumentManagementPipeline(storage)
        detail = docs.get_document(document_id)
        docs.rename_document(
            document_id, "renamed.txt", database_id=detail.database_id, if_match=detail.etag
        )
        StoryIngestionPipeline(storage.settings, storage).ingest_bytes(
            NEW_QA.encode(), source_name="same.txt"
        )
        assert docs.get_document(document_id).filename == "renamed.txt"


def test_same_document_id_in_other_library_is_not_deleted(databases):
    first, document_id = seed(databases, "A")
    second, other_id = seed(databases, "B")
    assert document_id == other_id
    with opened(databases, first) as storage:
        docs = DocumentManagementPipeline(storage)
        detail = docs.get_document(document_id)
        docs.delete_document(document_id, database_id=detail.database_id, if_match=detail.etag)
    with opened(databases, second) as storage:
        assert DocumentManagementPipeline(storage).get_document(other_id).node_count == 1


def test_replace_preserves_identity_and_clears_old_evidence(databases, tmp_path):
    record, document_id = seed(databases)
    revision_input = tmp_path / "revision.txt"
    revision_input.write_text(NEW_QA, encoding="utf-8")
    session = session_for(databases)
    try:
        session.select(db_id=record.db_id)
        session.query("amber", trace=True)
        docs = session.documents()
        old = docs.get_document(document_id)
        docs.rename_document(
            document_id, "custom.txt", database_id=old.database_id, if_match=old.etag
        )
        renamed = docs.get_document(document_id)
        old_ids = session.storage.document_point_ids(document_id)
        docs.replace_document(
            document_id, revision_input, database_id=old.database_id, if_match=renamed.etag
        )
        result = session.query("cobalt", trace=True)
        assert "Cobalt" in result["llm_input"]["context_str"]
        assert "Amber" not in result["llm_input"]["context_str"]
        assert "custom.txt" in result["llm_input"]["context_str"]
        assert not old_ids & session.storage.document_point_ids(document_id)
        assert docs.get_document(document_id).source == old.source
        assert revision_input.read_text(encoding="utf-8") == NEW_QA
        assert result["llm_calls"] == 0
    finally:
        session.close()


@pytest.mark.parametrize(
    "kind,stage",
    [
        ("rename", "payload"),
        ("rename", "persist"),
        ("replace", "insert"),
        ("replace", "persist"),
        ("replace", "cleanup"),
    ],
)
def test_document_update_failure_blocks_then_recovers(
    databases, tmp_path, monkeypatch, kind, stage
):
    record, document_id = seed(databases)
    path = tmp_path / "revision.txt"
    path.write_text(NEW_QA, encoding="utf-8")
    with opened(databases, record) as storage:
        docs = DocumentManagementPipeline(storage)
        detail = docs.get_document(document_id)
        kwargs = {"database_id": detail.database_id, "if_match": detail.etag}
        with monkeypatch.context() as patch:
            if stage == "payload":
                patch.setattr(
                    storage.client,
                    "set_payload",
                    lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected")),
                )
            elif stage == "persist":
                patch.setattr(
                    StorageBundle, "persist", lambda *a: (_ for _ in ()).throw(OSError("injected"))
                )
            elif stage == "insert":
                patch.setattr(
                    storage.index,
                    "insert_nodes",
                    lambda *a: (_ for _ in ()).throw(RuntimeError("injected")),
                )
            else:
                original = Path.unlink

                def fail_cleanup(target, *args, **kwargs):
                    if target == storage.mutation_journal:
                        raise OSError("injected")
                    return original(target, *args, **kwargs)

                patch.setattr(Path, "unlink", fail_cleanup)
            with pytest.raises(StorageUnavailableError):
                if kind == "rename":
                    docs.rename_document(document_id, "changed.txt", **kwargs)
                else:
                    docs.replace_document(document_id, path, **kwargs)
        with pytest.raises(StorageUnavailableError):
            docs.list_documents()
        assert storage.mutation_journal.exists()
    with pytest.raises(StorageUnavailableError):
        databases.open(record)
    preview = databases.recovery_preview(record)
    databases.recover(
        record, confirmation=record.db_id, operation_id=preview["operation"]["operation_id"]
    )
    with opened(databases, record) as restored:
        assert restored.is_ready()
        detail = DocumentManagementPipeline(restored).get_document(document_id)
        assert (
            detail.filename == "changed.txt" if kind == "rename" else detail.filename == "same.txt"
        )
        assert not restored.mutation_journal.exists()


def test_session_select_switch_isolation_and_failed_switch(databases):
    first, _ = seed(databases, "A")
    second, _ = seed(databases, "B", NEW_QA)
    session = session_for(databases)
    try:
        with pytest.raises(DatabaseError):
            session.query("amber")
        session.select("A")
        first_result = session.query("amber", trace=True)
        assert first_result["db_id"] == first.db_id
        assert all(metric["calls"] == 1 for metric in first_result["stage_metrics"].values())
        models = session.models
        with pytest.raises(DatabaseError):
            session.select("missing")
        assert session.current()["name"] == "A"
        session.select("B")
        second_result = session.query("cobalt", trace=True)
        assert second_result["db_id"] == second.db_id and session.models is models
        assert "Amber" not in second_result["llm_input"]["context_str"]
        assert "Cobalt" in second_result["llm_input"]["context_str"]
        with opened(databases, first):
            pass
    finally:
        session.close()
    with opened(databases, second):
        pass


def test_session_empty_library_and_export(databases, tmp_path):
    record = databases.create("empty")
    session = session_for(databases)
    path = tmp_path / "input.json"
    try:
        session.select(db_id=record.db_id)
        result = session.query("question", export=path)
        assert result["status"] == "no_context" and result["llm_input"]["messages"] == []
        assert json.loads(path.read_text(encoding="utf-8"))["llm_calls"] == 0
        with pytest.raises(DatabaseError):
            session.query("question", export=path)
        session.release()
        with pytest.raises(DatabaseError):
            session.current()
    finally:
        session.close()


@pytest.mark.parametrize(
    "line,expected",
    [
        ('db use "算法 面试"', ["db", "use", "算法 面试"]),
        ('query "What is RRF?"', ["query", "What is RRF?"]),
        (
            'query "问题" --export "D:\\reports\\新 输入.json"',
            ["query", "问题", "--export", "D:\\reports\\新 输入.json"],
        ),
        ("query 'What is BM25?'", ["query", "What is BM25?"]),
    ],
)
def test_session_parser_preserves_quotes_and_windows_paths(line, expected):
    assert split_command(line) == expected


def test_database_info_exposes_occupancy(databases):
    record = databases.create("A")
    assert not databases.info(record)["busy"]
    with opened(databases, record):
        assert databases.info(record)["busy"]
    assert not databases.info(record)["busy"]


def test_open_rejects_forged_storage_binding(databases):
    record = databases.create("A")
    forged = record.model_copy(update={"collection_name": "other_collection"})
    with pytest.raises(DatabaseError) as error:
        databases.open(forged)
    assert error.value.code == "DB_ID_MISMATCH"


def test_recovery_rechecks_operation_identity_after_lock(databases, monkeypatch):
    import uuid

    record, document_id = seed(databases)
    with opened(databases, record) as storage:
        docs = DocumentManagementPipeline(storage)
        detail = docs.get_document(document_id)
        with monkeypatch.context() as patch:
            patch.setattr(
                storage.client,
                "set_payload",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected")),
            )
            with pytest.raises(StorageUnavailableError):
                docs.rename_document(
                    document_id, "changed.txt", database_id=detail.database_id, if_match=detail.etag
                )
    preview = databases.info(record)
    path = Path(record.docstore_path).with_suffix(".mutation.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    original = payload["operation_id"]
    payload["operation_id"] = str(uuid.uuid4())
    path.write_text(json.dumps(payload), encoding="utf-8")
    with monkeypatch.context() as patch:
        patch.setattr(databases, "info", lambda _: preview)
        with pytest.raises(DatabaseError) as error:
            databases.recover(record, confirmation=record.db_id, operation_id=original)
        assert error.value.code == "OPERATION_ID_MISMATCH"
    payload["operation_id"] = original
    path.write_text(json.dumps(payload), encoding="utf-8")
    databases.recover(record, confirmation=record.db_id, operation_id=original)


def test_preview_uses_configured_model_name_without_llm_call(databases):
    record, _ = seed(databases)
    databases.settings.api_file.write_text("model: example-preview-model\n", encoding="utf-8")
    session = session_for(databases)
    try:
        session.select("A")
        result = session.query("amber")
        assert result["llm_input"]["model"] == "example-preview-model"
        assert result["llm_calls"] == 0 and result["db_id"] == record.db_id
    finally:
        session.close()


def test_pending_file_delete_requires_explicit_recovery(databases, monkeypatch):
    record, document_id = seed(databases)
    with opened(databases, record) as storage:
        docs = DocumentManagementPipeline(storage)
        detail = docs.get_document(document_id)
        with monkeypatch.context() as patch:
            patch.setattr(
                StorageBundle, "persist", lambda *a: (_ for _ in ()).throw(OSError("injected"))
            )
            with pytest.raises(StorageUnavailableError):
                docs.delete_document(
                    document_id, database_id=detail.database_id, if_match=detail.etag
                )
        journal = storage.deletion_journal
    snapshot = journal.read_bytes()
    assert databases.info(record)["requires_recovery"]
    with pytest.raises(StorageUnavailableError, match="explicit db recover"):
        databases.open(record)
    preview = databases.recovery_preview(record)
    assert journal.read_bytes() == snapshot
    databases.recover(
        record, confirmation=record.db_id, operation_id=preview["operation"]["operation_id"]
    )
    with opened(databases, record) as restored:
        assert restored.is_ready() and restored.document_count() == 0
    assert not journal.exists()


@pytest.mark.parametrize("operation", ["rename", "drop"])
def test_mutation_uses_registered_scope_lock(databases, operation):
    record = databases.create("A")
    forged = record.model_copy(update={"collection_name": "unrelated"})
    with opened(databases, record):
        with pytest.raises(DatabaseError) as error:
            if operation == "rename":
                databases.rename(forged, "B", confirmation=record.db_id)
            else:
                databases.drop(forged, confirmation=record.db_id)
        assert error.value.code == "DB_BUSY"
    assert databases.resolve("A").db_id == record.db_id


def test_session_preview_model_name_is_not_hot_reloaded(databases):
    seed(databases)
    databases.settings.api_file.write_text("model: initial-preview-model\n", encoding="utf-8")
    session = session_for(databases)
    try:
        session.select("A")
        databases.settings.api_file.write_text("model: changed-preview-model\n", encoding="utf-8")
        assert session.query("amber")["llm_input"]["model"] == "initial-preview-model"
        assert session.query("amber")["llm_calls"] == 0
    finally:
        session.close()
