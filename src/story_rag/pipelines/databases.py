from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

from llama_index.core.embeddings import MockEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from qdrant_client import QdrantClient, models

from story_rag.adapters.database_registry import (
    DatabaseRecord,
    DatabaseRegistry,
    FileLease,
    normalize_name,
    storage_fingerprint,
)
from story_rag.adapters.storage import (
    DENSE_VECTOR_NAME,
    StorageBundle,
    _persist_docstore,
    _validate_collection,
    _validate_node_ids,
    build_storage,
)
from story_rag.config import Settings
from story_rag.domain import DatabaseError

logger = logging.getLogger(__name__)


def qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
    )


class DatabasesPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: Callable = qdrant_client,
        embedding_factory: Callable | None = None,
    ) -> None:
        self.settings = settings
        self.registry = DatabaseRegistry(settings)
        self.client_factory = client_factory
        self.embedding_factory = embedding_factory

    def resolve(
        self, name: str | None = None, db_id: str | None = None, *, ready: bool = True
    ) -> DatabaseRecord:
        return self.registry.resolve(name, db_id, ready=ready)

    def list_databases(self, include_deleted: bool = False) -> list[dict[str, Any]]:
        return [
            r.model_dump() for r in self.registry.read() if include_deleted or r.status != "deleted"
        ]

    def delete_preview(self, record: DatabaseRecord) -> dict[str, Any]:
        return self.drop(record, confirmation="", dry_run=True)

    def rename_preview(self, record: DatabaseRecord, name: str) -> dict[str, Any]:
        return self.rename(record, name, confirmation="", dry_run=True)

    def recovery_preview(self, record: DatabaseRecord) -> dict[str, Any]:
        return self.recover(record, dry_run=True)

    def info(self, record: DatabaseRecord) -> dict[str, Any]:
        record = self.resolve(db_id=record.db_id, ready=False)
        settings = self.registry.bind(record)
        result = record.model_dump()
        result["storage_fingerprint"] = storage_fingerprint(settings)
        result["busy"] = self.registry.is_busy(record)
        path = settings.docstore_path
        mutation = path.with_suffix(".mutation.json")
        deletion = path.with_suffix(".delete.json")
        if mutation.exists():
            operation = json.loads(mutation.read_text(encoding="utf-8"))
            result["operation"] = {k: operation[k] for k in ("operation_id", "kind", "document_id")}
            result["requires_recovery"] = True
        elif deletion.exists():
            raw = deletion.read_bytes()
            operation = json.loads(raw)
            result["operation"] = {
                "operation_id": hashlib.sha256(raw).hexdigest(),
                "kind": "delete_document",
                "document_id": operation["document_id"],
            }
            result["requires_recovery"] = True
        else:
            result["requires_recovery"] = bool(
                record.operation or path.with_suffix(".delete.json").exists()
            )
        with closing(self.client_factory(settings)) as client:
            exists = client.collection_exists(record.collection_name)
            result["collection_exists"] = exists
            result["node_count"] = (
                client.count(record.collection_name, exact=True).count if exists else 0
            )
        result["document_count"] = 0
        if path.is_file():
            from llama_index.core import Document

            store = SimpleDocumentStore.from_persist_path(str(path))
            result["document_count"] = sum(isinstance(n, Document) for n in store.docs.values())
        return result

    def embedding(self) -> Any:
        if self.embedding_factory:
            return self.embedding_factory(self.settings)
        from story_rag.adapters.model_factory import build_embedding_model

        return build_embedding_model(self.settings)

    def create(self, name: str) -> DatabaseRecord:
        name = normalize_name(name)
        self.registry.check_name(self.registry.read(), name)
        dimension = len(self.embedding().get_text_embedding("embedding dimension probe"))
        if not dimension:
            raise DatabaseError("STORAGE_INCONSISTENT", "Embedding dimension is empty", 7)
        db_id = str(uuid.uuid4())
        short_id = db_id.replace("-", "")[:12]
        record = DatabaseRecord(
            db_id=db_id,
            name=name,
            collection_name=f"d_{short_id}",
            docstore_path=str(self.registry.root / short_id / "docstore.json"),
            qdrant_url=self.settings.qdrant_url.rstrip("/"),
            embedding_model=self.settings.embedding_model,
            embedding_base_url=self.settings.lmstudio_base_url,
            embedding_dimension=dimension,
            operation={"operation_id": str(uuid.uuid4()), "kind": "create"},
        )
        bound = self.registry.bind(record)
        with self.registry.lease(bound), closing(self.client_factory(bound)) as client:
            if client.collection_exists(record.collection_name) or bound.docstore_path.exists():
                raise DatabaseError(
                    "STORAGE_INCONSISTENT", "Generated storage scope already exists", 7
                )
            with self.registry.edit() as records:
                self.registry.check_name(records, name)
                if any(r.collection_name == record.collection_name for r in records):
                    raise DatabaseError(
                        "DB_NAME_EXISTS", "Generated scope collision; retry creation"
                    )
                records.append(record)
            return self._finish_create(record, client)

    def _finish_create(self, record: DatabaseRecord, client: Any) -> DatabaseRecord:
        bound = self.registry.bind(record)
        if not client.collection_exists(record.collection_name):
            client.create_collection(
                record.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=record.embedding_dimension, distance=models.Distance.EUCLID
                    )
                },
            )
        _validate_collection(
            client.get_collection(record.collection_name), record.embedding_dimension
        )
        if client.count(record.collection_name, exact=True).count:
            raise DatabaseError("STORAGE_INCONSISTENT", "Creating database must be empty", 7)
        client.create_payload_index(
            record.collection_name,
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        if bound.docstore_path.exists():
            if SimpleDocumentStore.from_persist_path(str(bound.docstore_path)).docs:
                raise DatabaseError("STORAGE_INCONSISTENT", "Creating docstore is not empty", 7)
        else:
            _persist_docstore(SimpleDocumentStore(), bound.docstore_path)
        record = record.model_copy(update={"status": "ready", "operation": None})
        self.registry.update(record)
        logger.info("Database created: db_id=%s name=%r", record.db_id, record.name)
        return record

    def register_default(self, *, dry_run: bool = False) -> dict[str, Any]:
        bound = self.settings
        with self.registry.lease(bound), closing(self.client_factory(bound)) as client:
            if not bound.docstore_path.is_file() or not client.collection_exists(
                bound.collection_name
            ):
                raise DatabaseError("STORAGE_INCONSISTENT", "Legacy paired storage is missing", 7)
            store = SimpleDocumentStore.from_persist_path(str(bound.docstore_path))
            _validate_node_ids(
                client,
                bound.collection_name,
                {n.node_id for n in store.docs.values() if n.ref_doc_id},
            )
            vectors = client.get_collection(bound.collection_name).config.params.vectors
            dense = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
            if dense is None:
                raise DatabaseError("STORAGE_INCONSISTENT", "Legacy vector schema differs", 7)
            _validate_collection(client.get_collection(bound.collection_name), dense.size)
            if any(
                bound.docstore_path.with_suffix(suffix).exists()
                for suffix in (".delete.json", ".mutation.json")
            ):
                raise DatabaseError("RECOVERY_REQUIRED", "Legacy storage requires recovery", 7)
            for record in self.registry.read():
                if (
                    record.collection_name == bound.collection_name
                    or Path(record.docstore_path).resolve() == bound.docstore_path.resolve()
                ):
                    if (
                        record.name == "default"
                        and record.status == "ready"
                        and record.legacy
                        and record.collection_name == bound.collection_name
                        and Path(record.docstore_path).resolve() == bound.docstore_path.resolve()
                    ):
                        return {**record.model_dump(), "status": "unchanged"}
                    raise DatabaseError(
                        "STORAGE_INCONSISTENT", "Legacy storage is already registered", 7
                    )
            record = DatabaseRecord(
                db_id=str(uuid.uuid4()),
                name="default",
                collection_name=bound.collection_name,
                docstore_path=str(bound.docstore_path),
                qdrant_url=bound.qdrant_url.rstrip("/"),
                embedding_model=bound.embedding_model,
                embedding_base_url=bound.lmstudio_base_url,
                embedding_dimension=dense.size,
                status="ready",
                legacy=True,
            )
            self.registry.check_name(self.registry.read(), "default")
            if not dry_run:
                with self.registry.edit() as records:
                    self.registry.check_name(records, "default")
                    records.append(record)
            return {**record.model_dump(), "dry_run": dry_run}

    def confirm(self, record: DatabaseRecord, confirmation: str) -> None:
        if confirmation != record.db_id:
            raise DatabaseError("DB_ID_MISMATCH", "Database UUID confirmation does not match", 5)

    def rename(
        self, record: DatabaseRecord, name: str, *, confirmation: str, dry_run: bool = False
    ) -> dict[str, Any]:
        record = self.resolve(db_id=record.db_id)
        name = normalize_name(name)
        if dry_run:
            self.registry.check_name(self.registry.read(), name, record.db_id)
            return {
                **self.info(record),
                "db_id": record.db_id,
                "old_name": record.name,
                "new_name": name,
                "dry_run": True,
            }
        self.confirm(record, confirmation)
        with self.registry.lease(self.registry.bind(record)):
            record = self.resolve(db_id=record.db_id)
            bound = self.registry.bind(record)
            if any(
                bound.docstore_path.with_suffix(s).exists()
                for s in (".mutation.json", ".delete.json")
            ):
                raise DatabaseError(
                    "RECOVERY_REQUIRED", "Recover pending document operation first", 7
                )
            if not bound.docstore_path.is_file():
                raise DatabaseError("STORAGE_INCONSISTENT", "Registered docstore is missing", 7)
            with closing(self.client_factory(bound)) as client:
                store = SimpleDocumentStore.from_persist_path(str(bound.docstore_path))
                _validate_collection(
                    client.get_collection(record.collection_name), record.embedding_dimension
                )
                _validate_node_ids(
                    client,
                    record.collection_name,
                    {n.node_id for n in store.docs.values() if n.ref_doc_id},
                )
            with self.registry.edit() as records:
                self.registry.check_name(records, name, record.db_id)
                updated = record.model_copy(update={"name": name})
                for i, current in enumerate(records):
                    if current.db_id == record.db_id:
                        records[i] = updated
            return {
                **updated.model_dump(),
                "status": "unchanged" if name == record.name else "renamed",
            }

    def open(
        self,
        record: DatabaseRecord,
        *,
        embedding: Any = None,
        probe: bool = False,
        recover: bool = False,
        operation_id: str | None = None,
    ) -> StorageBundle:
        bound = self.registry.bind(record)
        lease = self.registry.lease(bound).acquire()
        try:
            current = self.resolve(db_id=record.db_id)
            if storage_fingerprint(bound) != storage_fingerprint(self.registry.bind(current)):
                raise DatabaseError("DB_ID_MISMATCH", "Database storage binding changed", 5)
            if current.operation:
                raise DatabaseError("RECOVERY_REQUIRED", "Database requires explicit recovery", 7)
            if (
                current.embedding_model != bound.embedding_model
                or current.embedding_base_url != bound.lmstudio_base_url
            ):
                raise DatabaseError("STORAGE_INCONSISTENT", "Embedding configuration changed", 7)
            if operation_id is not None:
                mutation = bound.docstore_path.with_suffix(".mutation.json")
                deletion = bound.docstore_path.with_suffix(".delete.json")
                actual = (
                    json.loads(mutation.read_text(encoding="utf-8"))["operation_id"]
                    if mutation.exists()
                    else hashlib.sha256(deletion.read_bytes()).hexdigest()
                    if deletion.exists()
                    else None
                )
                if actual != operation_id:
                    raise DatabaseError("OPERATION_ID_MISMATCH", "Recovery operation changed", 7)
            embedding = embedding or MockEmbedding(embed_dim=current.embedding_dimension)
            return build_storage(
                bound,
                embedding,
                lease=lease,
                embedding_dimension=None if probe else current.embedding_dimension,
                recover_mutations=recover,
            )
        except BaseException:
            lease.close()
            raise

    def legacy_lease(self) -> FileLease:
        lease = self.registry.lease(self.settings).acquire()
        try:
            for record in self.registry.read():
                if (
                    record.collection_name == self.settings.collection_name
                    or Path(record.docstore_path).resolve() == self.settings.docstore_path.resolve()
                ) and (record.status != "ready" or record.operation):
                    raise DatabaseError("DB_NOT_READY", "Registered legacy storage is not ready")
            return lease
        except BaseException:
            lease.close()
            raise

    def drop(
        self, record: DatabaseRecord, *, confirmation: str, dry_run: bool = False
    ) -> dict[str, Any]:
        record = self.resolve(db_id=record.db_id)
        if dry_run:
            return {**self.info(record), "dry_run": True}
        self.confirm(record, confirmation)
        bound = self.registry.bind(record)
        with self.registry.lease(bound), closing(self.client_factory(bound)) as client:
            record = self.resolve(db_id=record.db_id)
            if any(
                bound.docstore_path.with_suffix(s).exists()
                for s in (".mutation.json", ".delete.json")
            ):
                raise DatabaseError(
                    "RECOVERY_REQUIRED", "Recover pending document operation first", 7
                )
            record = record.model_copy(
                update={
                    "status": "deleting",
                    "operation": {"kind": "drop", "operation_id": str(uuid.uuid4())},
                }
            )
            self.registry.update(record)
            return self._finish_drop(record, client).model_dump()

    def _finish_drop(self, record: DatabaseRecord, client: Any) -> DatabaseRecord:
        bound = self.registry.bind(record)
        if client.collection_exists(record.collection_name):
            client.delete_collection(record.collection_name)
        if client.collection_exists(record.collection_name):
            raise DatabaseError("RECOVERY_REQUIRED", "Collection deletion did not finish", 7)
        if bound.docstore_path.is_symlink():
            raise DatabaseError("STORAGE_INCONSISTENT", "Refusing to delete a linked docstore", 7)
        bound.docstore_path.unlink(missing_ok=True)
        record = record.model_copy(update={"status": "deleted", "operation": None})
        self.registry.update(record)
        logger.info("Database deleted: db_id=%s", record.db_id)
        return record

    def recover(
        self,
        record: DatabaseRecord,
        *,
        confirmation: str = "",
        operation_id: str = "",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        record = self.resolve(db_id=record.db_id, ready=False)
        preview = self.info(record)
        operation = preview.get("operation")
        if not operation:
            return {**preview, "status": "unchanged", "dry_run": dry_run}
        if dry_run:
            return {**preview, "dry_run": True}
        self.confirm(record, confirmation)
        if operation.get("operation_id") != operation_id:
            raise DatabaseError("OPERATION_ID_MISMATCH", "Recovery operation identity differs", 7)
        if operation["kind"] in {"rename", "replace", "delete_document"}:
            storage = self.open(record, recover=True, operation_id=operation_id)
            asyncio.run(storage.aclose())
            return {"db_id": record.db_id, "status": "recovered", "operation_id": operation_id}
        with (
            self.registry.lease(self.registry.bind(record)),
            closing(self.client_factory(self.registry.bind(record))) as client,
        ):
            current = self.resolve(db_id=record.db_id, ready=False)
            if not current.operation or current.operation["operation_id"] != operation_id:
                raise DatabaseError("OPERATION_ID_MISMATCH", "Recovery operation changed", 7)
            if current.status == "creating" and operation["kind"] == "create":
                return self._finish_create(current, client).model_dump()
            if current.status == "deleting" and operation["kind"] == "drop":
                return self._finish_drop(current, client).model_dump()
            raise DatabaseError("RECOVERY_REQUIRED", "Unknown pending operation", 7)
