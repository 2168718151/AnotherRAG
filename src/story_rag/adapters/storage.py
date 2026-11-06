from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from story_rag.adapters.database_registry import FileLease, storage_fingerprint
from story_rag.config import Settings, ensure_directories

logger = logging.getLogger(__name__)
DENSE_VECTOR_NAME = "text-dense"


class StorageUnavailableError(RuntimeError):
    pass


def database_identity(settings: Settings) -> str:
    return storage_fingerprint(settings)


def _atomic_write(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        writer(temporary)
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _persist_docstore(docstore: SimpleDocumentStore, path: Path) -> None:
    _atomic_write(path, lambda temporary: docstore.persist(persist_path=temporary))


def _document_filter(document_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=document_id))]
    )


def _point_ids(client: QdrantClient, collection: str, document_id: str) -> set[str]:
    result: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection,
            scroll_filter=_document_filter(document_id),
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        result.update(str(point.id) for point in points)
        if offset is None:
            return result


def _finish_delete(
    vector_store: QdrantVectorStore,
    docstore: SimpleDocumentStore,
    document_id: str,
    node_ids: list[str],
    index: VectorStoreIndex | None = None,
) -> None:
    if index is not None:
        index.delete_ref_doc(document_id, delete_from_docstore=True)
    else:
        vector_store.delete(document_id)
        docstore.delete_ref_doc(document_id, raise_error=False)
    # Recovery may see an already-removed ref mapping. Journal IDs complete cleanup.
    for node_id in node_ids:
        if index is not None and node_id in index.index_struct.nodes_dict:
            index.index_struct.delete(node_id)
        docstore.delete_document(node_id, raise_error=False)
    docstore.delete_document(document_id, raise_error=False)


def _verify_deleted(
    client: QdrantClient,
    collection: str,
    docstore: SimpleDocumentStore,
    document_id: str,
    node_ids: list[str],
) -> None:
    if _point_ids(client, collection, document_id) or (
        node_ids and client.retrieve(collection, ids=node_ids, with_payload=False)
    ):
        raise StorageUnavailableError("Deleted document still has Qdrant points")
    docs = docstore.docs
    if docstore.get_ref_doc_info(document_id) is not None or any(
        node_id in docs for node_id in [document_id, *node_ids]
    ):
        raise StorageUnavailableError("Deleted document still has docstore records")


def _recover_delete(
    settings: Settings,
    client: QdrantClient,
    vector_store: QdrantVectorStore,
    docstore: SimpleDocumentStore,
) -> None:
    journal = settings.docstore_path.with_suffix(".delete.json")
    if not journal.exists():
        return
    record = json.loads(journal.read_text(encoding="utf-8"))
    document_id = record.get("document_id", "")
    node_ids = record.get("node_ids")
    if (
        record.get("version") != 1
        or record.get("database_id") != database_identity(settings)
        or not isinstance(document_id, str)
        or not re.fullmatch(r"[0-9a-f]{64}", document_id)
        or not isinstance(node_ids, list)
        or not node_ids
        or any(
            not isinstance(item, str)
            or not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", item)
            for item in node_ids
        )
    ):
        raise StorageUnavailableError("Invalid deletion journal; manual recovery required")
    docs = docstore.docs
    ref = docstore.get_ref_doc_info(document_id)
    if (ref and set(ref.node_ids) != set(node_ids)) or any(
        node_id in docs and docs[node_id].ref_doc_id != document_id for node_id in node_ids
    ):
        raise StorageUnavailableError("Deletion journal does not match its document store")
    logger.warning("Recovering pending document deletion: %s", document_id)
    _finish_delete(vector_store, docstore, document_id, node_ids)
    _verify_deleted(client, settings.collection_name, docstore, document_id, node_ids)
    _persist_docstore(docstore, settings.docstore_path)
    journal.unlink()


class EuclidQdrantVectorStore(QdrantVectorStore):
    def parse_to_query_result(self, response: list[Any]) -> VectorStoreQueryResult:
        result = super().parse_to_query_result(response)
        # RRF sorts scores descending; Qdrant returns ascending Euclid distances.
        if result.similarities is not None:
            result.similarities = [1.0 / (1.0 + distance) for distance in result.similarities]
        return result


@dataclass(slots=True)
class StorageBundle:
    settings: Settings
    vector_store: QdrantVectorStore
    docstore: SimpleDocumentStore
    index: VectorStoreIndex
    client: QdrantClient = field(repr=False)
    aclient: AsyncQdrantClient = field(repr=False)
    embedding_dimension: int
    revision: int = 0
    lock: Any = field(default_factory=threading.RLock, repr=False)
    _unavailable: bool = field(default=False, repr=False)
    ownership: FileLease | None = field(default=None, repr=False)

    @property
    def database_id(self) -> str:
        return database_identity(self.settings)

    @property
    def deletion_journal(self) -> Path:
        return self.settings.docstore_path.with_suffix(".delete.json")

    @property
    def mutation_journal(self) -> Path:
        return self.settings.docstore_path.with_suffix(".mutation.json")

    def ensure_available(self) -> None:
        if self._unavailable or self.deletion_journal.exists() or self.mutation_journal.exists():
            raise StorageUnavailableError("Storage requires restart/recovery before use")

    def mark_unavailable(self) -> None:
        self._unavailable = True

    def persist(self) -> None:
        with self.lock:
            _persist_docstore(self.docstore, self.settings.docstore_path)

    def bump_revision(self) -> None:
        """Mark the corpus as changed so retrievers rebuild on the next query."""
        self.revision += 1

    def all_nodes(self) -> list[BaseNode]:
        return list(self.docstore.docs.values())

    def text_nodes(self) -> list[TextNode]:
        return [item for item in self.docstore.docs.values() if isinstance(item, TextNode)]

    def node_count(self) -> int:
        return sum(isinstance(item, TextNode) for item in self.docstore.docs.values())

    def document_count(self) -> int:
        return sum(isinstance(item, Document) for item in self.docstore.docs.values())

    def delete_document(self, document_id: str) -> None:
        with self.lock:
            self.ensure_available()
            ref = self.docstore.get_ref_doc_info(document_id)
            if ref is None:
                raise StorageUnavailableError("Document has no reference-node mapping")
            node_ids = list(ref.node_ids)
            if set(node_ids) != self.document_point_ids(document_id):
                raise StorageUnavailableError("Document nodes and Qdrant points differ")
            record = {
                "version": 1,
                "database_id": self.database_id,
                "document_id": document_id,
                "node_ids": node_ids,
            }
            _atomic_write(
                self.deletion_journal,
                lambda temporary: Path(temporary).write_text(json.dumps(record), encoding="utf-8"),
            )
            try:
                _finish_delete(self.vector_store, self.docstore, document_id, node_ids, self.index)
                _verify_deleted(
                    self.client,
                    self.settings.collection_name,
                    self.docstore,
                    document_id,
                    node_ids,
                )
                self.persist()
                self.deletion_journal.unlink()
                self.bump_revision()
            except Exception as exc:
                self.mark_unavailable()
                logger.exception("Document deletion incomplete: %s", document_id)
                raise StorageUnavailableError("Deletion incomplete; restart to recover") from exc

    def document_point_ids(self, document_id: str) -> set[str]:
        return _point_ids(self.client, self.settings.collection_name, document_id)

    def is_ready(self) -> bool:
        with self.lock:
            return self._is_ready()

    def _is_ready(self) -> bool:
        try:
            self.ensure_available()
            info = self.client.get_collection(self.settings.collection_name)
            _validate_collection(info, self.embedding_dimension)
            if info.status not in {models.CollectionStatus.GREEN, models.CollectionStatus.YELLOW}:
                return False
            count = self.client.count(self.settings.collection_name, exact=True).count
            return count == self.node_count()
        except Exception:
            logger.warning("Qdrant collection readiness check failed")
            return False

    async def aclose(self) -> None:
        try:
            await self.aclient.close()
        finally:
            try:
                self.client.close()
            finally:
                if self.ownership:
                    self.ownership.close()


def _validate_collection(info: models.CollectionInfo, embedding_dimension: int) -> None:
    vectors = info.config.params.vectors
    dense = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
    if (
        dense is None
        or dense.size != embedding_dimension
        or dense.distance != models.Distance.EUCLID
    ):
        raise ValueError(
            "Qdrant collection must use text-dense vectors with the current embedding "
            "dimension and Euclid distance; use a new collection and reimport TXT"
        )


def _validate_node_ids(client: QdrantClient, collection_name: str, node_ids: set[str]) -> None:
    stored_ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name, limit=256, offset=offset, with_payload=False, with_vectors=False
        )
        stored_ids.update(str(point.id) for point in points)
        if offset is None:
            break
    if stored_ids != node_ids:
        raise ValueError(
            "Qdrant point IDs and document store node IDs differ; restore a matched pair"
        )


def build_storage(
    settings: Settings,
    embed_model: Any,
    *,
    lease: FileLease | None = None,
    embedding_dimension: int | None = None,
    recover_mutations: bool = False,
) -> StorageBundle:
    ensure_directories(settings)
    pending = settings.docstore_path.with_suffix(".mutation.json").exists()
    if pending and not recover_mutations:
        raise StorageUnavailableError("Document update requires explicit db recover")
    if (
        settings.registered_db_id
        and settings.docstore_path.with_suffix(".delete.json").exists()
        and not recover_mutations
    ):
        raise StorageUnavailableError("Document deletion requires explicit db recover")
    if not settings.allow_storage_create and not settings.docstore_path.is_file():
        raise StorageUnavailableError("Registered document store is missing; refusing creation")

    if settings.docstore_path.is_file():
        logger.info("Loading document store from %s", settings.docstore_path)
        docstore = SimpleDocumentStore.from_persist_path(str(settings.docstore_path))
    else:
        logger.info("Creating a new document store")
        docstore = SimpleDocumentStore()

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
    )
    try:
        exists = client.collection_exists(settings.collection_name)
        if not exists and not settings.allow_storage_create:
            raise StorageUnavailableError("Registered collection is missing; refusing creation")
        if not exists and docstore.docs:
            raise ValueError(
                "Document store exists without its Qdrant collection; use an empty "
                "docstore and reimport TXT instead of silently skipping duplicate documents"
            )
        embedding_dimension = embedding_dimension or len(
            embed_model.get_text_embedding("embedding dimension probe")
        )
        if settings.expected_embedding_dimension not in {None, embedding_dimension}:
            raise ValueError("Embedding dimension differs from the registered database")
        if not exists:
            client.create_collection(
                collection_name=settings.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=embedding_dimension, distance=models.Distance.EUCLID
                    )
                },
            )
            client.create_payload_index(
                collection_name=settings.collection_name,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        _validate_collection(client.get_collection(settings.collection_name), embedding_dimension)
        # Diagnostic retrieval can run across different event loops. Do not pool
        # async sockets across those loops; both clients use the same server.
        aclient = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout_seconds,
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        vector_store = EuclidQdrantVectorStore(
            collection_name=settings.collection_name,
            client=client,
            aclient=aclient,
            enable_hybrid=False,
            dense_vector_name=DENSE_VECTOR_NAME,
        )
        # Complete an interrupted deletion before checking paired-store consistency.
        _recover_delete(settings, client, vector_store, docstore)
        node_count = sum(isinstance(item, TextNode) for item in docstore.docs.values())
        if not pending and client.count(settings.collection_name, exact=True).count != node_count:
            raise ValueError(
                "Qdrant points and document store nodes differ; restore a matched pair"
            )
        if not pending:
            _validate_node_ids(
                client,
                settings.collection_name,
                {item.node_id for item in docstore.docs.values() if isinstance(item, TextNode)},
            )
    except Exception:
        client.close()
        # No async request has been made at this point.
        raise

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )
    # Keep chunks in the JSON docstore for BM25 and original-document recovery;
    # vector retrieval restores its nodes from the paired Qdrant payloads.
    index = VectorStoreIndex(
        nodes=[],
        storage_context=storage_context,
        embed_model=embed_model,
        store_nodes_override=True,
    )
    # A restarted index has no in-memory mappings; restore them without embedding again.
    for node in docstore.docs.values():
        if isinstance(node, TextNode):
            index.index_struct.add_node(node, text_id=node.node_id)
    storage_context.index_store.add_index_struct(index.index_struct)
    logger.info(
        "Storage ready: collection=%s, documents=%s, nodes=%s",
        settings.collection_name,
        sum(isinstance(item, Document) for item in docstore.docs.values()),
        sum(isinstance(item, TextNode) for item in docstore.docs.values()),
    )
    storage = StorageBundle(
        settings=settings,
        vector_store=vector_store,
        docstore=docstore,
        index=index,
        client=client,
        aclient=aclient,
        embedding_dimension=embedding_dimension,
        ownership=lease,
    )
    if pending:
        from story_rag.adapters.document_updates import recover_update

        try:
            recover_update(storage)
        except BaseException:
            client.close()
            if lease:
                lease.close()
            raise
    return storage
