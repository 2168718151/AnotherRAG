from __future__ import annotations

import re
import unicodedata
import uuid
from typing import TYPE_CHECKING, Any

from llama_index.core import Document
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

from story_rag.adapters.database_registry import atomic_json
from story_rag.domain import DatabaseError

if TYPE_CHECKING:
    from story_rag.adapters.storage import StorageBundle


def validate_filename(name: str) -> str:
    name = unicodedata.normalize("NFC", name).strip()
    stem = name.split(".")[0].upper()
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if (
        not name.casefold().endswith(".txt")
        or len(name) > 255
        or any(c in '<>:"/\\|?*' or unicodedata.category(c).startswith("C") for c in name)
        or name.endswith((".", " "))
        or stem in reserved
        or len(name) <= 4
    ):
        raise DatabaseError("INVALID_ARGUMENT", "Use a valid TXT filename, not a path", 2)
    return name


def prepare_embeddings(storage: StorageBundle, nodes: list[TextNode]) -> None:
    vectors = storage.index._embed_model.get_text_embedding_batch(
        [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
        show_progress=False,
    )
    if len(vectors) != len(nodes) or any(len(v) != storage.embedding_dimension for v in vectors):
        raise DatabaseError("STORAGE_INCONSISTENT", "Prepared embedding dimensions differ", 7)
    for node, vector in zip(nodes, vectors, strict=True):
        node.embedding = vector


def write_update(
    storage: StorageBundle, kind: str, document: Document, nodes: list[TextNode]
) -> dict[str, Any]:
    storage.ensure_available()
    ref = storage.docstore.get_ref_doc_info(document.doc_id)
    if not ref or set(ref.node_ids) != storage.document_point_ids(document.doc_id):
        raise DatabaseError("STORAGE_INCONSISTENT", "Target document is inconsistent", 7)
    record = {
        "version": 1,
        "operation_id": str(uuid.uuid4()),
        "kind": kind,
        "database_id": storage.database_id,
        "document_id": document.doc_id,
        "old_node_ids": list(ref.node_ids),
        "document": document.to_dict(),
        "nodes": [node.to_dict() for node in nodes],
    }
    atomic_json(storage.mutation_journal, record)
    return apply_update(storage, record)


def apply_update(storage: StorageBundle, record: dict[str, Any]) -> dict[str, Any]:
    from story_rag.adapters.storage import (
        StorageUnavailableError,
        _finish_delete,
        _validate_node_ids,
    )

    try:
        document, nodes = validate_update(storage, record)
        target = document.doc_id
        if record["kind"] == "replace":
            _finish_delete(
                storage.vector_store,
                storage.docstore,
                target,
                record["old_node_ids"],
                storage.index,
            )
            storage.docstore.add_documents([document])
            storage.index.insert_nodes(nodes)
        else:
            for node in nodes:
                payload = node_to_metadata_dict(node, remove_text=False, flat_metadata=False)
                storage.client.set_payload(
                    storage.settings.collection_name, payload, points=[node.node_id], wait=True
                )
            storage.docstore.add_documents([document, *nodes], allow_update=True)
        storage.persist()
        expected = {node.node_id for node in nodes}
        ref = storage.docstore.get_ref_doc_info(target)
        if (
            not ref
            or set(ref.node_ids) != expected
            or storage.document_point_ids(target) != expected
        ):
            raise ValueError("Updated document has inconsistent nodes")
        _validate_node_ids(
            storage.client,
            storage.settings.collection_name,
            {n.node_id for n in storage.text_nodes()},
        )
        points = storage.client.retrieve(
            storage.settings.collection_name, ids=list(expected), with_payload=True
        )
        if len(points) != len(nodes):
            raise ValueError("Updated points are missing")
        for point in points:
            restored = metadata_dict_to_node(point.payload or {})
            if restored.metadata.get("display_name") != document.metadata.get("display_name"):
                raise ValueError("Updated display metadata differs")
        storage.mutation_journal.unlink()
        storage._unavailable = False
        storage.bump_revision()
        return {
            "operation_id": record["operation_id"],
            "status": "updated",
            "document_id": target,
            "kind": record["kind"],
        }
    except BaseException as exc:
        storage.mark_unavailable()
        raise StorageUnavailableError(
            "Document update incomplete; explicit db recover required"
        ) from exc


def validate_update(
    storage: StorageBundle, record: dict[str, Any]
) -> tuple[Document, list[TextNode]]:
    try:
        if (
            record["version"] != 1
            or record["database_id"] != storage.database_id
            or record["kind"] not in {"rename", "replace"}
            or not re.fullmatch(r"[0-9a-f]{64}", record["document_id"])
        ):
            raise ValueError("Invalid update identity")
        uuid.UUID(record["operation_id"])
        document = Document.from_dict(record["document"])
        nodes = [TextNode.from_dict(n) for n in record["nodes"]]
        if document.doc_id != record["document_id"] or not nodes:
            raise ValueError("Update document differs")
        if "display_name" in document.metadata:
            validate_filename(document.metadata["display_name"])
        new_ids = {n.node_id for n in nodes}
        if len(new_ids) != len(nodes):
            raise ValueError("Duplicate update nodes")
        for node in nodes:
            uuid.UUID(node.node_id)
            if node.ref_doc_id != document.doc_id:
                raise ValueError("Foreign update node")
            if record["kind"] == "replace" and (
                node.embedding is None or len(node.embedding) != storage.embedding_dimension
            ):
                raise ValueError("Missing prepared vector")
        for node_id in {*record["old_node_ids"], *new_ids}:
            uuid.UUID(node_id)
            existing = storage.docstore.docs.get(node_id)
            if existing and existing.ref_doc_id != document.doc_id:
                raise ValueError("Update would affect another document")
        if record["kind"] == "rename" and new_ids != set(record["old_node_ids"]):
            raise ValueError("Rename changed node identities")
        return document, nodes
    except (ValueError, KeyError, TypeError) as exc:
        raise DatabaseError("STORAGE_INCONSISTENT", "Invalid document update journal", 7) from exc


def recover_update(storage: StorageBundle) -> None:
    import json

    record = json.loads(storage.mutation_journal.read_text(encoding="utf-8"))
    apply_update(storage, record)
