from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import replace
from pathlib import Path

from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore.types import RefDocInfo

from story_rag.adapters.document_updates import prepare_embeddings, validate_filename, write_update
from story_rag.adapters.storage import StorageBundle, StorageUnavailableError
from story_rag.adapters.text_loader import build_decoded_text
from story_rag.domain import (
    DocumentContentPreview,
    DocumentDeletePreview,
    DocumentDeletionResult,
    DocumentDetail,
    DocumentPage,
    DocumentRecord,
)

logger = logging.getLogger(__name__)


class DocumentManagementError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class DocumentManagementPipeline:
    def __init__(self, storage: StorageBundle) -> None:
        self._storage = storage

    def list_documents(self, q: str = "", offset: int = 0, limit: int = 50) -> DocumentPage:
        if offset < 0 or not 1 <= limit <= 200:
            raise DocumentManagementError(422, "Invalid pagination")
        with self._storage.lock:
            self._storage.ensure_available()
            if not self._storage.is_ready():
                raise StorageUnavailableError("Qdrant and document store are not ready")
            docs = self._storage.docstore.docs
            refs = self._storage.docstore.get_all_ref_doc_info() or {}
            records = [
                self._record(document, refs.get(document.doc_id), docs)
                for document in docs.values()
                if isinstance(document, Document)
            ]
            query = q.strip().casefold()
            records = [
                item
                for item in records
                if not query or query in (f"{item.filename}\n{item.source}".casefold())
            ]
            records.sort(key=lambda item: (item.filename.casefold(), item.source, item.document_id))
            return DocumentPage(
                database_id=self._storage.database_id,
                collection_name=self._storage.settings.collection_name,
                total=len(records),
                offset=offset,
                limit=limit,
                items=records[offset : offset + limit],
            )

    def get_document(self, document_id: str) -> DocumentDetail:
        with self._storage.lock:
            self._storage.ensure_available()
            docs = self._storage.docstore.docs
            document = docs.get(document_id)
            if not isinstance(document, Document):
                raise DocumentManagementError(404, "Document not found")
            record = self._record(
                document, self._storage.docstore.get_ref_doc_info(document_id), docs
            )
            return DocumentDetail(
                **record.model_dump(),
                database_id=self._storage.database_id,
                collection_name=self._storage.settings.collection_name,
            )

    def preview_document(
        self, document_id: str, max_characters: int = 8000
    ) -> DocumentContentPreview:
        if not 1 <= max_characters <= 20000:
            raise DocumentManagementError(422, "Invalid preview length")
        with self._storage.lock:
            detail = self.get_document(document_id)
            text = self._storage.docstore.docs[document_id].text
            return DocumentContentPreview(
                **detail.model_dump(),
                text=text[:max_characters],
                total_characters=len(text),
                truncated=len(text) > max_characters,
            )

    def delete_document(
        self,
        document_id: str,
        *,
        database_id: str,
        if_match: str,
    ) -> DocumentDeletionResult:
        with self._storage.lock:
            self._storage.ensure_available()
            if database_id != self._storage.database_id:
                raise DocumentManagementError(409, "Database identity changed; refresh the list")
            record = self.get_document(document_id)
            if if_match != record.etag:
                raise DocumentManagementError(412, "Document changed; refresh before deleting")
            if record.status != "indexed":
                raise StorageUnavailableError("Document reference nodes are inconsistent")
            self._storage.delete_document(document_id)
            logger.info(
                "Document deleted: database=%s document=%s filename=%r nodes=%s",
                database_id,
                document_id,
                record.filename,
                record.node_count,
            )
            return DocumentDeletionResult(
                database_id=database_id,
                collection_name=record.collection_name,
                document_id=document_id,
                filename=record.filename,
                deleted_nodes=record.node_count,
            )

    def delete_preview(self, document_id: str) -> DocumentDeletePreview:
        with self._storage.lock:
            detail = self.get_document(document_id)
            ref = self._storage.docstore.get_ref_doc_info(document_id)
            node_ids = sorted(ref.node_ids) if ref else []
            point_ids = sorted(self._storage.document_point_ids(document_id))
            return DocumentDeletePreview(
                **detail.model_dump(),
                node_ids=node_ids,
                qdrant_point_ids=point_ids,
                consistent=detail.status == "indexed" and node_ids == point_ids,
            )

    def _check_update(self, document_id: str, database_id: str, if_match: str) -> DocumentDetail:
        self._storage.ensure_available()
        if database_id != self._storage.database_id:
            raise DocumentManagementError(409, "Database identity changed")
        detail = self.get_document(document_id)
        if detail.etag != if_match:
            raise DocumentManagementError(412, "Document changed; refresh before modifying")
        if not self.delete_preview(document_id).consistent:
            raise StorageUnavailableError("Document reference nodes are inconsistent")
        return detail

    def rename_preview(self, document_id: str, name: str) -> dict:
        name = validate_filename(name)
        with self._storage.lock:
            detail = self.delete_preview(document_id)
            return {
                **detail.model_dump(),
                "new_name": name,
                "same_name_count": sum(
                    r.filename.casefold() == name.casefold() and r.document_id != document_id
                    for r in (
                        self._record(
                            doc,
                            self._storage.docstore.get_ref_doc_info(doc.doc_id),
                            self._storage.docstore.docs,
                        )
                        for doc in self._storage.docstore.docs.values()
                        if isinstance(doc, Document)
                    )
                ),
                "dry_run": True,
            }

    def rename_document(
        self, document_id: str, name: str, *, database_id: str, if_match: str
    ) -> dict:
        name = validate_filename(name)
        with self._storage.lock:
            detail = self._check_update(document_id, database_id, if_match)
            if detail.filename == name:
                return {**detail.model_dump(), "status": "unchanged"}
            document = self._storage.docstore.docs[document_id].model_copy(deep=True)
            nodes = [
                n.model_copy(deep=True)
                for n in self._storage.text_nodes()
                if n.ref_doc_id == document_id
            ]
            for node in [document, *nodes]:
                node.metadata["display_name"] = name
                if "display_name" not in node.excluded_embed_metadata_keys:
                    node.excluded_embed_metadata_keys.append("display_name")
            result = write_update(self._storage, "rename", document, nodes)
            return {**result, "document": self.get_document(document_id).model_dump()}

    def _replacement(self, document_id: str, path: Path):
        from story_rag.pipelines.ingestion import StoryIngestionPipeline

        if not path.is_file() or path.suffix.casefold() != ".txt":
            raise ValueError("Replacement input must be one TXT file")
        original = self._storage.docstore.docs.get(document_id)
        if not isinstance(original, Document):
            raise DocumentManagementError(404, "Document not found")
        decoded = build_decoded_text(
            path.read_bytes(),
            source_name=original.metadata["source"],
            title=original.metadata.get("title"),
        )
        decoded = replace(decoded, document_id=document_id)
        ingestion = StoryIngestionPipeline(self._storage.settings, self._storage)
        kwargs = {
            "metadata": original.metadata,
            "user_instruction": original.metadata.get("user_instruction"),
        }
        document = ingestion._build_document(decoded, **kwargs)
        nodes = ingestion._build_nodes(decoded, **kwargs)
        if not nodes:
            raise ValueError("Replacement content produced no nodes")
        return original, document, nodes

    def replace_preview(self, document_id: str, path: Path) -> dict:
        with self._storage.lock:
            detail = self.delete_preview(document_id)
            original, document, nodes = self._replacement(document_id, path)
            return {
                **detail.model_dump(),
                "old_hash": original.metadata.get("content_hash"),
                "new_hash": document.metadata["content_hash"],
                "old_chars": len(original.text),
                "new_chars": len(document.text),
                "new_node_count": len(nodes),
                "input_source": str(path.resolve()),
                "dry_run": True,
            }

    def replace_document(
        self, document_id: str, path: Path, *, database_id: str, if_match: str
    ) -> dict:
        with self._storage.lock:
            detail = self._check_update(document_id, database_id, if_match)
            original, document, nodes = self._replacement(document_id, path)
            if original.metadata.get("content_hash") == document.metadata["content_hash"]:
                return {**detail.model_dump(), "status": "unchanged"}
            prepare_embeddings(self._storage, nodes)
            result = write_update(self._storage, "replace", document, nodes)
            return {
                **result,
                "input_source": str(path.resolve()),
                "document": self.get_document(document_id).model_dump(),
            }

    def _record(self, document: Document, ref: RefDocInfo | None, docs: dict) -> DocumentRecord:
        source = str(document.metadata.get("source", ""))
        node_ids = sorted(ref.node_ids) if ref else []
        actual_ids = sorted(
            item.node_id
            for item in docs.values()
            if isinstance(item, TextNode) and item.ref_doc_id == document.doc_id
        )
        version = {
            "database_id": self._storage.database_id,
            "document": document.to_dict(),
            "node_ids": node_ids,
        }
        etag = hashlib.sha256(
            json.dumps(version, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        return DocumentRecord(
            document_id=document.doc_id,
            filename=str(
                document.metadata.get("display_name")
                or source.replace("\\", "/").rsplit("/", 1)[-1]
            ),
            source=source,
            node_count=len(node_ids),
            status="indexed" if node_ids and node_ids == actual_ids else "inconsistent",
            etag=f'"{etag}"',
        )
