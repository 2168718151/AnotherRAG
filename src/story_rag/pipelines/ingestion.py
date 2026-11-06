from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode

from story_rag.adapters.qa_parser import extract_qa_pairs
from story_rag.adapters.storage import StorageBundle
from story_rag.adapters.text_loader import DecodedText, build_decoded_text, read_text_file
from story_rag.config import Settings
from story_rag.domain import IngestionResult
from story_rag.logging_config import TraceWriter

logger = logging.getLogger(__name__)


class StoryIngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        storage: StorageBundle,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._trace_writer = trace_writer
        self._lock = storage.lock

    def ingest_bytes(
        self,
        data: bytes,
        *,
        source_name: str,
        title: str | None = None,
        user_instruction: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        decoded = build_decoded_text(data, source_name=source_name, title=title)
        with self._lock:
            self._storage.ensure_available()
            return self._ingest_decoded(
                decoded,
                user_instruction=user_instruction,
                metadata=metadata or {},
            )

    def ingest_path(self, path: Path) -> IngestionResult:
        if not path.is_file():
            raise FileNotFoundError(path)
        decoded = read_text_file(path)
        with self._lock:
            self._storage.ensure_available()
            return self._ingest_decoded(decoded, user_instruction=None, metadata={})

    def ingest_directory(self, path: Path) -> list[IngestionResult]:
        if not path.is_dir():
            raise NotADirectoryError(path)
        results: list[IngestionResult] = []
        for file_path in sorted(path.rglob("*.txt")):
            results.append(self.ingest_path(file_path))
        return results

    def inspect_bytes(self, data: bytes, *, source_name: str) -> dict[str, Any]:
        decoded = build_decoded_text(data, source_name=source_name)
        nodes = self._build_nodes(decoded, user_instruction=None, metadata={})
        return {
            "source_name": decoded.source_name,
            "title": decoded.title,
            "encoding": decoded.encoding,
            "document_id": decoded.document_id,
            "content_hash": decoded.content_hash,
            "char_count": len(decoded.text),
            "node_count": len(nodes),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "text": node.text,
                    "metadata": node.metadata,
                }
                for node in nodes
            ],
        }

    def parse_nodes(
        self,
        data: bytes,
        *,
        source_name: str,
        title: str | None = None,
    ) -> list[BaseNode]:
        decoded = build_decoded_text(data, source_name=source_name, title=title)
        return self._build_nodes(decoded, user_instruction=None, metadata={})

    def _ingest_decoded(
        self,
        decoded: DecodedText,
        *,
        user_instruction: str | None,
        metadata: dict[str, Any],
    ) -> IngestionResult:
        existing = self._find_document(decoded.document_id)
        if existing is not None and existing.metadata.get("content_hash") == decoded.content_hash:
            node_count = sum(
                1 for node in self._storage.text_nodes() if node.ref_doc_id == decoded.document_id
            )
            logger.info("Skipping duplicate document %s", decoded.document_id)
            return IngestionResult(
                document_id=decoded.document_id,
                source_name=decoded.source_name,
                title=decoded.title,
                content_hash=decoded.content_hash,
                node_count=node_count,
                duplicate=True,
            )

        replaced = existing is not None
        if existing is not None and existing.metadata.get("display_name"):
            metadata = {**metadata, "display_name": existing.metadata["display_name"]}
        document = self._build_document(
            decoded,
            user_instruction=user_instruction,
            metadata=metadata,
        )
        nodes = self._build_nodes(
            decoded,
            user_instruction=user_instruction,
            metadata=metadata,
        )
        if not nodes:
            raise ValueError("TXT content produced no indexable nodes")

        if replaced:
            logger.info("Replacing changed document %s", decoded.document_id)
            self._storage.delete_document(decoded.document_id)
        try:
            self._storage.docstore.add_documents([document])
            self._storage.index.insert_nodes(nodes)
            self._storage.persist()
            self._storage.bump_revision()
        except Exception:
            self._storage.mark_unavailable()
            raise

        logger.info(
            "Ingested document=%s source=%s nodes=%s replaced=%s",
            decoded.document_id,
            decoded.source_name,
            len(nodes),
            replaced,
        )
        result = IngestionResult(
            document_id=decoded.document_id,
            source_name=decoded.source_name,
            title=decoded.title,
            content_hash=decoded.content_hash,
            node_count=len(nodes),
            replaced=replaced,
        )
        if self._trace_writer:
            self._trace_writer.write(
                decoded.document_id,
                "ingestion",
                {
                    "result": result.model_dump(),
                    "nodes": [
                        {"node_id": node.node_id, "metadata": node.metadata} for node in nodes
                    ],
                },
            )
        return result

    def _build_document(
        self,
        decoded: DecodedText,
        *,
        user_instruction: str | None,
        metadata: dict[str, Any],
    ) -> Document:
        document_metadata = self._flatten_metadata(metadata)
        document_metadata.update(
            {
                "source": decoded.source_name,
                "title": decoded.title,
                "document_id": decoded.document_id,
                "content_hash": decoded.content_hash,
                "encoding": decoded.encoding,
                "user_instruction": user_instruction or "",
            }
        )
        return Document(
            id_=decoded.document_id,
            text=decoded.text,
            metadata=document_metadata,
            excluded_embed_metadata_keys=["display_name"],
        )

    def _build_nodes(
        self,
        decoded: DecodedText,
        *,
        user_instruction: str | None,
        metadata: dict[str, Any],
    ) -> list[TextNode]:
        document = self._build_document(
            decoded,
            user_instruction=user_instruction,
            metadata=metadata,
        )
        pairs = extract_qa_pairs(decoded.text)
        segments = (
            [
                Document(
                    id_=decoded.document_id,
                    text=pair.text,
                    metadata={
                        **document.metadata,
                        "content_type": "qa",
                        "chunk_role": "qa_pair",
                        "question": pair.question,
                    },
                    excluded_embed_metadata_keys=[
                        "source",
                        "document_id",
                        "content_hash",
                        "encoding",
                        "user_instruction",
                        "chunk_role",
                        "display_name",
                    ],
                )
                for pair in pairs
            ]
            if pairs
            else [document]
        )
        typed_nodes: list[TextNode] = []
        for segment in segments:
            nodes = self._splitter.get_nodes_from_documents([segment], show_progress=False)
            question_line = segment.text.splitlines()[0]
            for node in nodes:
                if not isinstance(node, TextNode):
                    continue
                if pairs and len(nodes) > 1 and not node.text.startswith(question_line):
                    node.text = f"{question_line}\n{node.text}"
                index = len(typed_nodes)
                node.metadata["chunk_index"] = index
                legacy_id = hashlib.sha256(
                    f"{decoded.document_id}:{index}:{node.text}".encode()
                ).hexdigest()
                node.id_ = str(uuid.uuid5(uuid.NAMESPACE_URL, legacy_id))
                typed_nodes.append(node)
        return typed_nodes

    def _find_document(self, document_id: str) -> Document | None:
        for item in self._storage.docstore.docs.values():
            if isinstance(item, Document) and item.doc_id == document_id:
                return item
        return None

    @staticmethod
    def _flatten_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        flattened: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                flattened[str(key)] = value
            else:
                flattened[str(key)] = str(value)
        return flattened
