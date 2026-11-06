from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StoryMetadata(BaseModel):
    source_name: str
    title: str
    user_instruction: str | None = None
    content_hash: str | None = None
    language_hint: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    document_id: str
    source_name: str
    title: str
    content_hash: str
    node_count: int
    duplicate: bool = False
    replaced: bool = False
    warnings: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


class DatabaseError(RuntimeError):
    def __init__(self, code: str, message: str, exit_code: int = 3) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


class NodeTrace(BaseModel):
    node_id: str
    text: str
    retriever: str
    rank: int
    score: float | None = None
    rerank_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    rewritten_queries: list[str] = Field(default_factory=list)
    nodes: list[NodeTrace] = Field(default_factory=list)
    vector_candidates: list[NodeTrace] = Field(default_factory=list)
    bm25_candidates: list[NodeTrace] = Field(default_factory=list)
    fused_candidates: list[NodeTrace] = Field(default_factory=list)
    latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0


class AnswerResult(BaseModel):
    request_id: str
    status: str
    answer: str
    relevance_yes: bool | None = None
    citations: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class ReadinessResult(BaseModel):
    status: str
    llm_model: str
    embedding_model: str
    reranker_model: str
    collection_name: str
    debug_enabled: bool
    vector_backend: str
    database_ready: bool


class DebugConfigSnapshot(BaseModel):
    project_root: str
    api_file: str
    deepseek_base_url: str
    llm_model: str
    lmstudio_base_url: str
    embedding_model: str
    reranker_model_path: str
    vector_backend: str
    qdrant_url: str
    collection_name: str
    docstore_path: str
    log_dir: str
    debug_enabled: bool


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    source: str
    node_count: int
    status: str
    etag: str


class DocumentPage(BaseModel):
    database_id: str
    collection_name: str
    total: int
    offset: int
    limit: int
    items: list[DocumentRecord]


class DocumentDetail(DocumentRecord):
    database_id: str
    collection_name: str


class DocumentContentPreview(DocumentDetail):
    text: str
    total_characters: int
    truncated: bool


class DocumentDeletionResult(BaseModel):
    database_id: str
    collection_name: str
    document_id: str
    filename: str
    deleted_nodes: int
    status: str = "deleted"
    source_file_deleted: bool = False


class DocumentDeletePreview(DocumentDetail):
    node_ids: list[str]
    qdrant_point_ids: list[str]
    consistent: bool
