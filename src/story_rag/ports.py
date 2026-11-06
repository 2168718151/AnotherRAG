from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from story_rag.conversation_domain import (
    ConversationDisplayPage,
    DatabaseBinding,
    HistoryContentQuery,
    HistoryContentSnapshot,
    ObjectRef,
    ObjectVersionDraft,
    ObjectVersionRecord,
    QuestionVersionDraft,
    QuestionVersionRecord,
    RequestPayload,
    RequestRecord,
    SessionRecord,
    UsageMetadata,
)
from story_rag.domain import (
    AnswerResult,
    DebugConfigSnapshot,
    DocumentContentPreview,
    DocumentDeletePreview,
    DocumentDeletionResult,
    DocumentDetail,
    DocumentPage,
    IngestionResult,
    QueryRequest,
    RetrievalResult,
)
from story_rag.pipelines.evidence_mapping import EvidenceBatch, EvidenceTaskInput
from story_rag.pipelines.history_resolution import (
    HistoryLookup,
    HistoryResolution,
    HistoryScope,
    RevisionBatch,
    RevisionSpec,
)
from story_rag.pipelines.task_planning import TaskPlan, TaskPlanningDebugResult, TaskPlanningResult


class IngestionProductionPort(Protocol):
    def ingest_bytes(
        self,
        data: bytes,
        *,
        source_name: str,
        title: str | None = None,
        user_instruction: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult: ...

    def ingest_path(self, path: Path) -> IngestionResult: ...


class IngestionDebugPort(Protocol):
    def inspect_bytes(self, data: bytes, *, source_name: str) -> dict[str, Any]: ...

    def parse_nodes(
        self, data: bytes, *, source_name: str, title: str | None = None
    ) -> list[Any]: ...


class RetrievalProductionPort(Protocol):
    def retrieve(self, request: QueryRequest) -> RetrievalResult: ...


class RetrievalDebugPort(Protocol):
    def retrieve_trace(self, request: QueryRequest) -> RetrievalResult: ...


class GenerationProductionPort(Protocol):
    def answer(
        self, request: QueryRequest, retrieval: RetrievalResult | None = None
    ) -> AnswerResult: ...


class GenerationDebugPort(Protocol):
    def prepare_llm_input(
        self, request: QueryRequest, retrieval: RetrievalResult | None = None
    ) -> dict[str, Any]: ...

    def answer_trace(
        self, request: QueryRequest, retrieval: RetrievalResult | None = None
    ) -> dict[str, Any]: ...


class WorkflowProductionPort(Protocol):
    async def answer(self, request: QueryRequest) -> AnswerResult: ...


class WorkflowDebugPort(Protocol):
    async def stream_trace(self, request: QueryRequest) -> list[dict[str, Any]]: ...


class ConfigDebugPort(Protocol):
    def snapshot(self) -> DebugConfigSnapshot: ...


class DatabasesProductionPort(Protocol):
    def create(self, name: str) -> Any: ...

    def list_databases(self, include_deleted: bool = False) -> list[dict[str, Any]]: ...

    def info(self, record: Any) -> dict[str, Any]: ...

    def register_default(self, *, dry_run: bool = False) -> dict[str, Any]: ...

    def resolve(
        self, name: str | None = None, db_id: str | None = None, *, ready: bool = True
    ) -> Any: ...

    def open(
        self,
        record: Any,
        *,
        embedding: Any = None,
        probe: bool = False,
        recover: bool = False,
        operation_id: str | None = None,
    ) -> Any: ...

    def drop(self, record: Any, *, confirmation: str, dry_run: bool = False) -> dict[str, Any]: ...

    def rename(
        self, record: Any, name: str, *, confirmation: str, dry_run: bool = False
    ) -> dict[str, Any]: ...

    def recover(
        self, record: Any, *, confirmation: str = "", operation_id: str = "", dry_run: bool = False
    ) -> dict[str, Any]: ...


class DatabasesDebugPort(Protocol):
    def info(self, record: Any) -> dict[str, Any]: ...

    def delete_preview(self, record: Any) -> dict[str, Any]: ...

    def rename_preview(self, record: Any, name: str) -> dict[str, Any]: ...

    def recovery_preview(self, record: Any) -> dict[str, Any]: ...


class QuerySessionProductionPort(Protocol):
    def select(self, name: str | None = None, db_id: str | None = None) -> dict[str, Any]: ...

    def query(
        self,
        text: str,
        *,
        top_k: int | None = None,
        trace: bool = False,
        export: Path | None = None,
    ) -> dict[str, Any]: ...

    def release(self) -> dict[str, str]: ...

    def close(self) -> None: ...


class QuerySessionDebugPort(Protocol):
    def current(self) -> dict[str, Any]: ...


class ConversationLedgerProductionPort(Protocol):
    def create_session(self, name: str, binding: DatabaseBinding) -> SessionRecord: ...

    def get_session(
        self, session_id: UUID, binding: DatabaseBinding, *, include_archived: bool = False
    ) -> SessionRecord: ...

    def list_sessions(
        self, binding: DatabaseBinding, *, include_archived: bool = False
    ) -> list[SessionRecord]: ...

    def archive_session(
        self, session_id: UUID, binding: DatabaseBinding, *, expected_revision: int
    ) -> SessionRecord: ...

    def claim_request(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        request_id: str,
        payload: RequestPayload,
        expected_revision: int,
    ) -> RequestRecord: ...

    def get_request(
        self, session_id: UUID, binding: DatabaseBinding, request_id: str
    ) -> RequestRecord: ...

    def get_display_answer(
        self, session_id: UUID, binding: DatabaseBinding, turn_id: UUID
    ) -> str | None: ...

    def read_history_content(
        self, session_id: UUID, binding: DatabaseBinding, *, as_of_revision: int,
        before_turn_id: UUID, query: HistoryContentQuery,
    ) -> HistoryContentSnapshot | None: ...

    def list_display_turns(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        limit: int = 50,
        cursor: UUID | None = None,
    ) -> ConversationDisplayPage: ...

    def start_request(
        self, session_id: UUID, binding: DatabaseBinding, request_id: str
    ) -> RequestRecord: ...

    def fail_request(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        request_id: str,
        *,
        failure_code: str,
        usage: UsageMetadata | None = None,
        outcome_unknown: bool = False,
    ) -> RequestRecord: ...

    def commit_request(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        request_id: str,
        *,
        questions: list[QuestionVersionDraft],
        objects: list[ObjectVersionDraft] | None = None,
        usage: UsageMetadata | None = None,
        payload: RequestPayload | None = None,
        visible_answer: str | None = None,
    ) -> RequestRecord: ...

    def read_questions_at(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        as_of_revision: int,
        question_no: int | None = None,
        turn_sequence: int | None = None,
        question_id: UUID | None = None,
        pending_only: bool = False,
    ) -> list[QuestionVersionRecord]: ...

    def read_objects_at(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        as_of_revision: int,
        refs: list[ObjectRef],
    ) -> list[ObjectVersionRecord]: ...


class ConversationLedgerDebugPort(Protocol):
    def inspect_scope(self, session_id: UUID, binding: DatabaseBinding) -> dict[str, Any]: ...

    def inspect_display_scope(
        self, session_id: UUID, binding: DatabaseBinding
    ) -> dict[str, int]: ...

    def inspect_versions(self, session_id: UUID, binding: DatabaseBinding) -> dict[str, Any]: ...


class TaskPlanningProductionPort(Protocol):
    def plan(self, message: str) -> TaskPlanningResult: ...


class TaskPlanningDebugPort(Protocol):
    def validate_candidate(
        self, message: str, candidate: dict[str, Any] | TaskPlan
    ) -> TaskPlanningDebugResult: ...


class HistoryResolutionProductionPort(Protocol):
    def resolve(self, scope: HistoryScope, lookup: HistoryLookup) -> HistoryResolution: ...

    def current_questions(self, scope: HistoryScope) -> list[QuestionVersionRecord]: ...

    def next_question_no(self, scope: HistoryScope) -> int: ...

    def objects_for(
        self, scope: HistoryScope, refs: list[ObjectRef]
    ) -> list[ObjectVersionRecord]: ...

    def prepare_batch(
        self,
        scope: HistoryScope,
        request: RequestRecord,
        payload: RequestPayload,
        specs: list[RevisionSpec],
    ) -> RevisionBatch: ...


class HistoryResolutionDebugPort(Protocol):
    def resolve(self, scope: HistoryScope, lookup: HistoryLookup) -> HistoryResolution: ...

    def prepare_batch(
        self,
        scope: HistoryScope,
        request: RequestRecord,
        payload: RequestPayload,
        specs: list[RevisionSpec],
    ) -> RevisionBatch: ...


class EvidenceMappingProductionPort(Protocol):
    def map_evidence(
        self,
        scope: HistoryScope,
        request: RequestRecord,
        payload: RequestPayload,
        tasks: list[EvidenceTaskInput],
    ) -> EvidenceBatch: ...

    def validate_batch(
        self,
        scope: HistoryScope,
        request: RequestRecord,
        payload: RequestPayload,
        inputs: list[EvidenceTaskInput],
        batch: EvidenceBatch,
    ) -> set[str]: ...


class EvidenceMappingDebugPort(Protocol):
    def map_trace(
        self,
        scope: HistoryScope,
        request: RequestRecord,
        payload: RequestPayload,
        tasks: list[EvidenceTaskInput],
    ) -> EvidenceBatch: ...


class DocumentsProductionPort(Protocol):
    def list_documents(self, q: str = "", offset: int = 0, limit: int = 50) -> DocumentPage: ...

    def get_document(self, document_id: str) -> DocumentDetail: ...

    def preview_document(
        self, document_id: str, max_characters: int = 8000
    ) -> DocumentContentPreview: ...

    def delete_document(
        self,
        document_id: str,
        *,
        database_id: str,
        if_match: str,
    ) -> DocumentDeletionResult: ...

    def rename_document(
        self, document_id: str, name: str, *, database_id: str, if_match: str
    ) -> dict[str, Any]: ...

    def replace_document(
        self, document_id: str, path: Path, *, database_id: str, if_match: str
    ) -> dict[str, Any]: ...


class DocumentsDebugPort(Protocol):
    def delete_preview(self, document_id: str) -> DocumentDeletePreview: ...

    def rename_preview(self, document_id: str, name: str) -> dict[str, Any]: ...

    def replace_preview(self, document_id: str, path: Path) -> dict[str, Any]: ...


class EvaluationProductionPort(Protocol):
    def run(self, dataset_path: Path) -> dict[str, Any]: ...


class EvaluationDebugPort(Protocol):
    def generate_dataset(self, source_paths: list[Path], output_path: Path) -> Path: ...

    def preview_case(self, dataset_path: Path, index: int = 0) -> dict[str, Any]: ...
