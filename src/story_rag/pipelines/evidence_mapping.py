from __future__ import annotations

import hashlib
import logging
import time
from typing import Literal, Protocol
from uuid import UUID

from llama_index.core.schema import Document, TextNode
from pydantic import BaseModel, ConfigDict, Field, model_validator

from story_rag.adapters.qa_parser import extract_qa_pairs
from story_rag.adapters.storage import StorageUnavailableError
from story_rag.conversation_domain import (
    DatabaseBinding,
    EvidenceRef,
    ObjectRef,
    ObjectVersionDraft,
    QuestionVersionDraft,
    RequestPayload,
    RequestRecord,
    SessionRecord,
    request_digest,
)
from story_rag.domain import NodeTrace, QueryRequest, RetrievalResult
from story_rag.pipelines.history_resolution import HistoryScope

logger = logging.getLogger(__name__)


class EvidenceMappingError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserProvidedFact(EvidenceModel):
    field: str = Field(min_length=1, max_length=80)
    source_excerpt: str = Field(min_length=1, max_length=512)


class EvidenceTaskInput(EvidenceModel):
    task_key: str = Field(min_length=1, max_length=32)
    question: QuestionVersionDraft
    objects: list[ObjectVersionDraft] = Field(default_factory=list)
    allowed_document_ids: list[str] | None = None
    user_facts: list[UserProvidedFact] = Field(default_factory=list)
    prior_evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    prior_object_refs: list[ObjectRef] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)

    @model_validator(mode="after")
    def unique_documents(self) -> EvidenceTaskInput:
        if self.allowed_document_ids is not None and len(self.allowed_document_ids) != len(
            set(self.allowed_document_ids)
        ):
            raise ValueError("allowed document IDs must be unique")
        return self


class EvidencePoolEntry(EvidenceModel):
    evidence_id: str
    ref: EvidenceRef
    text: str
    source: str | None = None
    complete_qa_pairs: list[str] | None = None


class EvidenceRank(EvidenceModel):
    evidence_id: str
    rank: int = Field(ge=1)
    score: float | None = None
    rerank_score: float | None = None


class TaskEvidence(EvidenceModel):
    task_key: str
    question_id: UUID
    question_version: int
    search_query: str | None = None
    decision: Literal[
        "candidates", "no_candidates", "waiting_input", "needs_refresh", "cancelled",
        "technical_failure",
    ]
    sufficiency: Literal["unverified", "not_assessed"] = "not_assessed"
    allowed_evidence_ids: list[str] = Field(default_factory=list)
    ranks: list[EvidenceRank] = Field(default_factory=list)
    user_facts: list[UserProvidedFact] = Field(default_factory=list)
    current_fact_turn_id: UUID
    prior_evidence_status: Literal["none", "valid", "needs_refresh"] = "none"
    stale_candidate_ids: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    error_code: str | None = None
    latency_ms: float = Field(ge=0)
    retrieval_latency_ms: float = Field(default=0, ge=0)
    rerank_latency_ms: float = Field(default=0, ge=0)
    retrieval_reused: bool = False


class EvidenceBatch(EvidenceModel):
    session_id: UUID
    binding: DatabaseBinding
    as_of_revision: int
    tasks: list[TaskEvidence]
    evidence_pool: dict[str, EvidencePoolEntry]


class ScopedSessionReader(Protocol):
    def get_session(
        self, session_id: UUID, binding: DatabaseBinding, *, include_archived: bool = False
    ) -> SessionRecord: ...


class KnowledgeStorage(Protocol):
    settings: object
    database_id: str
    docstore: object
    lock: object

    def ensure_available(self) -> None: ...

    def document_point_ids(self, document_id: str) -> set[str]: ...


class KnowledgeRetriever(Protocol):
    def retrieve(self, request: QueryRequest) -> RetrievalResult: ...

    def retrieve_trace(self, request: QueryRequest) -> RetrievalResult: ...


class EvidenceMappingPipeline:
    """Scope and verify per-task retrieval without treating rank as sufficiency."""

    def __init__(
        self,
        session_reader: ScopedSessionReader,
        storage: KnowledgeStorage,
        retriever: KnowledgeRetriever,
    ) -> None:
        self._session_reader = session_reader
        self._storage = storage
        self._retriever = retriever

    def map_evidence(
        self,
        scope: HistoryScope,
        request: RequestRecord,
        payload: RequestPayload,
        tasks: list[EvidenceTaskInput],
    ) -> EvidenceBatch:
        return self._map(scope, request, payload, tasks, trace=False)

    def map_trace(
        self,
        scope: HistoryScope,
        request: RequestRecord,
        payload: RequestPayload,
        tasks: list[EvidenceTaskInput],
    ) -> EvidenceBatch:
        return self._map(scope, request, payload, tasks, trace=True)

    def validate_batch(
        self, scope: HistoryScope, request: RequestRecord, payload: RequestPayload,
        inputs: list[EvidenceTaskInput], batch: EvidenceBatch,
    ) -> set[str]:
        self._verify_scope(scope, request, payload, inputs)
        if (
            batch.session_id != scope.session_id or batch.binding != scope.binding
            or batch.as_of_revision != scope.as_of_revision
            or {item.task_key for item in batch.tasks} != {item.task_key for item in inputs}
        ):
            raise EvidenceMappingError("BATCH_MISMATCH", "Evidence batch scope changed")
        stale: set[str] = set()
        point_cache: dict[str, set[str]] = {}
        with self._storage.lock:
            self._storage.ensure_available()
            for item in batch.tasks:
                for evidence_id in item.allowed_evidence_ids:
                    entry = batch.evidence_pool.get(evidence_id)
                    if entry is None:
                        stale.add(item.task_key)
                        break
                    candidate = NodeTrace(
                        node_id=entry.ref.node_id, text="", retriever="final_check",
                        rank=1, metadata={"document_id": entry.ref.document_id},
                    )
                    current = self._verify_candidate(
                        candidate, point_cache, check_text=False
                    )
                    if current is None or current.ref != entry.ref:
                        stale.add(item.task_key)
                        break
        return stale

    def _map(
        self,
        scope: HistoryScope,
        request: RequestRecord,
        payload: RequestPayload,
        tasks: list[EvidenceTaskInput],
        *,
        trace: bool,
    ) -> EvidenceBatch:
        self._verify_scope(scope, request, payload, tasks)
        pool: dict[str, EvidencePoolEntry] = {}
        results: list[TaskEvidence] = []
        retrieval_cache: dict[tuple[str, int | None], RetrievalResult] = {}
        point_cache: dict[str, set[str]] = {}
        with self._storage.lock:
            for task in tasks:
                started = time.perf_counter()
                result = TaskEvidence(
                    task_key=task.task_key,
                    question_id=task.question.question_id,
                    question_version=task.question.version,
                    decision="no_candidates",
                    user_facts=task.user_facts,
                    current_fact_turn_id=request.turn_id,
                    missing_inputs=task.question.missing_inputs,
                    latency_ms=0,
                )
                task_entries: dict[str, EvidencePoolEntry] = {}
                try:
                    self._storage.ensure_available()
                    result.prior_evidence_status = self._prior_status(task, point_cache)
                    if task.question.status == "cancelled":
                        result.decision = "cancelled"
                    elif task.question.status == "waiting_input":
                        result.decision = "waiting_input"
                    else:
                        query = self._search_query(task)
                        result.search_query = query
                        cache_key = (query, task.top_k)
                        if cache_key not in retrieval_cache:
                            query_request = QueryRequest(query=query, top_k=task.top_k)
                            retrieval_cache[cache_key] = (
                                self._retriever.retrieve_trace(query_request)
                                if trace else self._retriever.retrieve(query_request)
                            )
                            result.retrieval_latency_ms = retrieval_cache[cache_key].latency_ms
                            result.rerank_latency_ms = retrieval_cache[cache_key].rerank_latency_ms
                        else:
                            result.retrieval_reused = True
                        retrieved = retrieval_cache[cache_key]
                        allowed_docs = (
                            set(task.allowed_document_ids)
                            if task.allowed_document_ids is not None else None
                        )
                        for candidate in retrieved.nodes:
                            if (
                                allowed_docs is not None
                                and candidate.metadata.get("document_id") not in allowed_docs
                            ):
                                continue
                            entry = self._verify_candidate(candidate, point_cache)
                            if entry is None:
                                result.stale_candidate_ids.append(candidate.node_id)
                                continue
                            if (
                                allowed_docs is not None
                                and entry.ref.document_id not in allowed_docs
                            ):
                                continue
                            if entry.evidence_id in result.allowed_evidence_ids:
                                continue
                            task_entries[entry.evidence_id] = entry
                            result.allowed_evidence_ids.append(entry.evidence_id)
                            result.ranks.append(
                                EvidenceRank(
                                    evidence_id=entry.evidence_id,
                                    rank=candidate.rank,
                                    score=candidate.score,
                                    rerank_score=candidate.rerank_score,
                                )
                            )
                        if result.allowed_evidence_ids:
                            result.decision = "candidates"
                            result.sufficiency = "unverified"
                        elif result.stale_candidate_ids:
                            result.decision = "needs_refresh"
                    pool.update(task_entries)
                except Exception as exc:
                    result.decision = "technical_failure"
                    if isinstance(exc, EvidenceMappingError):
                        result.error_code = exc.code
                    elif isinstance(exc, StorageUnavailableError):
                        result.error_code = "STORAGE_UNAVAILABLE"
                    else:
                        result.error_code = "RETRIEVAL_FAILED"
                    logger.error(
                        "Evidence mapping failed task=%s code=%s",
                        task.task_key,
                        result.error_code,
                    )
                    result.allowed_evidence_ids.clear()
                    result.ranks.clear()
                finally:
                    result.latency_ms = (time.perf_counter() - started) * 1000
                results.append(result)
        return EvidenceBatch(
            session_id=scope.session_id,
            binding=scope.binding,
            as_of_revision=scope.as_of_revision,
            tasks=results,
            evidence_pool=pool,
        )

    def _verify_scope(
        self,
        scope: HistoryScope,
        request: RequestRecord,
        payload: RequestPayload,
        tasks: list[EvidenceTaskInput],
    ) -> None:
        registered_id = getattr(self._storage.settings, "registered_db_id", None)
        if (
            registered_id != str(scope.binding.database_id)
            or self._storage.database_id != scope.binding.storage_fingerprint
        ):
            raise EvidenceMappingError("DATABASE_MISMATCH", "Knowledge store binding differs")
        session = self._session_reader.get_session(scope.session_id, scope.binding)
        if session.revision != scope.as_of_revision:
            raise EvidenceMappingError("STALE_REVISION", "Session revision changed")
        if (
            request.session_id != scope.session_id
            or request.expected_revision != scope.as_of_revision
            or request.request_digest != request_digest(payload, scope.as_of_revision)
            or request.status not in {"queued", "running"}
            or payload.use_history != scope.use_history
        ):
            raise EvidenceMappingError("REQUEST_MISMATCH", "Request and scope differ")
        if len({task.task_key for task in tasks}) != len(tasks):
            raise EvidenceMappingError("DUPLICATE_TASK", "Task keys must be unique")
        if len({task.question.question_id for task in tasks}) != len(tasks):
            raise EvidenceMappingError("DUPLICATE_QUESTION", "Question IDs must be unique")
        for task in tasks:
            if any(fact.source_excerpt not in payload.message for fact in task.user_facts):
                raise EvidenceMappingError("INVALID_FACT_SOURCE", "Fact is not in current message")
            if not payload.use_history and task.prior_evidence_refs:
                raise EvidenceMappingError("HISTORY_DISABLED", "Prior evidence cannot be read")
            expected = {(ref.object_id, ref.version) for ref in task.question.object_refs}
            actual = {(obj.object_id, obj.version) for obj in task.objects}
            if expected != actual or len(actual) != len(task.objects):
                raise EvidenceMappingError("OBJECT_MISMATCH", "Object versions differ")

    @staticmethod
    def _search_query(task: EvidenceTaskInput) -> str:
        parts = [task.question.question_text.strip()]
        for obj in sorted(task.objects, key=lambda item: str(item.object_id)):
            parts.append(obj.kind + ": " + ", ".join(
                f"{key}={value}" for key, value in sorted(obj.conditions.items())
            ))
        # Presentation requirements and prior goals do not belong in search text.
        parts.extend(task.question.effective_state.get("retrieval_constraints", []))
        query = "\n".join(dict.fromkeys(part.strip() for part in parts if part.strip()))
        if len(query) > 4000:
            raise EvidenceMappingError("QUERY_TOO_LONG", "Search query exceeds phase-four limit")
        return query

    def _prior_status(
        self, task: EvidenceTaskInput, point_cache: dict[str, set[str]]
    ) -> Literal["none", "valid", "needs_refresh"]:
        if not task.prior_evidence_refs:
            return "none"
        if task.prior_object_refs is None or set(
            (ref.object_id, ref.version) for ref in task.prior_object_refs
        ) != set((ref.object_id, ref.version) for ref in task.question.object_refs):
            return "needs_refresh"
        for ref in task.prior_evidence_refs:
            candidate = NodeTrace(
                node_id=ref.node_id,
                text="",
                retriever="prior",
                rank=1,
                metadata={"document_id": ref.document_id},
            )
            entry = self._verify_candidate(candidate, point_cache, check_text=False)
            if entry is None or entry.ref != ref:
                return "needs_refresh"
        return "valid"

    def _verify_candidate(
        self,
        candidate: NodeTrace,
        point_cache: dict[str, set[str]],
        *,
        check_text: bool = True,
    ) -> EvidencePoolEntry | None:
        docs = self._storage.docstore.docs
        node = docs.get(candidate.node_id)
        if not isinstance(node, TextNode):
            return None
        document_id = node.ref_doc_id
        document = docs.get(document_id)
        if not isinstance(document, Document):
            return None
        version = document.metadata.get("content_hash")
        if not isinstance(version, str) or not version:
            return None
        if (
            node.metadata.get("content_hash") != version
            or node.metadata.get("document_id") != document_id
            or candidate.metadata.get("document_id") != document_id
        ):
            return None
        if check_text and (
            candidate.text != node.get_content()
            or candidate.metadata.get("content_hash") != version
        ):
            return None
        if document_id not in point_cache:
            point_cache[document_id] = self._storage.document_point_ids(document_id)
        if candidate.node_id not in point_cache[document_id]:
            return None
        node_hash = hashlib.sha256(node.text.encode("utf-8")).hexdigest()
        ref = EvidenceRef(
            document_id=document_id,
            node_id=candidate.node_id,
            document_version=version,
            node_hash=node_hash,
        )
        db_id = self._storage.settings.registered_db_id
        evidence_id = hashlib.sha256(
            f"{db_id}:{version}:{candidate.node_id}:{node_hash}".encode()
        ).hexdigest()
        node_questions = {pair.question for pair in extract_qa_pairs(node.text)}
        if isinstance(node.metadata.get("question"), str):
            node_questions.add(node.metadata["question"])
        # Read the complete pair from this verified document, never return a split answer.
        complete_pairs = [
            pair.text for pair in extract_qa_pairs(document.text)
            if pair.question in node_questions
        ]
        return EvidencePoolEntry(
            evidence_id=evidence_id,
            ref=ref,
            text=node.text,
            source=document.metadata.get("source"),
            complete_qa_pairs=complete_pairs,
        )
