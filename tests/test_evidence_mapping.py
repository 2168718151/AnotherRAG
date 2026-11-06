from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import threading
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.schema import Document, NodeRelationship, RelatedNodeInfo, TextNode

from story_rag.adapters.model_factory import ModelBundle
from story_rag.container import build_container
from story_rag.conversation_domain import (
    DatabaseBinding,
    EvidenceRef,
    RequestPayload,
    RequestRecord,
    SessionRecord,
    request_digest,
)
from story_rag.domain import NodeTrace, QueryRequest, RetrievalResult
from story_rag.pipelines.evidence_mapping import (
    EvidenceMappingError,
    EvidenceMappingPipeline,
    EvidenceTaskInput,
    UserProvidedFact,
)
from story_rag.pipelines.history_resolution import HistoryScope
from tests.conftest import KeywordReranker, ScriptedLLM
from tests.test_conversation_ledger import _case_messages, _frozen_versions

pytestmark = pytest.mark.contract


class FixedSessionReader:
    def __init__(self, session: SessionRecord) -> None:
        self.session = session

    def get_session(self, session_id, binding, *, include_archived=False):
        assert session_id == self.session.session_id
        assert binding.database_id == self.session.database_id
        assert binding.storage_fingerprint == self.session.storage_fingerprint
        return self.session


class FixedKnowledgeStore:
    def __init__(self, binding: DatabaseBinding) -> None:
        self.settings = SimpleNamespace(registered_db_id=str(binding.database_id))
        self.database_id = binding.storage_fingerprint
        self.docstore = SimpleNamespace(docs={})
        self.lock = threading.RLock()
        self.points: dict[str, set[str]] = {}
        self.nodes: list[TextNode] = []

    def add(self, name: str, body: str) -> str:
        document_id = hashlib.sha256(name.encode()).hexdigest()
        version = hashlib.sha256(body.encode()).hexdigest()
        doc = Document(
            id_=document_id,
            text=body,
            metadata={"source": name, "content_hash": version},
        )
        node = TextNode(
            id_=str(uuid4()),
            text=body,
            metadata={"document_id": document_id, "content_hash": version},
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=document_id)},
        )
        self.docstore.docs[document_id] = doc
        self.docstore.docs[node.node_id] = node
        self.points[document_id] = {node.node_id}
        self.nodes.append(node)
        return document_id

    def ensure_available(self) -> None:
        pass

    def document_point_ids(self, document_id: str) -> set[str]:
        return self.points[document_id]


class NoisyRetriever:
    def __init__(self, store: FixedKnowledgeStore) -> None:
        self.store = store
        self.calls: list[str] = []
        self.trace_calls = 0
        self.fail = False

    def retrieve(self, request: QueryRequest) -> RetrievalResult:
        self.calls.append(request.query)
        if self.fail:
            raise RuntimeError("isolated retriever failure")
        nodes = [
            NodeTrace(
                node_id=node.node_id,
                text=node.get_content(),
                retriever="rerank",
                rank=rank,
                score=float(10 - rank),
                rerank_score=float(10 - rank),
                metadata=dict(node.metadata),
            )
            for rank, node in enumerate(self.store.nodes, start=1)
        ]
        return RetrievalResult(query=request.query, nodes=nodes)

    def retrieve_trace(self, request: QueryRequest) -> RetrievalResult:
        self.trace_calls += 1
        return self.retrieve(request)


def _runtime():
    binding = DatabaseBinding(database_id=uuid4(), storage_fingerprint="a" * 64)
    session = SessionRecord(
        session_id=uuid4(),
        name="phase-zero",
        revision=0,
        database_id=binding.database_id,
        storage_fingerprint=binding.storage_fingerprint,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    storage = FixedKnowledgeStore(binding)
    docs = {
        "policy": storage.add("policy.txt", "服务端鉴权独立于提示词和用户确认。"),
        "draft": storage.add("draft.txt", "草稿标题变更仅能作用于已授权资源。"),
        "shared": storage.add("approval.txt", "共享执行器执行前复核主体与租户授权。"),
        "platform": storage.add("platform.txt", "人工平台审批并执行受限变更单。"),
        "ticket": storage.add("ticket.txt", "ticket.create 要求写 scope；deny 后不执行。"),
    }
    retriever = NoisyRetriever(storage)
    mapper = EvidenceMappingPipeline(FixedSessionReader(session), storage, retriever)
    return session, binding, storage, docs, retriever, mapper


def _request(session: SessionRecord, message: str):
    payload = RequestPayload(message=message)
    request = RequestRecord(
        session_id=session.session_id,
        turn_id=uuid4(),
        request_id=f"turn-{session.revision + 1}",
        request_digest=request_digest(payload, session.revision),
        expected_revision=session.revision,
        status="queued",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    scope = HistoryScope(
        session_id=session.session_id,
        binding=DatabaseBinding(
            database_id=session.database_id,
            storage_fingerprint=session.storage_fingerprint,
        ),
        as_of_revision=session.revision,
    )
    return scope, request, payload


def _inputs(turn: int, message: str, docs: dict[str, str], previous):
    questions, objects = _frozen_versions(turn, message)
    allowed = {
        "general_policy": [docs["policy"], docs["shared"]],
        "draft_title": [docs["draft"], docs["shared"], docs["platform"]],
        "independent_ticket": [docs["ticket"]],
    }
    tasks = []
    for question, obj in zip(questions, objects, strict=True):
        facts = []
        if turn == 5:
            facts = [
                UserProvidedFact(field=field, source_excerpt=excerpt)
                for field, excerpt in (
                    ("schema", "参数Schema校验通过"),
                    ("authentication", "身份认证通过"),
                    ("ownership", "租户和目标资源归属检查通过"),
                    ("scope", "授权上下文只有ticket.read"),
                    ("action", "请求动作是ticket.create"),
                    ("decision", "权限策略返回deny"),
                    ("reason", "拒绝原因是missing_write_scope"),
                    ("executor", "执行器没有调用后面的业务处理接口"),
                    ("record", "也没有创建工单记录"),
                )
            ]
        previous_entry = previous.get(question.question_id)
        tasks.append(
            EvidenceTaskInput(
                task_key=f"turn-{turn}-q{question.question_no}",
                question=question,
                objects=[obj],
                allowed_document_ids=allowed[obj.kind],
                user_facts=facts,
                prior_evidence_refs=previous_entry[0] if previous_entry else [],
                prior_object_refs=previous_entry[1] if previous_entry else None,
            )
        )
    return tasks


def test_phase_zero_five_turn_evidence_boundaries() -> None:
    messages = _case_messages()
    session, binding, _, docs, retriever, mapper = _runtime()
    previous = {}
    batches = []
    for turn, message in enumerate(messages, start=1):
        session.revision = turn - 1
        scope, request, payload = _request(session, message)
        tasks = _inputs(turn, message, docs, previous)
        batch = mapper.map_evidence(scope, request, payload, tasks)
        batches.append(batch)
        assert batch.binding == binding
        assert batch.as_of_revision == turn - 1
        assert batch.session_id == session.session_id
        for item, task in zip(batch.tasks, tasks, strict=True):
            assert item.current_fact_turn_id == request.turn_id
            assert all(
                evidence_id in batch.evidence_pool for evidence_id in item.allowed_evidence_ids
            )
            assert len(item.allowed_evidence_ids) == len(set(item.allowed_evidence_ids))
            assert item.sufficiency != "verified"
            allowed = set(task.allowed_document_ids or [])
            assert all(
                batch.evidence_pool[evidence_id].ref.document_id in allowed
                for evidence_id in item.allowed_evidence_ids
            )
            previous[task.question.question_id] = (
                [batch.evidence_pool[eid].ref for eid in item.allowed_evidence_ids],
                task.question.object_refs,
            )

    first_ticket = batches[0].tasks[2]
    assert first_ticket.decision == "waiting_input"
    assert first_ticket.allowed_evidence_ids == []
    assert first_ticket.user_facts == []
    assert batches[1].tasks[0].prior_evidence_status == "needs_refresh"
    assert batches[2].tasks[1].prior_evidence_status == "needs_refresh"
    assert len(batches[0].evidence_pool) < sum(
        len(item.allowed_evidence_ids) for item in batches[0].tasks
    )
    assert len(batches[3].tasks) == 1
    assert batches[3].tasks[0].question_id == batches[2].tasks[1].question_id
    assert all("independent_ticket" not in query for query in retriever.calls[:-1])
    last_ticket = batches[4].tasks[0]
    assert last_ticket.decision == "candidates"
    assert {fact.field for fact in last_ticket.user_facts} >= {"scope", "action", "reason"}
    assert all(fact.source_excerpt in messages[4] for fact in last_ticket.user_facts)
    assert not any(
        "missing_write_scope" in entry.text for entry in batches[4].evidence_pool.values()
    )


def test_phase_zero_scope_staleness_and_unquoted_fact_rejected() -> None:
    message = _case_messages()[4]
    session, binding, storage, docs, _, mapper = _runtime()
    session.revision = 4
    scope, request, payload = _request(session, message)
    tasks = _inputs(5, message, docs, {})
    tasks[0].user_facts.append(UserProvidedFact(field="reason", source_excerpt="未提供的原因"))
    with pytest.raises(EvidenceMappingError, match="current message"):
        mapper.map_evidence(scope, request, payload, tasks)
    tasks[0].user_facts.pop()
    storage.settings.registered_db_id = str(uuid4())
    with pytest.raises(EvidenceMappingError) as error:
        mapper.map_evidence(scope, request, payload, tasks)
    assert error.value.code == "DATABASE_MISMATCH"
    storage.settings.registered_db_id = str(binding.database_id)
    session.revision = 5
    with pytest.raises(EvidenceMappingError) as error:
        mapper.map_evidence(scope, request, payload, tasks)
    assert error.value.code == "STALE_REVISION"


def test_phase_zero_stale_point_no_candidates_and_technical_fault() -> None:
    message = _case_messages()[3]
    session, _, storage, docs, retriever, mapper = _runtime()
    session.revision = 3
    scope, request, payload = _request(session, message)
    task = _inputs(4, message, docs, {})[0]
    task.allowed_document_ids = []
    assert mapper.map_evidence(scope, request, payload, [task]).tasks[0].decision == "no_candidates"
    task.allowed_document_ids = [docs["draft"]]
    storage.points[docs["policy"]].clear()
    unrelated = mapper.map_evidence(scope, request, payload, [task]).tasks[0]
    assert unrelated.decision == "candidates"
    assert unrelated.stale_candidate_ids == []
    storage.points[docs["draft"]].clear()
    result = mapper.map_evidence(scope, request, payload, [task]).tasks[0]
    assert result.decision == "needs_refresh"
    assert result.stale_candidate_ids
    retriever.fail = True
    result = mapper.map_evidence(scope, request, payload, [task]).tasks[0]
    assert result.decision == "technical_failure"
    assert result.error_code == "RETRIEVAL_FAILED"
    assert result.allowed_evidence_ids == []


def test_phase_zero_debug_uses_trace_and_validates_prior_hash() -> None:
    message = _case_messages()[1]
    session, _, storage, docs, retriever, mapper = _runtime()
    session.revision = 1
    scope, request, payload = _request(session, message)
    task = _inputs(2, message, docs, {})[0]
    first = mapper.map_trace(scope, request, payload, [task])
    assert retriever.trace_calls == 1
    ref = first.evidence_pool[first.tasks[0].allowed_evidence_ids[0]].ref
    task.prior_evidence_refs = [ref]
    task.prior_object_refs = task.question.object_refs
    assert mapper.map_evidence(scope, request, payload, [task]).tasks[0].prior_evidence_status == (
        "valid"
    )
    task.prior_evidence_refs = [
        EvidenceRef(**{**ref.model_dump(), "node_hash": "0" * 64})
    ]
    assert mapper.map_evidence(scope, request, payload, [task]).tasks[0].prior_evidence_status == (
        "needs_refresh"
    )
    storage.docstore.docs[ref.document_id].metadata["content_hash"] = "changed"
    changed = mapper.map_evidence(scope, request, payload, [task]).tasks[0]
    assert changed.prior_evidence_status == "needs_refresh"
    assert ref.node_id in changed.stale_candidate_ids
    assert changed.decision == "candidates"


def test_phase_zero_duplicate_query_reuses_search_not_permissions() -> None:
    message = _case_messages()[4]
    session, _, _, docs, retriever, mapper = _runtime()
    session.revision = 4
    scope, request, payload = _request(session, message)
    first = _inputs(5, message, docs, {})[0]
    second = first.model_copy(deep=True)
    second.task_key = "turn-5-ticket-follow-up"
    second.question.question_id = uuid4()
    batch = mapper.map_evidence(scope, request, payload, [first, second])
    assert len(retriever.calls) == 1
    assert batch.tasks[0].retrieval_reused is False
    assert batch.tasks[1].retrieval_reused is True
    assert batch.tasks[1].retrieval_latency_ms == 0
    assert batch.tasks[0].allowed_evidence_ids == batch.tasks[1].allowed_evidence_ids
    assert len(batch.evidence_pool) == len(batch.tasks[0].allowed_evidence_ids)


def test_phase_zero_final_revalidation_marks_only_changed_source() -> None:
    message = _case_messages()[0]
    session, _, storage, docs, _, mapper = _runtime()
    scope, request, payload = _request(session, message)
    tasks = _inputs(1, message, docs, {})
    batch = mapper.map_evidence(scope, request, payload, tasks)
    assert mapper.validate_batch(scope, request, payload, tasks, batch) == set()

    storage.docstore.docs[docs["policy"]].metadata["content_hash"] = "updated"
    assert mapper.validate_batch(scope, request, payload, tasks, batch) == {
        tasks[0].task_key
    }


def test_phase_zero_real_storage_adapter_with_isolated_qdrant(settings) -> None:
    message = _case_messages()[0]
    database_id = uuid4()
    settings = dataclasses.replace(settings, registered_db_id=str(database_id))
    container = build_container(
        settings,
        models=ModelBundle(
            llm=ScriptedLLM(),
            embedding=MockEmbedding(embed_dim=8),
            reranker=KeywordReranker(top_n=3),
        ),
    )
    try:
        ingested = container.ingestion.ingest_bytes(
            "服务端鉴权独立于提示词和用户确认。".encode(),
            source_name="policy.txt",
        )
        binding = DatabaseBinding(
            database_id=database_id, storage_fingerprint=container.storage.database_id
        )
        session = SessionRecord(
            session_id=uuid4(),
            name="phase-zero",
            revision=0,
            database_id=binding.database_id,
            storage_fingerprint=binding.storage_fingerprint,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        scope, request, payload = _request(session, message)
        questions, objects = _frozen_versions(1, message)
        task = EvidenceTaskInput(
            task_key="t1",
            question=questions[0],
            objects=[objects[0]],
            allowed_document_ids=[ingested.document_id],
        )
        mapper = EvidenceMappingPipeline(
            FixedSessionReader(session), container.storage, container.retrieval
        )
        batch = mapper.map_evidence(scope, request, payload, [task])
        result = batch.tasks[0]
        assert result.decision == "candidates", result.model_dump()
        assert result.allowed_evidence_ids
        entry = batch.evidence_pool[result.allowed_evidence_ids[0]]
        assert entry.ref.document_id == ingested.document_id
        assert entry.ref.document_version == ingested.content_hash
        assert entry.ref.node_id in container.storage.document_point_ids(ingested.document_id)
    finally:
        asyncio.run(container.storage.aclose())
