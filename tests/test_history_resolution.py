from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from story_rag.adapters import session_store as session_store_module
from story_rag.adapters.session_store import PostgresSessionStore
from story_rag.conversation_domain import DatabaseBinding, LedgerError, RequestPayload
from story_rag.pipelines.execution_preflight import (
    AuthorizationState,
    ConfirmationGrant,
    WriteAttempt,
    WritePolicy,
    check_write_preflight,
    parameters_digest,
)
from story_rag.pipelines.history_resolution import (
    ConditionPatch,
    GapResolution,
    HistoryLookup,
    HistoryResolutionError,
    HistoryResolutionPipeline,
    HistoryScope,
    RevisionSpec,
)
from story_rag.pipelines.task_planning import TaskPlanningPipeline
from tests.conftest import ScriptedLLM
from tests.test_conversation_ledger import (
    OBJECT_IDS,
    QUESTION_IDS,
    _case_messages,
    _frozen_versions,
    postgres_dsn,
)
from tests.test_task_planning import _frozen_plan

pytestmark = pytest.mark.contract
__all__ = ["postgres_dsn"]


def _scope(session, binding, revision: int) -> HistoryScope:
    return HistoryScope(
        session_id=session.session_id,
        binding=binding,
        as_of_revision=revision,
    )


def _specs(turn: int, message: str) -> list[RevisionSpec]:
    plan = (
        TaskPlanningPipeline(ScriptedLLM())
        .validate_candidate(message, _frozen_plan(message, turn))
        .plan
    )
    assert plan is not None
    task = {item.task_key: item for item in plan.tasks}
    expected_references = {
        2: {"t1": "Agent写权限", "t2": "修改草稿标题"},
        3: {"t1": "写权限控制", "t2": "草稿修改流程"},
        4: {"t1": "修改草稿标题"},
        5: {"t1": "独立工单", "t2": "独立工单"},
    }
    for key, anchor in expected_references[turn].items():
        assert anchor in task[key].history_reference.source_excerpt

    def spec(key: str, kind: str, question_no: int, *, patches=(), gaps=()):
        item = task[key]
        return RevisionSpec(
            task_key=key,
            kind=kind,
            question_text=item.objective.text,
            source_excerpt=item.objective.source_excerpt,
            lookup=HistoryLookup(question_no=question_no),
            patches=list(patches),
            resolved_gaps=list(gaps),
            requirements=[
                *([entry.text for entry in item.constraints]),
                *([entry.text for entry in item.prohibited_actions]),
            ],
        )

    def add(field: str, value: str, excerpt: str) -> ConditionPatch:
        assert excerpt in message
        return ConditionPatch(
            field=field,
            op="supplement",
            value=value,
            source_excerpt=excerpt,
        )

    if turn == 2:
        return [
            spec(
                "t1",
                "supplement",
                1,
                patches=[
                    add("tenancy", "多租户", "改成多租户"),
                    add("execution_mode", "后台队列", "进入后台队列"),
                    add("executor", "共享服务账号", "使用共享服务账号"),
                ],
            ),
            spec(
                "t2",
                "supplement",
                2,
                patches=[
                    add("tenancy", "多租户", "改成多租户"),
                    add("execution_mode", "后台队列", "进入后台队列"),
                    add("executor", "共享服务账号", "使用共享服务账号"),
                ],
            ),
        ]
    if turn == 3:
        return [
            spec(
                "t1",
                "supplement",
                1,
                patches=[
                    add("confirmation_boundary", "服务端授权", "用户确认和服务端授权之间"),
                ],
            ),
            spec(
                "t2",
                "supplement",
                2,
                patches=[
                    add("confirmation_rule", "提交前复核", "提交前复核"),
                ],
            ),
        ]
    if turn == 4:
        return [
            spec(
                "t1",
                "replace",
                2,
                patches=[
                    add("execution_channel", "人工审批平台", "人工审批平台执行"),
                    add("assistant_role", "待审批的变更单", "待审批的变更单"),
                ],
            ),
        ]
    if turn == 5:
        first = spec(
            "t1",
            "supplement",
            3,
            patches=[
                add("granted_scope", "ticket.read", "授权上下文只有ticket.read"),
                add("requested_action", "ticket.create", "请求动作是ticket.create"),
                add("policy_decision", "deny", "权限策略返回deny"),
                add("denial_reason", "missing_write_scope", "拒绝原因是missing_write_scope"),
            ],
            gaps=[
                GapResolution(
                    missing_input="工单调用记录",
                    source_excerpt="现在补充刚才那个独立工单测试环境的调用记录",
                )
            ],
        )
        item = task["t2"]
        second = RevisionSpec(
            task_key="t2",
            kind="extend",
            question_text=item.objective.text,
            source_excerpt=item.objective.source_excerpt,
            lookup=HistoryLookup(question_no=3),
            requirements=[entry.text for entry in item.constraints],
        )
        return [first, second]
    raise AssertionError(turn)


@pytest.mark.usefixtures("postgres_dsn")
def test_five_turn_history_resolution_and_revision_isolation(postgres_dsn: str) -> None:
    messages = _case_messages()
    store = PostgresSessionStore(postgres_dsn, schema=f"history_{uuid4().hex[:12]}")
    assert store.migrate() == 5
    binding = DatabaseBinding(database_id=uuid4(), storage_fingerprint="a" * 64)
    session = store.create_session("固定五轮", binding)
    resolver = HistoryResolutionPipeline(store)
    first_payload = RequestPayload(message=messages[0])
    store.claim_request(
        session.session_id,
        binding,
        request_id="turn-1",
        payload=first_payload,
        expected_revision=0,
    )
    questions, objects = _frozen_versions(1, messages[0])
    store.commit_request(
        session.session_id,
        binding,
        "turn-1",
        questions=questions,
        objects=objects,
        payload=first_payload,
    )
    scope = _scope(session, binding, 1)
    pending = resolver.resolve(scope, HistoryLookup(pending_only=True))
    assert pending.selected.question_id == QUESTION_IDS[2]
    assert pending.selected.status == "waiting_input"
    assert (
        resolver.resolve(scope, HistoryLookup(turn_sequence=1, question_no=3)).selected.question_id
        == QUESTION_IDS[2]
    )
    assert (
        resolver.resolve(scope, HistoryLookup(turn_sequence=1, question_no=99)).decision
        == "not_found"
    )
    assert resolver.resolve(scope, HistoryLookup(turn_sequence=1)).decision == "ambiguous"

    for turn in range(2, 6):
        payload = RequestPayload(message=messages[turn - 1])
        scope = _scope(session, binding, turn - 1)
        request = store.claim_request(
            session.session_id,
            binding,
            request_id=f"turn-{turn}",
            payload=payload,
            expected_revision=turn - 1,
        )
        batch = resolver.prepare_batch(scope, request, payload, _specs(turn, payload.message))
        if turn == 2:
            with pytest.raises(HistoryResolutionError) as mismatched_history:
                resolver.prepare_batch(
                    scope.model_copy(update={"use_history": False}),
                    request,
                    payload,
                    _specs(turn, payload.message),
                )
            assert mismatched_history.value.code == "REQUEST_MISMATCH"
            disabled_payload = RequestPayload(message=payload.message, use_history=False)
            disabled_request = store.claim_request(
                session.session_id,
                binding,
                request_id="turn-2-no-history",
                payload=disabled_payload,
                expected_revision=1,
            )
            with pytest.raises(HistoryResolutionError) as disabled_history:
                resolver.prepare_batch(
                    scope.model_copy(update={"use_history": False}),
                    disabled_request,
                    disabled_payload,
                    _specs(turn, payload.message),
                )
            assert disabled_history.value.code == "HISTORY_DISABLED"
            bad_specs = _specs(turn, payload.message)
            bad_patch = bad_specs[0].patches[0].model_copy(update={"value": "invented"})
            bad_specs[0] = bad_specs[0].model_copy(
                update={"patches": [bad_patch, *bad_specs[0].patches[1:]]}
            )
            with pytest.raises(HistoryResolutionError) as invented:
                resolver.prepare_batch(scope, request, payload, bad_specs)
            assert invented.value.code == "UNVERIFIED_VALUE"
            with pytest.raises(LedgerError) as absent_source:
                store.commit_request(
                    session.session_id,
                    binding,
                    request.request_id,
                    questions=batch.questions,
                    objects=batch.objects,
                )
            assert absent_source.value.code == "SOURCE_REQUIRED"
            altered = batch.objects[0].model_copy(
                update={"conditions": {**batch.objects[0].conditions, "stage": "invented"}}
            )
            with pytest.raises(LedgerError) as untracked_change:
                store.commit_request(
                    session.session_id,
                    binding,
                    request.request_id,
                    questions=batch.questions,
                    objects=[altered, *batch.objects[1:]],
                    payload=payload,
                )
            assert untracked_change.value.code == "INVALID_CHANGE"
            bad_change = (
                batch.objects[0]
                .changes[0]
                .model_copy(update={"source_excerpt": "not in the user message"})
            )
            altered_source = batch.objects[0].model_copy(
                update={"changes": [bad_change, *batch.objects[0].changes[1:]]}
            )
            with pytest.raises(LedgerError) as unverified_source:
                store.commit_request(
                    session.session_id,
                    binding,
                    request.request_id,
                    questions=batch.questions,
                    objects=[altered_source, *batch.objects[1:]],
                    payload=payload,
                )
            assert unverified_source.value.code == "INVALID_SOURCE"
            assert store.get_session(session.session_id, binding).revision == 1
        assert all(outcome.decision == "prepared" for outcome in batch.outcomes)
        assert all(item.source_excerpt in payload.message for item in batch.questions)
        assert all(
            change.source_turn_id == request.turn_id and change.source_excerpt in payload.message
            for obj in batch.objects
            for change in obj.changes
        )
        store.commit_request(
            session.session_id,
            binding,
            request.request_id,
            questions=batch.questions,
            objects=batch.objects,
            payload=payload,
        )
        if turn == 3:
            general_v3 = resolver.resolve(_scope(session, binding, 3), HistoryLookup(question_no=1))
            assert general_v3.selected.version == 3
            assert general_v3.objects[0].conditions["confirmation_boundary"] == "服务端授权"
        if turn == 4:
            general = resolver.resolve(_scope(session, binding, 4), HistoryLookup(question_no=1))
            draft = resolver.resolve(_scope(session, binding, 4), HistoryLookup(question_no=2))
            assert general.selected.version == 3
            assert "execution_channel" not in general.objects[0].conditions
            assert draft.selected.version == 4
            assert draft.objects[0].conditions["execution_channel"] == "人工审批平台"
            assert "禁止发布、删除或通知" in draft.selected.requirements
        if turn == 5:
            ticket = resolver.resolve(_scope(session, binding, 5), HistoryLookup(question_no=3))
            auth_decision = resolver.resolve(
                _scope(session, binding, 5), HistoryLookup(question_no=4)
            )
            assert ticket.selected.status == "ready"
            assert ticket.selected.missing_inputs == []
            assert "说明需要的调用记录和错误日志" not in ticket.selected.requirements
            assert ticket.objects[0].conditions["denial_reason"] == "missing_write_scope"
            assert auth_decision.selected.relation.target.question_id == QUESTION_IDS[2]
            assert auth_decision.objects[0].object_id == OBJECT_IDS[2]
            assert auth_decision.objects[0].version == 2
            assert auth_decision.objects[0].object_id != OBJECT_IDS[1]

    assert (
        resolver.resolve(_scope(session, binding, 3), HistoryLookup(question_no=2)).selected.version
        == 3
    )
    wrong_binding = DatabaseBinding(
        database_id=binding.database_id,
        storage_fingerprint="b" * 64,
    )
    with pytest.raises(LedgerError) as wrong_scope:
        resolver.resolve(_scope(session, wrong_binding, 5), HistoryLookup(question_no=2))
    assert wrong_scope.value.code == "SCOPE_MISMATCH"
    wrong_database = DatabaseBinding(database_id=uuid4(), storage_fingerprint="a" * 64)
    with pytest.raises(LedgerError) as wrong_database_scope:
        resolver.resolve(_scope(session, wrong_database, 5), HistoryLookup(question_no=2))
    assert wrong_database_scope.value.code == "SCOPE_MISMATCH"
    other = store.create_session("独立会话", binding)
    assert (
        resolver.resolve(
            _scope(other, binding, 0), HistoryLookup(question_id=QUESTION_IDS[1])
        ).decision
        == "not_found"
    )
    with pytest.raises(LedgerError) as future:
        resolver.resolve(_scope(session, binding, 6), HistoryLookup(question_no=1))
    assert future.value.code == "REVISION_CONFLICT"
    with pytest.raises(HistoryResolutionError) as disabled:
        resolver.resolve(
            _scope(session, binding, 5).model_copy(update={"use_history": False}),
            HistoryLookup(question_no=1),
        )
    assert disabled.value.code == "HISTORY_DISABLED"
    assert store.get_session(session.session_id, binding).revision == 5
    assert (
        store.list_question_versions(session.session_id, binding, QUESTION_IDS[1])[-1].relation.kind
        == "replace"
    )
    assert store.list_object_versions(session.session_id, binding)[-1].source_turn_id is not None


def test_v1_ledger_migrates_to_history_revision(postgres_dsn: str) -> None:
    store = PostgresSessionStore(postgres_dsn, schema=f"upgrade_{uuid4().hex[:12]}")
    with psycopg.connect(postgres_dsn) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(store.schema)))
        conn.execute(
            sql.SQL("SET LOCAL search_path TO {}, pg_catalog").format(sql.Identifier(store.schema))
        )
        conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY)")
        conn.execute(session_store_module._migration_sql(), prepare=False)
        conn.execute("INSERT INTO schema_migrations (version) VALUES (1)")
    assert store.migrate() == 5
    assert store.migrate() == 5
    with psycopg.connect(postgres_dsn) as conn:
        columns = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = 'object_versions'",
                (store.schema,),
            )
        }
    assert {"changes", "derived_from"} <= columns


def test_confirmation_and_authorization_are_separate_checks() -> None:
    now = datetime(2026, 10, 23, tzinfo=UTC)
    digest = parameters_digest({"title": "new title"})
    grant = ConfirmationGrant(
        subject_id="user-1",
        tenant_id="tenant-1",
        action="draft.title.update",
        resource_id="draft-1",
        parameters_digest=digest,
        resource_version="v1",
        authorization_version="a1",
        expires_at=now + timedelta(minutes=5),
    )
    attempt = WriteAttempt(
        subject_id="user-1",
        tenant_id="tenant-1",
        action="draft.title.update",
        resource_id="draft-1",
        parameters_digest=digest,
        resource_version="v1",
        executor="assistant",
    )
    auth = AuthorizationState(
        subject_id="user-1",
        tenant_id="tenant-1",
        resource_id="draft-1",
        scopes=frozenset({"draft.title.update"}),
        version="a1",
    )
    normal = WritePolicy(
        allowed_actions=frozenset({"draft.title.update"}),
        required_scope="draft.title.update",
        required_executor="assistant",
    )
    assert check_write_preflight(grant, attempt, auth, normal, now=now).can_execute
    changed = {
        "parameters_digest": parameters_digest({"title": "other"}),
        "resource_id": "draft-2",
        "resource_version": "v2",
        "subject_id": "user-2",
        "tenant_id": "tenant-2",
        "action": "draft.publish",
    }
    for field, value in changed.items():
        result = check_write_preflight(
            grant,
            attempt.model_copy(update={field: value}),
            auth,
            normal,
            now=now,
        )
        assert not result.can_execute
        assert not result.confirmation_valid
    assert not check_write_preflight(
        grant,
        attempt,
        auth.model_copy(update={"version": "a2"}),
        normal,
        now=now,
    ).confirmation_valid
    assert not check_write_preflight(
        grant,
        attempt,
        auth.model_copy(update={"revoked": True}),
        normal,
        now=now,
    ).authorization_valid
    assert not check_write_preflight(
        grant,
        attempt,
        auth,
        normal,
        now=now + timedelta(minutes=5),
    ).confirmation_valid
    assert not check_write_preflight(
        grant.model_copy(update={"revoked_at": now}),
        attempt,
        auth,
        normal,
        now=now,
    ).confirmation_valid
    assert not check_write_preflight(
        grant,
        attempt,
        auth.model_copy(update={"scopes": frozenset()}),
        normal,
        now=now,
    ).authorization_valid
    for field, value in (
        ("subject_id", "user-2"),
        ("tenant_id", "tenant-2"),
        ("resource_id", "draft-2"),
    ):
        assert not check_write_preflight(
            grant,
            attempt,
            auth.model_copy(update={field: value}),
            normal,
            now=now,
        ).authorization_valid

    platform = normal.model_copy(
        update={
            "required_executor": "approval_platform",
            "platform_approval_required": True,
        }
    )
    assert not check_write_preflight(grant, attempt, auth, platform, now=now).can_execute
    platform_attempt = attempt.model_copy(update={"executor": "approval_platform"})
    assert not check_write_preflight(grant, platform_attempt, auth, platform, now=now).can_execute
    assert check_write_preflight(
        grant,
        platform_attempt,
        auth,
        platform,
        now=now,
        platform_approved=True,
    ).can_execute
    ticket = WritePolicy(
        allowed_actions=frozenset({"ticket.create"}),
        required_scope="ticket.create",
        required_executor="assistant",
    )
    ticket_attempt = attempt.model_copy(
        update={
            "action": "ticket.create",
            "resource_id": "ticket-target",
        }
    )
    result = check_write_preflight(
        grant,
        ticket_attempt,
        AuthorizationState(
            subject_id="user-1",
            tenant_id="tenant-1",
            resource_id="ticket-target",
            scopes=frozenset({"ticket.read"}),
            version="a1",
        ),
        ticket,
        now=now,
        platform_approved=True,
    )
    assert not result.can_execute
    assert "REQUIRED_SCOPE_MISSING" in result.reasons
    assert "CONFIRMATION_ACTION_CHANGED" in result.reasons
