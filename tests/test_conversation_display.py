from __future__ import annotations

from uuid import uuid4

import psycopg
import pytest

from story_rag.adapters.session_store import PostgresSessionStore
from story_rag.conversation_domain import DatabaseBinding, LedgerError, RequestPayload
from tests.test_conversation_ledger import postgres_dsn

pytestmark = pytest.mark.contract
__all__ = ["postgres_dsn"]


def _store(dsn: str) -> PostgresSessionStore:
    store = PostgresSessionStore(dsn, schema=f"display_{uuid4().hex[:12]}")
    assert store.migrate() == 5
    return store


def test_schema_readiness_reports_unmigrated_and_current_schema(postgres_dsn: str) -> None:
    store = PostgresSessionStore(postgres_dsn, schema=f"ready_{uuid4().hex[:12]}")
    assert store.is_ready() is False
    assert store.migrate() == 5
    assert store.is_ready() is True


def test_display_turns_keep_original_and_final_answer_after_restart(postgres_dsn: str) -> None:
    store = _store(postgres_dsn)
    binding = DatabaseBinding(database_id=uuid4(), storage_fingerprint="a" * 64)
    session = store.create_session("展示记录", binding)
    first = RequestPayload(message="第一轮原话：中英双语 question?")
    request = store.claim_request(
        session.session_id, binding, request_id="first", payload=first, expected_revision=0
    )
    pending = store.list_display_turns(session.session_id, binding).items[0]
    assert pending.user_message == first.message
    assert pending.assistant_answer is None and pending.status == "queued"
    store.start_request(session.session_id, binding, "first")
    store.commit_request(
        session.session_id,
        binding,
        "first",
        questions=[],
        payload=first,
        visible_answer="1. 最终可见回答",
    )
    assert store.claim_request(
        session.session_id, binding, request_id="first", payload=first, expected_revision=0
    ).turn_id == request.turn_id

    second = RequestPayload(message="第二轮失败的原话")
    store.claim_request(
        session.session_id, binding, request_id="second", payload=second, expected_revision=1
    )
    store.start_request(session.session_id, binding, "second")
    store.fail_request(session.session_id, binding, "second", failure_code="MODEL_UNAVAILABLE")

    reopened = PostgresSessionStore(postgres_dsn, schema=store.schema)
    page = reopened.list_display_turns(session.session_id, binding, limit=1)
    assert len(page.items) == 1 and page.next_cursor == request.turn_id
    assert page.items[0].assistant_answer == "1. 最终可见回答"
    assert reopened.get_display_answer(session.session_id, binding, request.turn_id) == (
        "1. 最终可见回答"
    )
    second_page = reopened.list_display_turns(
        session.session_id, binding, limit=1, cursor=page.next_cursor
    )
    assert second_page.next_cursor is None
    assert second_page.items[0].user_message == second.message
    assert second_page.items[0].assistant_answer is None
    assert second_page.items[0].failure_code == "MODEL_UNAVAILABLE"
    assert second_page.items[0].status == "failed"
    assert reopened.inspect_display_scope(session.session_id, binding) == {
        "stored_turns": 2,
        "stored_answers": 1,
    }
    assert reopened.get_session(session.session_id, binding).revision == 1

    wrong = DatabaseBinding(database_id=uuid4(), storage_fingerprint=binding.storage_fingerprint)
    with pytest.raises(LedgerError, match="another database scope"):
        reopened.list_display_turns(session.session_id, wrong)
    other = reopened.create_session("另一个会话", binding)
    with pytest.raises(LedgerError) as invalid_cursor:
        reopened.list_display_turns(other.session_id, binding, cursor=request.turn_id)
    assert invalid_cursor.value.code == "INVALID_CURSOR"


def test_display_failure_rolls_back_ledger_commit(postgres_dsn: str) -> None:
    store = _store(postgres_dsn)
    binding = DatabaseBinding(database_id=uuid4(), storage_fingerprint="b" * 64)
    session = store.create_session("原子写入", binding)
    payload = RequestPayload(message="原话")
    store.claim_request(
        session.session_id, binding, request_id="atomic", payload=payload, expected_revision=0
    )
    store.start_request(session.session_id, binding, "atomic")
    with store._connect() as conn:
        conn.execute(
            "ALTER TABLE conversation_display_turns ADD CONSTRAINT reject_probe "
            "CHECK (assistant_answer IS DISTINCT FROM 'forbidden')"
        )
    with pytest.raises(psycopg.errors.CheckViolation):
        store.commit_request(
            session.session_id,
            binding,
            "atomic",
            questions=[],
            payload=payload,
            visible_answer="forbidden",
        )
    assert store.get_session(session.session_id, binding).revision == 0
    assert store.get_request(session.session_id, binding, "atomic").status == "running"
    assert store.list_display_turns(session.session_id, binding).items[0].assistant_answer is None
    with store._connect() as conn:
        conn.execute("ALTER TABLE conversation_display_turns DROP CONSTRAINT reject_probe")
    committed = store.commit_request(
        session.session_id, binding, "atomic", questions=[], payload=payload,
        visible_answer="真正的最终回答",
    )
    assert committed.turn_sequence == 1
    assert store.get_display_answer(session.session_id, binding, committed.turn_id) == (
        "真正的最终回答"
    )


def test_old_turn_without_display_body_is_explicit(postgres_dsn: str) -> None:
    store = _store(postgres_dsn)
    binding = DatabaseBinding(database_id=uuid4(), storage_fingerprint="c" * 64)
    session = store.create_session("旧记录", binding)
    payload = RequestPayload(message="旧输入无法从摘要反推")
    request = store.claim_request(
        session.session_id, binding, request_id="legacy", payload=payload, expected_revision=0
    )
    store.commit_request(session.session_id, binding, "legacy", questions=[], payload=payload)
    with store._connect() as conn:
        conn.execute(
            "DELETE FROM conversation_display_turns WHERE session_id = %s AND turn_id = %s",
            (session.session_id, request.turn_id),
        )
    old = store.list_display_turns(session.session_id, binding).items[0]
    assert old.user_message is None and old.assistant_answer is None
    assert old.answer_not_retained is True
    assert store.get_display_answer(session.session_id, binding, request.turn_id) is None


def test_unknown_turn_keeps_original_but_never_candidate_answer(postgres_dsn: str) -> None:
    store = _store(postgres_dsn)
    binding = DatabaseBinding(database_id=uuid4(), storage_fingerprint="d" * 64)
    session = store.create_session("结果未知", binding)
    payload = RequestPayload(message="用户原话仍需展示", use_history=False)
    store.claim_request(
        session.session_id, binding, request_id="unknown", payload=payload, expected_revision=0
    )
    store.start_request(session.session_id, binding, "unknown")
    store.fail_request(
        session.session_id,
        binding,
        "unknown",
        failure_code="MODEL_RESULT_UNKNOWN",
        outcome_unknown=True,
    )
    item = store.list_display_turns(session.session_id, binding).items[0]
    assert item.user_message == payload.message
    assert item.assistant_answer is None
    assert item.status == "unknown"
    assert item.failure_code == "MODEL_RESULT_UNKNOWN"
    store.acknowledge_unknown_request(
        session.session_id,
        binding,
        "unknown",
        acknowledge_result_unknown=True,
    )
    acknowledged = store.list_display_turns(session.session_id, binding).items[0]
    assert acknowledged.status == "failed"
    assert acknowledged.assistant_answer is None


def test_v4_upgrade_retains_ledger_without_inventing_old_body(postgres_dsn: str) -> None:
    store = _store(postgres_dsn)
    binding = DatabaseBinding(database_id=uuid4(), storage_fingerprint="e" * 64)
    session = store.create_session("升级前轮次", binding)
    payload = RequestPayload(message="旧原文不可反推")
    store.claim_request(
        session.session_id, binding, request_id="old", payload=payload, expected_revision=0
    )
    store.commit_request(session.session_id, binding, "old", questions=[], payload=payload)
    with store._connect() as conn:
        conn.execute("DROP TABLE conversation_display_turns")
        conn.execute("DROP INDEX turn_requests_display_page_idx")
        conn.execute("DELETE FROM schema_migrations WHERE version = 5")

    assert store.migrate() == 5
    assert store.migrate() == 5
    old = store.list_display_turns(session.session_id, binding).items[0]
    assert old.status == "succeeded"
    assert old.user_message is None
    assert old.assistant_answer is None
    assert old.answer_not_retained is True
    assert store.get_session(session.session_id, binding).revision == 1
