from __future__ import annotations

import re
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import psycopg
import pytest
from pydantic import ValidationError

from story_rag.adapters import session_store as session_store_module
from story_rag.adapters.session_store import PostgresSessionStore
from story_rag.conversation_domain import (
    DatabaseBinding,
    LedgerError,
    ObjectRef,
    ObjectVersionDraft,
    QuestionRef,
    QuestionRelation,
    QuestionVersionDraft,
    RequestPayload,
    UsageMetadata,
)

pytestmark = pytest.mark.contract
ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "docs" / "多轮询问模块重建子方案" / "00-唯一连续回归用例.md"
PG_BIN = ROOT / "tools" / "postgresql" / "17.11-5" / "pgsql" / "bin"
QUESTION_IDS = [uuid.uuid5(uuid.NAMESPACE_URL, f"ledger-case-question-{n}") for n in range(1, 4)]
OBJECT_IDS = [uuid.uuid5(uuid.NAMESPACE_URL, f"ledger-case-object-{n}") for n in range(1, 4)]
VERSIONS = [(1, 1, 1), (2, 2, 0), (3, 3, 0), (0, 4, 0), (0, 0, 2)]


def _case_messages() -> list[str]:
    source = CASE_PATH.read_text(encoding="utf-8")
    messages = re.findall(r"(?ms)^## 第 [1-5] 轮\s*\n\s*> (.+?)(?=\n\n## |\Z)", source)
    assert len(messages) == 5
    return [message.strip() for message in messages]


def _frozen_versions(turn: int, message: str):
    questions = []
    objects = []
    for index, version in enumerate(VERSIONS[turn - 1]):
        if version == 0:
            continue
        objects.append(
            ObjectVersionDraft(
                object_id=OBJECT_IDS[index],
                version=version,
                kind=("general_policy", "draft_title", "independent_ticket")[index],
                conditions={"stage": str(turn), "scope": ("general", "draft", "ticket")[index]},
                source_excerpt=message[:120],
            )
        )
        status = "waiting_input" if turn == 1 and index == 2 else "ready"
        relation = (
            QuestionRelation(
                kind="replace" if turn in {3, 4} else "follow_up",
                target=QuestionRef(question_id=QUESTION_IDS[index], version=version - 1),
            )
            if version > 1
            else None
        )
        questions.append(
            QuestionVersionDraft(
                question_id=QUESTION_IDS[index],
                question_no=index + 1,
                version=version,
                question_text=("Agent 写权限原则", "草稿标题修改流程", "独立工单失败诊断")[index],
                requirements=["保持当前对象边界"],
                status=status,
                relation=relation,
                object_refs=[ObjectRef(object_id=OBJECT_IDS[index], version=version)],
                missing_inputs=["工单调用记录"] if status == "waiting_input" else [],
                source_excerpt=message[:120],
            )
        )
    return questions, objects


@pytest.fixture(scope="module")
def postgres_dsn():
    if not (PG_BIN / "initdb.exe").is_file():
        pytest.fail("Isolated PostgreSQL binaries are missing")
    parent = ROOT / "data" / "postgres"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ledger_test_", dir=parent) as temporary:
        root = Path(temporary)
        cluster = root / "cluster"
        init = subprocess.run(
            [str(PG_BIN / "initdb.exe"), "-D", str(cluster), "-A", "trust", "-U", "postgres"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        assert init.returncode == 0, init.stderr[-1000:]
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        dsn = f"host=127.0.0.1 port={port} user=postgres dbname=postgres"
        with (root / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    str(PG_BIN / "postgres.exe"),
                    "-D",
                    str(cluster),
                    "-h",
                    "127.0.0.1",
                    "-p",
                    str(port),
                ],
                stdout=log,
                stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            try:
                for _ in range(100):
                    if process.poll() is not None:
                        pytest.fail("Isolated PostgreSQL exited before accepting connections")
                    try:
                        with psycopg.connect(dsn, connect_timeout=1):
                            break
                    except psycopg.OperationalError:
                        time.sleep(0.1)
                else:
                    pytest.fail("Isolated PostgreSQL startup timed out")
                yield dsn
            finally:
                subprocess.run(
                    [str(PG_BIN / "pg_ctl.exe"), "-D", str(cluster), "stop", "-m", "fast", "-w"],
                    capture_output=True,
                    timeout=30,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=15)


def _store(dsn: str) -> PostgresSessionStore:
    return PostgresSessionStore(dsn, schema=f"ledger_{uuid.uuid4().hex[:12]}")


def test_postgres_session_store_normalizes_asyncpg_url() -> None:
    store = PostgresSessionStore(
        "postgresql+asyncpg://story_rag:example@127.0.0.1:55432/story_sessions"
    )

    assert store._dsn == "postgresql://story_rag:example@127.0.0.1:55432/story_sessions"


def test_five_turn_ledger_contract_and_restore(postgres_dsn: str) -> None:
    messages = _case_messages()
    store = _store(postgres_dsn)
    assert store.migrate() == 5
    assert store.migrate() == 5
    binding = DatabaseBinding(database_id=uuid.uuid4(), storage_fingerprint="a" * 64)
    session = store.create_session("写权限与工单诊断", binding)
    assert store.list_sessions(binding)[0].session_id == session.session_id
    assert session.revision == 0
    wrong_scope = DatabaseBinding(database_id=binding.database_id, storage_fingerprint="b" * 64)
    with pytest.raises(LedgerError, match="another database scope"):
        store.get_session(session.session_id, wrong_scope)
    assert store.list_sessions(wrong_scope) == []

    for sequence, message in enumerate(messages, 1):
        request_id = f"fixed-case-turn-{sequence}"
        payload = RequestPayload(message=message)
        request = store.claim_request(
            session.session_id,
            binding,
            request_id=request_id,
            payload=payload,
            expected_revision=sequence - 1,
        )
        assert request.status == "queued"
        assert (
            store.claim_request(
                session.session_id,
                binding,
                request_id=request_id,
                payload=payload,
                expected_revision=sequence - 1,
            ).turn_id
            == request.turn_id
        )
        with pytest.raises(LedgerError) as conflict:
            store.claim_request(
                session.session_id,
                binding,
                request_id=request_id,
                payload=RequestPayload(message=messages[sequence % 5]),
                expected_revision=sequence - 1,
            )
        assert conflict.value.code == "IDEMPOTENCY_CONFLICT"
        with pytest.raises(LedgerError) as changed_options:
            store.claim_request(
                session.session_id,
                binding,
                request_id=request_id,
                payload=RequestPayload(message=message, use_history=False),
                expected_revision=sequence - 1,
            )
        assert changed_options.value.code == "IDEMPOTENCY_CONFLICT"
        questions, objects = _frozen_versions(sequence, message)
        if sequence == 2:
            invalid = questions[0].model_copy(update={"question_no": 99})
            with pytest.raises(LedgerError) as version_error:
                store.commit_request(
                    session.session_id,
                    binding,
                    request_id,
                    questions=[invalid, *questions[1:]],
                    objects=objects,
                )
            assert version_error.value.code == "VERSION_CONFLICT"
            assert store.get_session(session.session_id, binding).revision == 1
            foreign_object = questions[0].model_copy(
                update={"object_refs": [ObjectRef(object_id=uuid.uuid4(), version=1)]}
            )
            with pytest.raises(LedgerError) as scope_error:
                store.commit_request(
                    session.session_id,
                    binding,
                    request_id,
                    questions=[foreign_object, *questions[1:]],
                    objects=objects,
                )
            assert scope_error.value.code == "SCOPE_MISMATCH"
        committed = store.commit_request(
            session.session_id,
            binding,
            request_id,
            questions=questions,
            objects=objects,
            usage=UsageMetadata(llm_calls=0),
        )
        assert committed.turn_sequence == sequence
        assert committed.status == "succeeded"
        assert len(committed.result_refs) == len(questions)
        assert (
            store.commit_request(
                session.session_id, binding, request_id, questions=questions, objects=objects
            ).turn_id
            == request.turn_id
        )
        assert store.get_session(session.session_id, binding).revision == sequence
        if sequence == 1:
            with pytest.raises(LedgerError) as stale:
                store.claim_request(
                    session.session_id,
                    binding,
                    request_id="stale-fixed-case",
                    payload=RequestPayload(message=messages[1]),
                    expected_revision=0,
                )
            assert stale.value.code == "REVISION_CONFLICT"

    reopened = PostgresSessionStore(postgres_dsn, schema=store.schema)
    assert reopened.get_session(session.session_id, binding).revision == 5
    replayed = reopened.claim_request(
        session.session_id,
        binding,
        request_id="fixed-case-turn-5",
        payload=RequestPayload(message=messages[4]),
        expected_revision=4,
    )
    assert replayed.status == "succeeded" and replayed.turn_sequence == 5
    versions = reopened.list_question_versions(session.session_id, binding)
    assert [(row.question_no, row.version) for row in versions] == [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 1),
        (3, 2),
    ]
    assert versions[-1].status == "ready"
    assert versions[-1].relation.target == QuestionRef(question_id=QUESTION_IDS[2], version=1)
    assert len(reopened.list_object_versions(session.session_id, binding)) == 9
    assert reopened.inspect_scope(session.session_id, binding)["counts"] == {
        "turn_requests": 5,
        "question_versions": 9,
        "object_versions": 9,
    }
    assert reopened.get_request(session.session_id, binding, "fixed-case-turn-5").turn_sequence == 5
    with pytest.raises(LedgerError) as wrong_ref:
        reopened.list_question_versions(session.session_id, wrong_scope)
    assert wrong_ref.value.code == "SCOPE_MISMATCH"

    with psycopg.connect(postgres_dsn) as conn:
        columns = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name IN "
                "('conversation_sessions', 'turn_requests', "
                "'question_versions', 'object_versions')",
                (store.schema,),
            )
        }
    assert not {"message", "prompt", "answer", "response_body"} & columns
    for message in messages:
        assert all(message not in row.question_text for row in versions)
    with pytest.raises(ValidationError):
        QuestionVersionDraft.model_validate(
            {**_frozen_versions(1, messages[0])[0][0].model_dump(), "answer": "不可存储"}
        )

    archived = reopened.archive_session(session.session_id, binding, expected_revision=5)
    assert archived.revision == 6
    assert reopened.list_sessions(binding) == []
    assert len(reopened.list_sessions(binding, include_archived=True)) == 1
    with pytest.raises(LedgerError) as archived_error:
        reopened.get_session(session.session_id, binding)
    assert archived_error.value.code == "SESSION_ARCHIVED"


def test_migration_rolls_back_failed_schema(postgres_dsn: str, monkeypatch) -> None:
    store = _store(postgres_dsn)
    monkeypatch.setattr(
        session_store_module,
        "_migration_sql",
        lambda: "CREATE TABLE rollback_probe (id INTEGER); SELECT 1 / 0;",
    )
    with pytest.raises(psycopg.Error):
        store.migrate()
    with psycopg.connect(postgres_dsn) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_namespace WHERE nspname = %s", (store.schema,)
        ).fetchone()
    assert exists is None


def test_competing_claims_cannot_both_commit(postgres_dsn: str) -> None:
    message = _case_messages()[0]
    store = _store(postgres_dsn)
    store.migrate()
    binding = DatabaseBinding(database_id=uuid.uuid4(), storage_fingerprint="c" * 64)
    session = store.create_session("并发请求账本", binding)
    for request_id in ("first", "second"):
        store.claim_request(
            session.session_id,
            binding,
            request_id=request_id,
            payload=RequestPayload(message=message),
            expected_revision=0,
        )
    questions, objects = _frozen_versions(1, message)
    store.commit_request(session.session_id, binding, "first", questions=questions, objects=objects)
    with pytest.raises(LedgerError) as stale:
        store.commit_request(
            session.session_id, binding, "second", questions=questions, objects=objects
        )
    assert stale.value.code == "REVISION_CONFLICT"
    assert store.get_session(session.session_id, binding).revision == 1
    assert store.get_request(session.session_id, binding, "second").status == "queued"
