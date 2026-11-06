from __future__ import annotations

import os
import socket
import unicodedata
import uuid
from importlib.resources import files
from typing import Any
from uuid import UUID

import psutil
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from story_rag.adapters.database_registry import (
    DatabaseRecord,
    DatabaseRegistry,
    storage_fingerprint,
)
from story_rag.config import ConfigError, Settings
from story_rag.conversation_domain import (
    ConversationDisplayPage,
    ConversationDisplayTurn,
    DatabaseBinding,
    HistoryContentQuery,
    HistoryContentSnapshot,
    LedgerError,
    ObjectRef,
    ObjectVersionDraft,
    ObjectVersionRecord,
    QuestionRef,
    QuestionVersionDraft,
    QuestionVersionRecord,
    RequestPayload,
    RequestRecord,
    SessionRecord,
    UsageMetadata,
    request_digest,
)

MIGRATION_VERSION = 5


def binding_for_database(registry: DatabaseRegistry, record: DatabaseRecord) -> DatabaseBinding:
    current = registry.resolve(db_id=record.db_id)
    return DatabaseBinding(
        database_id=current.db_id,
        storage_fingerprint=storage_fingerprint(registry.bind(current)),
    )


def build_session_store(settings: Settings) -> PostgresSessionStore:
    if not settings.postgres_dsn:
        raise ConfigError("STORY_RAG_POSTGRES_DSN is required for conversation storage")
    return PostgresSessionStore(settings.postgres_dsn, schema=settings.postgres_schema)


def _migration_sql() -> str:
    return (
        files("story_rag.migrations")
        .joinpath("001_conversation_ledger.sql")
        .read_text(encoding="utf-8")
    )


def _history_migration_sql() -> str:
    return (
        files("story_rag.migrations")
        .joinpath("002_history_revision.sql")
        .read_text(encoding="utf-8")
    )


def _index_migration_sql() -> str:
    return (
        files("story_rag.migrations")
        .joinpath("003_history_index_outbox.sql")
        .read_text(encoding="utf-8")
    )


class PostgresSessionStore:
    """Scoped, append-only conversation ledger. Migrations are explicit."""

    def __init__(self, dsn: str, *, schema: str = "story_rag") -> None:
        if not dsn.strip():
            raise LedgerError("CONFIG_ERROR", "PostgreSQL DSN is required")
        if (
            not schema.isascii()
            or not schema.replace("_", "a").isalnum()
            or not schema[0].isalpha()
        ):
            raise LedgerError("CONFIG_ERROR", "Invalid PostgreSQL schema name")
        if len(schema) > 63:
            raise LedgerError("CONFIG_ERROR", "PostgreSQL schema name is too long")
        self._dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        self.schema = schema

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn, row_factory=dict_row, connect_timeout=5)
        try:
            conn.execute(
                sql.SQL("SET search_path TO {}, pg_catalog").format(sql.Identifier(self.schema))
            )
        except Exception:
            conn.close()
            raise
        return conn

    def is_ready(self) -> bool:
        try:
            with self._connect() as conn:
                versions = [
                    row["version"]
                    for row in conn.execute(
                        "SELECT version FROM schema_migrations ORDER BY version"
                    )
                ]
        except psycopg.Error:
            return False
        return versions == list(range(1, MIGRATION_VERSION + 1))

    def migrate(self) -> int:
        with psycopg.connect(self._dsn, row_factory=dict_row, connect_timeout=5) as conn:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))", (f"story_rag:{self.schema}",)
            )
            conn.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema))
            )
            conn.execute(
                sql.SQL("SET LOCAL search_path TO {}, pg_catalog").format(
                    sql.Identifier(self.schema)
                )
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            versions = [
                row["version"] for row in conn.execute("SELECT version FROM schema_migrations")
            ]
            if sorted(versions) == list(range(1, MIGRATION_VERSION + 1)):
                return MIGRATION_VERSION
            if sorted(versions) not in ([], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]):
                raise LedgerError("MIGRATION_CONFLICT", "Unsupported conversation schema version")
            if not versions:
                conn.execute(_migration_sql(), prepare=False)
                conn.execute("INSERT INTO schema_migrations(version) VALUES (1)")
            if 2 not in versions:
                conn.execute(_history_migration_sql(), prepare=False)
                conn.execute("INSERT INTO schema_migrations(version) VALUES (2)")
            if 3 not in versions:
                conn.execute(_index_migration_sql(), prepare=False)
                conn.execute("INSERT INTO schema_migrations(version) VALUES (3)")
            if 4 not in versions:
                conn.execute(
                    files("story_rag.migrations")
                    .joinpath("004_effective_state.sql")
                    .read_text(encoding="utf-8"),
                    prepare=False,
                )
                conn.execute("INSERT INTO schema_migrations(version) VALUES (4)")
            conn.execute(
                files("story_rag.migrations")
                .joinpath("005_conversation_display.sql")
                .read_text(encoding="utf-8"),
                prepare=False,
            )
            conn.execute("INSERT INTO schema_migrations(version) VALUES (5)")
        return MIGRATION_VERSION

    @staticmethod
    def _session(
        conn: psycopg.Connection,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        lock: bool = False,
    ) -> dict[str, Any]:
        suffix = " FOR UPDATE" if lock else ""
        row = conn.execute(
            "SELECT * FROM conversation_sessions WHERE session_id = %s" + suffix,
            (session_id,),
        ).fetchone()
        if row is None:
            raise LedgerError("SESSION_NOT_FOUND", "Conversation does not exist")
        if (
            row["database_id"] != binding.database_id
            or row["storage_fingerprint"].strip() != binding.storage_fingerprint
        ):
            raise LedgerError("SCOPE_MISMATCH", "Conversation belongs to another database scope")
        return row

    @staticmethod
    def _session_record(row: dict[str, Any]) -> SessionRecord:
        return SessionRecord.model_validate(row)

    @staticmethod
    def _request_record(row: dict[str, Any]) -> RequestRecord:
        return RequestRecord.model_validate(row)

    def create_session(self, name: str, binding: DatabaseBinding) -> SessionRecord:
        normalized = unicodedata.normalize("NFC", name).strip()
        if not 1 <= len(normalized) <= 64 or any(
            unicodedata.category(char).startswith("C") for char in normalized
        ):
            raise LedgerError("INVALID_NAME", "Conversation name must be 1-64 visible characters")
        session_id = uuid.uuid4()
        with self._connect() as conn:
            row = conn.execute(
                "INSERT INTO conversation_sessions "
                "(session_id, name, database_id, storage_fingerprint) "
                "VALUES (%s, %s, %s, %s) RETURNING *",
                (session_id, normalized, binding.database_id, binding.storage_fingerprint),
            ).fetchone()
        return self._session_record(row)

    def get_session(
        self, session_id: UUID, binding: DatabaseBinding, *, include_archived: bool = False
    ) -> SessionRecord:
        with self._connect() as conn:
            row = self._session(conn, session_id, binding)
            if row["archived_at"] and not include_archived:
                raise LedgerError("SESSION_ARCHIVED", "Conversation is archived")
            return self._session_record(row)

    def list_sessions(
        self, binding: DatabaseBinding, *, include_archived: bool = False
    ) -> list[SessionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversation_sessions WHERE database_id = %s "
                "AND storage_fingerprint = %s "
                + ("" if include_archived else "AND archived_at IS NULL ")
                + "ORDER BY updated_at DESC, session_id",
                (binding.database_id, binding.storage_fingerprint),
            ).fetchall()
        return [self._session_record(row) for row in rows]

    def archive_session(
        self, session_id: UUID, binding: DatabaseBinding, *, expected_revision: int
    ) -> SessionRecord:
        with self._connect() as conn:
            row = self._session(conn, session_id, binding, lock=True)
            if row["archived_at"]:
                return self._session_record(row)
            if row["revision"] != expected_revision:
                raise LedgerError("REVISION_CONFLICT", "Conversation revision changed")
            updated = conn.execute(
                "UPDATE conversation_sessions SET archived_at = now(), revision = revision + 1, "
                "updated_at = now() WHERE session_id = %s RETURNING *",
                (session_id,),
            ).fetchone()
        return self._session_record(updated)

    def claim_request(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        request_id: str,
        payload: RequestPayload,
        expected_revision: int,
    ) -> RequestRecord:
        if not 1 <= len(request_id) <= 128 or expected_revision < 0:
            raise LedgerError("INVALID_REQUEST", "Invalid request ID or expected revision")
        digest = request_digest(payload, expected_revision)
        with self._connect() as conn:
            session = self._session(conn, session_id, binding, lock=True)
            existing = conn.execute(
                "SELECT * FROM turn_requests WHERE session_id = %s AND request_id = %s",
                (session_id, request_id),
            ).fetchone()
            if existing:
                if existing["request_digest"].strip() != digest:
                    raise LedgerError(
                        "IDEMPOTENCY_CONFLICT", "Request ID was used with other content"
                    )
                return self._request_record(existing)
            if session["archived_at"]:
                raise LedgerError("SESSION_ARCHIVED", "Conversation is archived")
            if session["revision"] != expected_revision:
                raise LedgerError("REVISION_CONFLICT", "Conversation revision changed")
            uncertain = conn.execute(
                "SELECT 1 FROM turn_requests WHERE session_id = %s AND status = 'unknown' LIMIT 1",
                (session_id,),
            ).fetchone()
            if uncertain:
                raise LedgerError(
                    "SESSION_UNCERTAIN", "Resolve the unknown request before new work"
                )
            created = conn.execute(
                "INSERT INTO turn_requests "
                "(session_id, turn_id, request_id, request_digest, expected_revision, status) "
                "VALUES (%s, %s, %s, %s, %s, 'queued') RETURNING *",
                (session_id, uuid.uuid4(), request_id, digest, expected_revision),
            ).fetchone()
            conn.execute(
                "INSERT INTO conversation_display_turns "
                "(session_id, turn_id, user_message, created_at) VALUES (%s, %s, %s, %s)",
                (session_id, created["turn_id"], payload.message, created["created_at"]),
            )
        return self._request_record(created)

    def get_request(
        self, session_id: UUID, binding: DatabaseBinding, request_id: str
    ) -> RequestRecord:
        with self._connect() as conn:
            self._session(conn, session_id, binding)
            row = conn.execute(
                "SELECT * FROM turn_requests WHERE session_id = %s AND request_id = %s",
                (session_id, request_id),
            ).fetchone()
        if row is None:
            raise LedgerError("REQUEST_NOT_FOUND", "Request does not exist")
        return self._request_record(row)

    def get_display_answer(
        self, session_id: UUID, binding: DatabaseBinding, turn_id: UUID
    ) -> str | None:
        with self._connect() as conn:
            self._session(conn, session_id, binding)
            row = conn.execute(
                "SELECT d.assistant_answer FROM conversation_display_turns d "
                "JOIN turn_requests t ON t.session_id = d.session_id AND t.turn_id = d.turn_id "
                "WHERE d.session_id = %s AND d.turn_id = %s AND t.status = 'succeeded'",
                (session_id, turn_id),
            ).fetchone()
        return row["assistant_answer"] if row else None

    def read_history_content(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        as_of_revision: int,
        before_turn_id: UUID,
        query: HistoryContentQuery,
    ) -> HistoryContentSnapshot | None:
        """Read original display data, never substitute the current question version."""
        with self._connect() as conn:
            session = self._session(conn, session_id, binding)
            if session["archived_at"]:
                raise LedgerError("SESSION_ARCHIVED", "Conversation is archived")
            if not 0 <= as_of_revision <= session["revision"]:
                raise LedgerError("REVISION_CONFLICT", "History revision is not available")
            boundary = conn.execute(
                "SELECT created_at, expected_revision FROM turn_requests "
                "WHERE session_id = %s AND turn_id = %s",
                (session_id, before_turn_id),
            ).fetchone()
            if boundary is None or boundary["expected_revision"] != as_of_revision:
                raise LedgerError("INVALID_HISTORY_REFERENCE", "Invalid history read boundary")
            conditions = [
                "t.session_id = %s", "t.turn_id <> %s", "t.created_at <= %s",
                "t.expected_revision <= %s",
                "(t.turn_sequence IS NULL OR t.turn_sequence <= %s)",
            ]
            params: list[Any] = [
                session_id, before_turn_id, boundary["created_at"],
                as_of_revision, as_of_revision,
            ]
            selected = None
            if query.selection == "question":
                filters = [*conditions, "q.question_no = %s", "t.status = 'succeeded'"]
                qparams = [*params, query.question_no]
                if query.question_version is not None:
                    filters.append("q.version = %s")
                    qparams.append(query.question_version)
                order = "ASC" if query.version_selection == "first" else "DESC"
                row = conn.execute(
                    "SELECT q.* FROM question_versions q JOIN turn_requests t "
                    "ON t.session_id = q.session_id AND t.turn_id = q.source_turn_id WHERE "
                    + " AND ".join(filters) + f" ORDER BY q.version {order} LIMIT 1",
                    qparams,
                ).fetchone()
                if row is None:
                    return None
                selected = QuestionVersionRecord.model_validate(row)
                conditions.append("t.turn_id = %s")
                params.append(selected.source_turn_id)
            elif query.selection == "turn":
                conditions.append("t.turn_sequence = %s")
                params.append(query.turn_sequence)
            order = "ASC" if query.selection == "first_turn" else "DESC"
            row = conn.execute(
                "SELECT t.session_id, t.turn_id, t.request_id, t.turn_sequence, t.status, "
                "t.result_refs, "
                "d.user_message, d.assistant_answer, t.created_at, "
                "t.usage->>'failure_code' AS failure_code, "
                "(t.status = 'succeeded' AND d.assistant_answer IS NULL) AS answer_not_retained "
                "FROM turn_requests t LEFT JOIN conversation_display_turns d "
                "ON d.session_id = t.session_id AND d.turn_id = t.turn_id WHERE "
                + " AND ".join(conditions)
                + f" ORDER BY t.created_at {order}, t.turn_id {order} LIMIT 1",
                params,
            ).fetchone()
            if row is None:
                return None
            refs = row.pop("result_refs")
            display_order = {
                (ref["question_id"], ref["version"]): index
                for index, ref in enumerate(refs)
            }
            questions = [
                QuestionVersionRecord.model_validate(value)
                for value in conn.execute(
                    "SELECT * FROM question_versions WHERE session_id = %s "
                    "AND source_turn_id = %s ORDER BY question_no",
                    (session_id, row["turn_id"]),
                )
            ]
            questions.sort(key=lambda q: display_order.get(
                (str(q.question_id), q.version), len(display_order)
            ))
            if query.question_order is not None:
                if query.question_order > len(questions):
                    return None
                selected = questions[query.question_order - 1]
            return HistoryContentSnapshot(
                turn=ConversationDisplayTurn.model_validate(row),
                questions=questions, selected_question=selected,
            )

    def list_display_turns(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        limit: int = 50,
        cursor: UUID | None = None,
    ) -> ConversationDisplayPage:
        if not 1 <= limit <= 100:
            raise LedgerError("INVALID_PAGE", "Page size must be 1-100")
        with self._connect() as conn:
            self._session(conn, session_id, binding)
            after = None
            if cursor is not None:
                after = conn.execute(
                    "SELECT created_at, turn_id FROM turn_requests "
                    "WHERE session_id = %s AND turn_id = %s",
                    (session_id, cursor),
                ).fetchone()
                if after is None:
                    raise LedgerError(
                        "INVALID_CURSOR", "Cursor does not belong to this conversation"
                    )
            rows = conn.execute(
                "SELECT t.session_id, t.turn_id, t.request_id, t.turn_sequence, t.status, "
                "d.user_message, d.assistant_answer, t.created_at, "
                "t.usage->>'failure_code' AS failure_code, "
                "(t.status = 'succeeded' AND d.assistant_answer IS NULL) AS answer_not_retained "
                "FROM turn_requests t LEFT JOIN conversation_display_turns d "
                "ON d.session_id = t.session_id AND d.turn_id = t.turn_id "
                "WHERE t.session_id = %s "
                + ("AND (t.created_at, t.turn_id) > (%s, %s) " if after else "")
                + "ORDER BY t.created_at, t.turn_id LIMIT %s",
                (session_id, after["created_at"], after["turn_id"], limit + 1)
                if after
                else (session_id, limit + 1),
            ).fetchall()
        more = len(rows) > limit
        return ConversationDisplayPage(
            items=[ConversationDisplayTurn.model_validate(row) for row in rows[:limit]],
            next_cursor=rows[limit - 1]["turn_id"] if more else None,
        )

    def start_request(
        self, session_id: UUID, binding: DatabaseBinding, request_id: str
    ) -> RequestRecord:
        with self._connect() as conn:
            session = self._session(conn, session_id, binding, lock=True)
            row = conn.execute(
                "SELECT * FROM turn_requests WHERE session_id = %s AND request_id = %s FOR UPDATE",
                (session_id, request_id),
            ).fetchone()
            if row is None:
                raise LedgerError("REQUEST_NOT_FOUND", "Request does not exist")
            if session["revision"] != row["expected_revision"]:
                raise LedgerError("REVISION_CONFLICT", "Conversation revision changed")
            if row["status"] == "queued":
                active = conn.execute(
                    "SELECT request_id, status FROM turn_requests WHERE session_id = %s "
                    "AND status IN ('running', 'unknown') AND request_id <> %s LIMIT 1",
                    (session_id, request_id),
                ).fetchone()
                if active:
                    code = "SESSION_UNCERTAIN" if active["status"] == "unknown" else "SESSION_BUSY"
                    raise LedgerError(code, "Another request blocks this conversation")
                row = conn.execute(
                    "UPDATE turn_requests SET status = 'running', execution_owner = %s, "
                    "updated_at = now() "
                    "WHERE session_id = %s AND request_id = %s RETURNING *",
                    (
                        Jsonb(
                            {
                                "pid": os.getpid(),
                                "created": psutil.Process().create_time(),
                                "host": socket.gethostname(),
                                "stage": "claimed",
                                "attempt": 0,
                            }
                        ),
                        session_id,
                        request_id,
                    ),
                ).fetchone()
            elif row["status"] == "running":
                raise LedgerError("REQUEST_IN_PROGRESS", "This request is already running")
            elif row["status"] != "succeeded":
                raise LedgerError("REQUEST_NOT_STARTABLE", "Request requires recovery")
        return self._request_record(row)

    def mark_model_attempt(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        request_id: str,
        stage: str,
    ) -> None:
        """Persist dispatch intent in a short transaction before a remote call."""
        with self._connect() as conn:
            self._session(conn, session_id, binding, lock=True)
            row = conn.execute(
                "SELECT * FROM turn_requests WHERE session_id=%s AND request_id=%s FOR UPDATE",
                (session_id, request_id),
            ).fetchone()
            owner = row.get("execution_owner") if row else None
            if (
                not row
                or row["status"] != "running"
                or not owner
                or owner.get("pid") != os.getpid()
                or owner.get("created") != psutil.Process().create_time()
            ):
                raise LedgerError("EXECUTION_OWNER_MISMATCH", "Request execution owner differs")
            owner = {**owner, "stage": stage, "attempt": owner.get("attempt", 0) + 1}
            from story_rag.pipelines.token_budget import TokenBudget

            budget = TokenBudget()
            reserve = (
                budget.planning_output
                if stage == "planning"
                else budget.gate_output
                if "gate" in stage
                else budget.generation_output
            )
            usage = UsageMetadata.model_validate(row["usage"])
            provisional = usage.model_copy(
                update={
                    "llm_calls": usage.llm_calls + 1,
                    "unknown_usage_calls": usage.unknown_usage_calls + 1,
                    "estimated_prompt_tokens": usage.estimated_prompt_tokens
                    + budget.input_limit(reserve),
                    "reserved_output_tokens": usage.reserved_output_tokens + reserve,
                    "accounted_tokens": usage.accounted_tokens
                    + budget.context_limit
                    - budget.safety_margin,
                }
            )
            conn.execute(
                "UPDATE turn_requests SET execution_owner=%s, usage=%s, updated_at=now() "
                "WHERE session_id=%s AND request_id=%s",
                (Jsonb(owner), Jsonb(provisional.model_dump(mode="json")), session_id, request_id),
            )

    def recover_orphaned_request(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        request_id: str,
    ) -> RequestRecord:
        """Only a verified dead local execution becomes unknown; never retry it."""
        with self._connect() as conn:
            self._session(conn, session_id, binding, lock=True)
            row = conn.execute(
                "SELECT * FROM turn_requests WHERE session_id=%s AND request_id=%s FOR UPDATE",
                (session_id, request_id),
            ).fetchone()
            if row is None:
                raise LedgerError("REQUEST_NOT_FOUND", "Request does not exist")
            if row["status"] in {"succeeded", "failed", "unknown"}:
                return self._request_record(row)
            owner = row.get("execution_owner")
            if row["status"] != "running" or not owner or owner.get("host") != socket.gethostname():
                raise LedgerError(
                    "OWNER_UNVERIFIABLE", "Cannot verify the original execution owner"
                )
            try:
                active = psutil.Process(owner["pid"]).create_time() == owner["created"]
            except psutil.NoSuchProcess:
                active = False
            except (psutil.AccessDenied, KeyError):
                raise LedgerError(
                    "OWNER_UNVERIFIABLE", "Cannot inspect the original execution"
                ) from None
            if active:
                raise LedgerError("REQUEST_IN_PROGRESS", "The original execution is still alive")
            usage = UsageMetadata.model_validate(row["usage"]).model_copy(
                update={
                    "failure_code": "ORPHANED_EXECUTION",
                    "outcome_unknown": True,
                }
            )
            updated = conn.execute(
                "UPDATE turn_requests SET status='unknown', usage=%s, updated_at=now() "
                "WHERE session_id=%s AND request_id=%s RETURNING *",
                (Jsonb(usage.model_dump(mode="json")), session_id, request_id),
            ).fetchone()
        return self._request_record(updated)

    def fail_request(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        request_id: str,
        *,
        failure_code: str,
        usage: UsageMetadata | None = None,
        outcome_unknown: bool = False,
    ) -> RequestRecord:
        if not 1 <= len(failure_code) <= 80:
            raise LedgerError("INVALID_REQUEST", "Failure code must be 1-80 characters")
        usage = usage or UsageMetadata()
        usage = usage.model_copy(
            update={"failure_code": failure_code, "outcome_unknown": outcome_unknown}
        )
        status = "unknown" if outcome_unknown else "failed"
        with self._connect() as conn:
            session = self._session(conn, session_id, binding, lock=True)
            row = conn.execute(
                "SELECT * FROM turn_requests WHERE session_id = %s AND request_id = %s FOR UPDATE",
                (session_id, request_id),
            ).fetchone()
            if row is None:
                raise LedgerError("REQUEST_NOT_FOUND", "Request does not exist")
            if row["status"] in {"succeeded", "failed", "unknown"}:
                return self._request_record(row)
            if session["revision"] != row["expected_revision"]:
                raise LedgerError("REVISION_CONFLICT", "Conversation revision changed")
            updated = conn.execute(
                "UPDATE turn_requests SET status = %s, usage = %s, updated_at = now() "
                "WHERE session_id = %s AND request_id = %s RETURNING *",
                (
                    status,
                    Jsonb(usage.model_dump(mode="json")),
                    session_id,
                    request_id,
                ),
            ).fetchone()
        return self._request_record(updated)

    def acknowledge_unknown_request(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        request_id: str,
        *,
        acknowledge_result_unknown: bool,
    ) -> RequestRecord:
        if not acknowledge_result_unknown:
            raise LedgerError("CONFIRMATION_REQUIRED", "Unknown outcome must be acknowledged")
        with self._connect() as conn:
            session = self._session(conn, session_id, binding, lock=True)
            row = conn.execute(
                "SELECT * FROM turn_requests WHERE session_id = %s AND request_id = %s FOR UPDATE",
                (session_id, request_id),
            ).fetchone()
            if row is None:
                raise LedgerError("REQUEST_NOT_FOUND", "Request does not exist")
            if row["status"] != "unknown":
                raise LedgerError("REQUEST_NOT_RECOVERABLE", "Request is not unknown")
            if session["revision"] != row["expected_revision"]:
                raise LedgerError("REVISION_CONFLICT", "Conversation revision changed")
            committed = conn.execute(
                "SELECT 1 FROM question_versions WHERE session_id = %s "
                "AND source_turn_id = %s LIMIT 1",
                (session_id, row["turn_id"]),
            ).fetchone()
            if committed:
                raise LedgerError("REQUEST_NOT_RECOVERABLE", "Request has committed versions")
            usage = dict(row["usage"])
            usage["failure_code"] = "UNKNOWN_ACKNOWLEDGED"
            usage["outcome_unknown"] = True
            updated = conn.execute(
                "UPDATE turn_requests SET status = 'failed', usage = %s, updated_at = now() "
                "WHERE session_id = %s AND request_id = %s RETURNING *",
                (Jsonb(usage), session_id, request_id),
            ).fetchone()
        return self._request_record(updated)

    @staticmethod
    def _version(
        conn: psycopg.Connection, table: str, session_id: UUID, item_id: UUID
    ) -> dict[str, Any] | None:
        column = "question_id" if table == "question_versions" else "object_id"
        return conn.execute(
            sql.SQL(
                "SELECT * FROM {} WHERE session_id = %s AND {} = %s ORDER BY version DESC LIMIT 1"
            ).format(sql.Identifier(table), sql.Identifier(column)),
            (session_id, item_id),
        ).fetchone()

    def _validate_objects(
        self,
        conn: psycopg.Connection,
        session_id: UUID,
        objects: list[ObjectVersionDraft],
        *,
        turn_id: UUID,
        message: str | None,
    ) -> None:
        if len({item.object_id for item in objects}) != len(objects):
            raise LedgerError("VERSION_CONFLICT", "Duplicate object in one request")
        for item in objects:
            previous = self._version(conn, "object_versions", session_id, item.object_id)
            if item.version != (previous["version"] + 1 if previous else 1):
                raise LedgerError("VERSION_CONFLICT", "Object version is not next in sequence")
            if item.derived_from and previous:
                raise LedgerError("INVALID_CHANGE", "Only a new object can be derived")
            base = previous
            if item.derived_from:
                if item.derived_from.object_id == item.object_id:
                    raise LedgerError("INVALID_CHANGE", "Object cannot derive from itself")
                base = conn.execute(
                    "SELECT * FROM object_versions WHERE session_id = %s "
                    "AND object_id = %s AND version = %s",
                    (session_id, item.derived_from.object_id, item.derived_from.version),
                ).fetchone()
                if base is None:
                    raise LedgerError("SCOPE_MISMATCH", "Derived object is not in this session")
            if not item.changes:
                if item.derived_from and item.conditions != base["conditions"]:
                    raise LedgerError("INVALID_CHANGE", "Unrecorded condition change")
                continue
            if message is None:
                raise LedgerError("SOURCE_REQUIRED", "Original request is required for changes")
            before = dict(base["conditions"]) if base else {}
            after = dict(before)
            fields: set[str] = set()
            for change in item.changes:
                if change.field in fields:
                    raise LedgerError("INVALID_CHANGE", "A field changes more than once")
                fields.add(change.field)
                if (
                    change.source_turn_id != turn_id
                    or change.source_excerpt not in message
                    or change.from_version != (base["version"] if base else 0)
                    or change.to_version != item.version
                ):
                    raise LedgerError("INVALID_SOURCE", "Change source does not match request")
                old = before.get(change.field)
                if change.op == "supplement" and change.field in before:
                    raise LedgerError("INVALID_CHANGE", "Supplement cannot overwrite a field")
                if change.op in {"replace", "remove"} and change.field not in before:
                    raise LedgerError("INVALID_CHANGE", "Replacement requires an old field")
                if type(old) is not type(change.before) or old != change.before:
                    raise LedgerError("VERSION_CONFLICT", "Old field value changed")
                if change.op == "remove":
                    del after[change.field]
                else:
                    after[change.field] = change.after
            if after != item.conditions:
                raise LedgerError("INVALID_CHANGE", "Unrecorded condition change")

    def _validate_questions(
        self,
        conn: psycopg.Connection,
        session_id: UUID,
        questions: list[QuestionVersionDraft],
        objects: list[ObjectVersionDraft],
    ) -> None:
        if len({item.question_id for item in questions}) != len(questions) or len(
            {item.question_no for item in questions}
        ) != len(questions):
            raise LedgerError("VERSION_CONFLICT", "Duplicate question in one request")
        fresh_objects = {(item.object_id, item.version) for item in objects}
        fresh_questions = {item.question_id for item in questions}
        for item in questions:
            previous = self._version(conn, "question_versions", session_id, item.question_id)
            if previous:
                if (
                    item.version != previous["version"] + 1
                    or item.question_no != previous["question_no"]
                ):
                    raise LedgerError("VERSION_CONFLICT", "Question identity or version changed")
            elif (
                item.version != 1
                or conn.execute(
                    "SELECT 1 FROM question_versions "
                    "WHERE session_id = %s AND question_no = %s LIMIT 1",
                    (session_id, item.question_no),
                ).fetchone()
            ):
                raise LedgerError("VERSION_CONFLICT", "Question number is already assigned")
            for ref in item.object_refs:
                if (ref.object_id, ref.version) not in fresh_objects and not conn.execute(
                    "SELECT 1 FROM object_versions WHERE session_id = %s "
                    "AND object_id = %s AND version = %s",
                    (session_id, ref.object_id, ref.version),
                ).fetchone():
                    raise LedgerError("SCOPE_MISMATCH", "Object reference is not in this session")
            if item.relation:
                ref = item.relation.target
                if not conn.execute(
                    "SELECT 1 FROM question_versions WHERE session_id = %s "
                    "AND question_id = %s AND version = %s",
                    (session_id, ref.question_id, ref.version),
                ).fetchone():
                    raise LedgerError("SCOPE_MISMATCH", "Question relation is not in this session")
            for dependency in item.dependencies:
                if dependency not in fresh_questions and not self._version(
                    conn, "question_versions", session_id, dependency
                ):
                    raise LedgerError(
                        "SCOPE_MISMATCH", "Question dependency is not in this session"
                    )

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
    ) -> RequestRecord:
        objects = objects or []
        usage = usage or UsageMetadata()
        with self._connect() as conn:
            session = self._session(conn, session_id, binding, lock=True)
            request = conn.execute(
                "SELECT * FROM turn_requests WHERE session_id = %s AND request_id = %s FOR UPDATE",
                (session_id, request_id),
            ).fetchone()
            if request is None:
                raise LedgerError("REQUEST_NOT_FOUND", "Request does not exist")
            if request["status"] == "succeeded":
                return self._request_record(request)
            if request["status"] not in {"queued", "running"}:
                raise LedgerError("REQUEST_NOT_COMMITTABLE", "Request requires recovery")
            if session["archived_at"]:
                raise LedgerError("SESSION_ARCHIVED", "Conversation is archived")
            if session["revision"] != request["expected_revision"]:
                raise LedgerError("REVISION_CONFLICT", "Conversation revision changed")
            if payload is not None and request_digest(payload, request["expected_revision"]) != (
                request["request_digest"].strip()
            ):
                raise LedgerError("IDEMPOTENCY_CONFLICT", "Original request does not match")
            if payload is not None and any(
                item.source_excerpt and item.source_excerpt not in payload.message
                for item in [*objects, *questions]
            ):
                raise LedgerError("INVALID_SOURCE", "Version source is not in request")
            if visible_answer is not None and payload is None:
                raise LedgerError("INVALID_REQUEST", "Visible answer requires the original request")
            self._validate_objects(
                conn,
                session_id,
                objects,
                turn_id=request["turn_id"],
                message=payload.message if payload else None,
            )
            self._validate_questions(conn, session_id, questions, objects)
            for item in objects:
                conn.execute(
                    "INSERT INTO object_versions (session_id, object_id, version, source_turn_id, "
                    "kind, conditions, changes, derived_from, source_excerpt) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        session_id,
                        item.object_id,
                        item.version,
                        request["turn_id"],
                        item.kind,
                        Jsonb(item.conditions),
                        Jsonb([change.model_dump(mode="json") for change in item.changes]),
                        Jsonb(item.derived_from.model_dump(mode="json"))
                        if item.derived_from
                        else None,
                        item.source_excerpt,
                    ),
                )
            for item in questions:
                conn.execute(
                    "INSERT INTO question_versions "
                    "(session_id, question_id, question_no, version, "
                    "source_turn_id, question_text, "
                    "requirements, status, relation, object_refs, dependencies, missing_inputs, "
                    "evidence_refs, answer_hash, source_excerpt, revision_kind, effective_state) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        session_id,
                        item.question_id,
                        item.question_no,
                        item.version,
                        request["turn_id"],
                        item.question_text,
                        Jsonb(item.requirements),
                        item.status,
                        Jsonb(item.relation.model_dump(mode="json")) if item.relation else None,
                        Jsonb([ref.model_dump(mode="json") for ref in item.object_refs]),
                        Jsonb([str(ref) for ref in item.dependencies]),
                        Jsonb(item.missing_inputs),
                        Jsonb([ref.model_dump(mode="json") for ref in item.evidence_refs]),
                        item.answer_hash,
                        item.source_excerpt,
                        item.revision_kind,
                        Jsonb(item.effective_state),
                    ),
                )
            next_revision = session["revision"] + 1
            refs = [
                QuestionRef(question_id=item.question_id, version=item.version)
                for item in questions
            ]
            updated = conn.execute(
                "UPDATE turn_requests SET status = 'succeeded', turn_sequence = %s, "
                "result_refs = %s, usage = %s, updated_at = now() "
                "WHERE session_id = %s AND request_id = %s RETURNING *",
                (
                    next_revision,
                    Jsonb([ref.model_dump(mode="json") for ref in refs]),
                    Jsonb(usage.model_dump(mode="json")),
                    session_id,
                    request_id,
                ),
            ).fetchone()
            conn.execute(
                "UPDATE conversation_sessions SET revision = %s, updated_at = now() "
                "WHERE session_id = %s",
                (next_revision, session_id),
            )
            conn.execute(
                "INSERT INTO history_index_outbox (session_id, turn_id, turn_sequence) "
                "VALUES (%s, %s, %s) ON CONFLICT (session_id, turn_id) DO NOTHING",
                (session_id, request["turn_id"], next_revision),
            )
            if visible_answer is not None:
                conn.execute(
                    "INSERT INTO conversation_display_turns "
                    "(session_id, turn_id, user_message, assistant_answer, created_at) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (session_id, turn_id) DO UPDATE "
                    "SET assistant_answer = EXCLUDED.assistant_answer",
                    (
                        session_id,
                        request["turn_id"],
                        payload.message,
                        visible_answer,
                        request["created_at"],
                    ),
                )
        return self._request_record(updated)

    def process_history_index(
        self, session_id: UUID, binding: DatabaseBinding, *, limit: int = 100
    ) -> int:
        if not 1 <= limit <= 1000:
            raise LedgerError("INVALID_REQUEST", "Index processing limit must be 1-1000")
        with self._connect() as conn:
            self._session(conn, session_id, binding, lock=True)
            pending = conn.execute(
                "SELECT turn_id FROM history_index_outbox "
                "WHERE session_id = %s AND processed_at IS NULL "
                "ORDER BY turn_sequence LIMIT %s FOR UPDATE SKIP LOCKED",
                (session_id, limit),
            ).fetchall()
            for row in pending:
                conn.execute(
                    "INSERT INTO history_question_index "
                    "(session_id, question_id, question_no, version, question_text, "
                    "status, source_turn_id) "
                    "SELECT session_id, question_id, question_no, version, question_text, "
                    "status, source_turn_id FROM question_versions "
                    "WHERE session_id = %s AND source_turn_id = %s "
                    "ON CONFLICT (session_id, question_id) DO UPDATE SET "
                    "question_no = EXCLUDED.question_no, version = EXCLUDED.version, "
                    "question_text = EXCLUDED.question_text, status = EXCLUDED.status, "
                    "source_turn_id = EXCLUDED.source_turn_id, updated_at = now() "
                    "WHERE history_question_index.version < EXCLUDED.version",
                    (session_id, row["turn_id"]),
                )
                conn.execute(
                    "UPDATE history_index_outbox SET processed_at = now() "
                    "WHERE session_id = %s AND turn_id = %s AND processed_at IS NULL",
                    (session_id, row["turn_id"]),
                )
        return len(pending)

    def pending_history_index(self, session_id: UUID, binding: DatabaseBinding) -> int:
        with self._connect() as conn:
            self._session(conn, session_id, binding)
            row = conn.execute(
                "SELECT count(*) AS pending FROM history_index_outbox "
                "WHERE session_id = %s AND processed_at IS NULL",
                (session_id,),
            ).fetchone()
        return row["pending"]

    def list_question_versions(
        self, session_id: UUID, binding: DatabaseBinding, question_id: UUID | None = None
    ) -> list[QuestionVersionRecord]:
        with self._connect() as conn:
            self._session(conn, session_id, binding)
            rows = conn.execute(
                "SELECT * FROM question_versions WHERE session_id = %s "
                + ("AND question_id = %s " if question_id else "")
                + "ORDER BY question_no, version",
                (session_id, question_id) if question_id else (session_id,),
            ).fetchall()
        return [QuestionVersionRecord.model_validate(row) for row in rows]

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
    ) -> list[QuestionVersionRecord]:
        if (
            as_of_revision < 0
            or (question_no is not None and question_no < 1)
            or (turn_sequence is not None and turn_sequence < 1)
            or not any(
                (
                    question_no is not None,
                    turn_sequence is not None,
                    question_id is not None,
                    pending_only,
                )
            )
        ):
            raise LedgerError("INVALID_LOOKUP", "A bounded history selector is required")
        with self._connect() as conn:
            session = self._session(conn, session_id, binding)
            if session["archived_at"]:
                raise LedgerError("SESSION_ARCHIVED", "Conversation is archived")
            if as_of_revision > session["revision"]:
                raise LedgerError("REVISION_CONFLICT", "History revision is not available")
            ids: list[UUID] | None = None
            if turn_sequence is not None:
                matches = conn.execute(
                    "SELECT q.question_id FROM question_versions q "
                    "JOIN turn_requests t ON t.session_id = q.session_id "
                    "AND t.turn_id = q.source_turn_id "
                    "WHERE q.session_id = %s AND t.turn_sequence = %s "
                    "AND t.status = 'succeeded' AND t.turn_sequence <= %s "
                    + ("AND q.question_no = %s " if question_no is not None else "")
                    + ("AND q.question_id = %s " if question_id is not None else ""),
                    (session_id, turn_sequence, as_of_revision)
                    + ((question_no,) if question_no is not None else ())
                    + ((question_id,) if question_id is not None else ()),
                ).fetchall()
                ids = [row["question_id"] for row in matches]
                if not ids:
                    return []
            filters = ["q.session_id = %s", "t.status = 'succeeded'", "t.turn_sequence <= %s"]
            params: list[Any] = [session_id, as_of_revision]
            if ids is not None:
                filters.append("q.question_id = ANY(%s)")
                params.append(ids)
            if question_no is not None:
                filters.append("q.question_no = %s")
                params.append(question_no)
            if question_id is not None:
                filters.append("q.question_id = %s")
                params.append(question_id)
            rows = conn.execute(
                "SELECT * FROM (SELECT DISTINCT ON (q.question_id) q.* "
                "FROM question_versions q JOIN turn_requests t "
                "ON t.session_id = q.session_id AND t.turn_id = q.source_turn_id "
                "WHERE "
                + " AND ".join(filters)
                + " ORDER BY q.question_id, q.version DESC) latest "
                + ("WHERE latest.status = 'waiting_input' " if pending_only else "")
                + "ORDER BY latest.question_no",
                params,
            ).fetchall()
        return [QuestionVersionRecord.model_validate(row) for row in rows]

    def current_questions_at(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        as_of_revision: int,
    ) -> list[QuestionVersionRecord]:
        with self._connect() as conn:
            session = self._session(conn, session_id, binding)
            if session["archived_at"]:
                raise LedgerError("SESSION_ARCHIVED", "Conversation is archived")
            if as_of_revision < 0 or as_of_revision > session["revision"]:
                raise LedgerError("REVISION_CONFLICT", "History revision is not available")
            rows = conn.execute(
                "SELECT * FROM (SELECT DISTINCT ON (q.question_id) q.* "
                "FROM question_versions q JOIN turn_requests t "
                "ON t.session_id = q.session_id AND t.turn_id = q.source_turn_id "
                "WHERE q.session_id = %s AND t.status = 'succeeded' "
                "AND t.turn_sequence <= %s ORDER BY q.question_id, q.version DESC) latest "
                "ORDER BY question_no",
                (session_id, as_of_revision),
            ).fetchall()
        return [QuestionVersionRecord.model_validate(row) for row in rows]

    def read_objects_at(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        as_of_revision: int,
        refs: list[ObjectRef],
    ) -> list[ObjectVersionRecord]:
        with self._connect() as conn:
            session = self._session(conn, session_id, binding)
            if session["archived_at"]:
                raise LedgerError("SESSION_ARCHIVED", "Conversation is archived")
            if as_of_revision < 0 or as_of_revision > session["revision"]:
                raise LedgerError("REVISION_CONFLICT", "History revision is not available")
            rows = []
            for ref in refs:
                row = conn.execute(
                    "SELECT o.* FROM object_versions o JOIN turn_requests t "
                    "ON t.session_id = o.session_id AND t.turn_id = o.source_turn_id "
                    "WHERE o.session_id = %s AND o.object_id = %s AND o.version = %s "
                    "AND t.status = 'succeeded' AND t.turn_sequence <= %s",
                    (session_id, ref.object_id, ref.version, as_of_revision),
                ).fetchone()
                if row is None:
                    raise LedgerError("SCOPE_MISMATCH", "Object version is not visible")
                rows.append(row)
        return [ObjectVersionRecord.model_validate(row) for row in rows]

    def next_question_no(
        self, session_id: UUID, binding: DatabaseBinding, *, as_of_revision: int
    ) -> int:
        with self._connect() as conn:
            session = self._session(conn, session_id, binding)
            if session["archived_at"]:
                raise LedgerError("SESSION_ARCHIVED", "Conversation is archived")
            if as_of_revision < 0 or as_of_revision > session["revision"]:
                raise LedgerError("REVISION_CONFLICT", "History revision is not available")
            row = conn.execute(
                "SELECT coalesce(max(q.question_no), 0) AS maximum "
                "FROM question_versions q JOIN turn_requests t "
                "ON t.session_id = q.session_id AND t.turn_id = q.source_turn_id "
                "WHERE q.session_id = %s AND t.status = 'succeeded' "
                "AND t.turn_sequence <= %s",
                (session_id, as_of_revision),
            ).fetchone()
        return row["maximum"] + 1

    def current_object_at(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        as_of_revision: int,
        object_id: UUID,
    ) -> ObjectVersionRecord:
        with self._connect() as conn:
            session = self._session(conn, session_id, binding)
            if session["archived_at"]:
                raise LedgerError("SESSION_ARCHIVED", "Conversation is archived")
            if as_of_revision < 0 or as_of_revision > session["revision"]:
                raise LedgerError("REVISION_CONFLICT", "History revision is not available")
            row = conn.execute(
                "SELECT o.* FROM object_versions o JOIN turn_requests t "
                "ON t.session_id = o.session_id AND t.turn_id = o.source_turn_id "
                "WHERE o.session_id = %s AND o.object_id = %s "
                "AND t.status = 'succeeded' AND t.turn_sequence <= %s "
                "ORDER BY o.version DESC LIMIT 1",
                (session_id, object_id, as_of_revision),
            ).fetchone()
        if row is None:
            raise LedgerError("SCOPE_MISMATCH", "Object is not visible")
        return ObjectVersionRecord.model_validate(row)

    def list_object_versions(
        self, session_id: UUID, binding: DatabaseBinding
    ) -> list[ObjectVersionRecord]:
        with self._connect() as conn:
            self._session(conn, session_id, binding)
            rows = conn.execute(
                "SELECT * FROM object_versions WHERE session_id = %s ORDER BY object_id, version",
                (session_id,),
            ).fetchall()
        return [ObjectVersionRecord.model_validate(row) for row in rows]

    def inspect_scope(self, session_id: UUID, binding: DatabaseBinding) -> dict[str, Any]:
        with self._connect() as conn:
            session = self._session(conn, session_id, binding)
            counts = {}
            for table in ("turn_requests", "question_versions", "object_versions"):
                counts[table] = conn.execute(
                    sql.SQL("SELECT count(*) AS n FROM {} WHERE session_id = %s").format(
                        sql.Identifier(table)
                    ),
                    (session_id,),
                ).fetchone()["n"]
        return {"session": self._session_record(session), "counts": counts}

    def inspect_display_scope(self, session_id: UUID, binding: DatabaseBinding) -> dict[str, int]:
        with self._connect() as conn:
            self._session(conn, session_id, binding)
            row = conn.execute(
                "SELECT count(*) AS stored_turns, "
                "count(*) FILTER (WHERE assistant_answer IS NOT NULL) AS stored_answers "
                "FROM conversation_display_turns WHERE session_id = %s",
                (session_id,),
            ).fetchone()
        return {"stored_turns": row["stored_turns"], "stored_answers": row["stored_answers"]}

    def inspect_versions(
        self, session_id: UUID, binding: DatabaseBinding
    ) -> dict[str, list[QuestionVersionRecord] | list[ObjectVersionRecord]]:
        return {
            "questions": self.list_question_versions(session_id, binding),
            "objects": self.list_object_versions(session_id, binding),
        }
