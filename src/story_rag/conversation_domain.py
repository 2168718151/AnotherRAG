from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


def literal_in_source(value: str | int | float | bool, source: str) -> bool:
    """Check literal provenance; do not infer units or take digits from a dtype."""
    literal = str(value).casefold()
    number = r"[+−-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?(?:e[+−-]?[0-9]+)?"
    if re.fullmatch(number, literal):
        return any(
            match.group().casefold() == literal
            for match in re.finditer(
                r"(?<![A-Za-z0-9_,.+−-])" + number + r"(?![0-9_]|[.,][0-9]|e[+−-]?[0-9])",
                source,
                re.I,
            )
        )
    return literal in source.casefold()


class LedgerError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class LedgerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatabaseBinding(LedgerModel):
    database_id: UUID
    storage_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class SessionRecord(DatabaseBinding):
    session_id: UUID
    name: str
    revision: int = Field(ge=0)
    archived_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class RequestPayload(LedgerModel):
    message: str = Field(min_length=1)
    reply_to_question_id: UUID | None = None
    use_history: bool = True

    @model_validator(mode="after")
    def validate_history_reference(self) -> RequestPayload:
        if not self.use_history and self.reply_to_question_id is not None:
            raise ValueError("Explicit history reference requires use_history=true")
        return self


class ConversationTurnInput(LedgerModel):
    request_id: str = Field(min_length=1, max_length=128)
    expected_revision: int = Field(ge=0)
    message: str = Field(min_length=1)
    reply_to_question_id: UUID | None = None
    use_history: bool = True

    @model_validator(mode="after")
    def validate_history_reference(self) -> ConversationTurnInput:
        if not self.use_history and self.reply_to_question_id is not None:
            raise ValueError("Explicit history reference requires use_history=true")
        return self


def request_digest(payload: RequestPayload, expected_revision: int) -> str:
    canonical = json.dumps(
        {
            "schema_version": 1,
            "expected_revision": expected_revision,
            **payload.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class QuestionRef(LedgerModel):
    question_id: UUID
    version: int = Field(ge=1)


class ObjectRef(LedgerModel):
    object_id: UUID
    version: int = Field(ge=1)


class EvidenceRef(LedgerModel):
    document_id: str = Field(min_length=1, max_length=128)
    node_id: str = Field(min_length=1, max_length=128)
    document_version: str | None = Field(default=None, max_length=128)
    node_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class QuestionRelation(LedgerModel):
    kind: Literal[
        "follow_up",
        "supplement",
        "replace",
        "extend",
        "compare",
        "format",
        "cancel",
        "merge",
    ]
    target: QuestionRef


class UsageMetadata(LedgerModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    elapsed_ms: float = Field(default=0.0, ge=0)
    estimated_prompt_tokens: int = Field(default=0, ge=0)
    reserved_output_tokens: int = Field(default=0, ge=0)
    unknown_usage_calls: int = Field(default=0, ge=0)
    settled_tokens: int = Field(default=0, ge=0)
    accounted_tokens: int = Field(default=0, ge=0)
    failure_code: str | None = Field(default=None, max_length=80)
    outcome_unknown: bool = False


class ConditionChange(LedgerModel):
    field: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    op: Literal["supplement", "replace", "remove"]
    before: str | int | float | bool | None = None
    after: str | int | float | bool | None
    unit: str | None = Field(default=None, max_length=32)
    source_turn_id: UUID
    source_excerpt: str = Field(min_length=1, max_length=512)
    from_version: int = Field(ge=0)
    to_version: int = Field(ge=1)

    @model_validator(mode="after")
    def removal_value(self) -> ConditionChange:
        if (self.after is None) != (self.op == "remove"):
            raise ValueError("Only removal has a null after value")
        return self


class ObjectVersionDraft(LedgerModel):
    object_id: UUID
    version: int = Field(ge=1)
    kind: str = Field(min_length=1, max_length=80)
    conditions: dict[str, str | int | float | bool] = Field(default_factory=dict)
    changes: list[ConditionChange] = Field(default_factory=list)
    derived_from: ObjectRef | None = None
    source_excerpt: str | None = Field(default=None, max_length=512)


class QuestionVersionDraft(LedgerModel):
    question_id: UUID
    question_no: int = Field(ge=1)
    version: int = Field(ge=1)
    question_text: str = Field(min_length=1, max_length=4000)
    requirements: list[str] = Field(default_factory=list)
    effective_state: dict[str, Any] = Field(default_factory=dict)
    status: Literal[
        "ready",
        "waiting_input",
        "answered",
        "partial",
        "no_evidence",
        "needs_refresh",
        "cancelled",
        "technical_failure",
    ] = "ready"
    relation: QuestionRelation | None = None
    object_refs: list[ObjectRef] = Field(default_factory=list)
    dependencies: list[UUID] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    answer_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_excerpt: str | None = Field(default=None, max_length=512)
    revision_kind: (
        Literal[
            "supplement",
            "replace",
            "extend",
            "compare",
            "format",
            "cancel",
            "new",
        ]
        | None
    ) = None


class RequestRecord(LedgerModel):
    session_id: UUID
    turn_id: UUID
    request_id: str
    request_digest: str
    expected_revision: int = Field(ge=0)
    turn_sequence: int | None = None
    status: Literal["queued", "running", "succeeded", "failed", "unknown"]
    result_refs: list[QuestionRef] = Field(default_factory=list)
    usage: UsageMetadata = Field(default_factory=UsageMetadata)
    execution_owner: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class ConversationDisplayTurn(LedgerModel):
    session_id: UUID
    turn_id: UUID
    request_id: str
    turn_sequence: int | None = None
    status: Literal["queued", "running", "succeeded", "failed", "unknown"]
    user_message: str | None = None
    assistant_answer: str | None = None
    failure_code: str | None = None
    answer_not_retained: bool = False
    created_at: datetime


class ConversationDisplayPage(LedgerModel):
    items: list[ConversationDisplayTurn]
    next_cursor: UUID | None = None


class QuestionVersionRecord(QuestionVersionDraft):
    session_id: UUID
    source_turn_id: UUID
    created_at: datetime


class ObjectVersionRecord(ObjectVersionDraft):
    session_id: UUID
    source_turn_id: UUID
    created_at: datetime


class HistoryContentQuery(LedgerModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation: Literal["read_user", "read_answer", "transform", "followup"]
    selection: Literal["first_turn", "previous_turn", "turn", "question"] = "previous_turn"
    turn_sequence: int | None = Field(default=None, ge=1)
    question_no: int | None = Field(default=None, ge=1)
    question_order: int | None = Field(default=None, ge=1)
    question_version: int | None = Field(default=None, ge=1)
    version_selection: Literal["first", "latest"] = "first"

    @model_validator(mode="after")
    def validate_selector(self) -> HistoryContentQuery:
        if (self.selection == "turn") != (self.turn_sequence is not None):
            raise ValueError("turn selection requires exactly one turn_sequence")
        if (self.selection == "question") != (self.question_no is not None):
            raise ValueError("question selection requires exactly one question_no")
        if self.question_version is not None and self.selection != "question":
            raise ValueError("question_version requires question selection")
        if self.question_order is not None and self.selection == "question":
            raise ValueError("question_order is local to a selected turn")
        return self


class HistoryContentSnapshot(LedgerModel):
    turn: ConversationDisplayTurn
    questions: list[QuestionVersionRecord] = Field(default_factory=list)
    selected_question: QuestionVersionRecord | None = None


class HistoryContentRef(LedgerModel):
    turn_id: UUID
    turn_sequence: int | None = None
    question_id: UUID | None = None
    question_version: int | None = None
    role: Literal["user", "assistant"]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
