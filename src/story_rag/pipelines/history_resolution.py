from __future__ import annotations

from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from story_rag.conversation_domain import (
    ConditionChange,
    DatabaseBinding,
    ObjectRef,
    ObjectVersionDraft,
    ObjectVersionRecord,
    QuestionRef,
    QuestionRelation,
    QuestionVersionDraft,
    QuestionVersionRecord,
    RequestPayload,
    RequestRecord,
    literal_in_source,
    request_digest,
)


class HistoryResolutionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class HistoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HistoryScope(HistoryModel):
    session_id: UUID
    binding: DatabaseBinding
    as_of_revision: int = Field(ge=0)
    use_history: bool = True


class HistoryLookup(HistoryModel):
    question_no: int | None = Field(default=None, ge=1)
    turn_sequence: int | None = Field(default=None, ge=1)
    question_id: UUID | None = None
    pending_only: bool = False

    @model_validator(mode="after")
    def bounded(self) -> HistoryLookup:
        if not any(
            (
                self.question_no is not None,
                self.turn_sequence is not None,
                self.question_id is not None,
                self.pending_only,
            )
        ):
            raise ValueError("A bounded history selector is required")
        return self


class HistoryResolution(HistoryModel):
    decision: Literal["matched", "ambiguous", "not_found"]
    selected: QuestionVersionRecord | None = None
    objects: list[ObjectVersionRecord] = Field(default_factory=list)
    candidates: list[QuestionRef] = Field(default_factory=list)


class ConditionPatch(HistoryModel):
    field: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    op: Literal["supplement", "replace", "remove"]
    value: str | int | float | bool | None
    unit: str | None = Field(default=None, max_length=32)
    source_excerpt: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def removal_value(self) -> ConditionPatch:
        if (self.value is None) != (self.op == "remove"):
            raise ValueError("Only removal has a null value")
        return self


class GapResolution(HistoryModel):
    missing_input: str
    source_excerpt: str = Field(min_length=1, max_length=512)


class RevisionSpec(HistoryModel):
    task_key: str = Field(min_length=1, max_length=32)
    kind: Literal[
        "supplement",
        "replace",
        "extend",
        "compare",
        "format",
        "cancel",
        "new",
        "ambiguous",
        "not_found",
    ]
    source_excerpt: str = Field(min_length=1, max_length=512)
    question_text: str = Field(min_length=1, max_length=4000)
    lookup: HistoryLookup | None = None
    target_object_id: UUID | None = None
    new_object_kind: str | None = Field(default=None, max_length=80)
    patches: list[ConditionPatch] = Field(default_factory=list)
    resolved_gaps: list[GapResolution] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    requirements_complete: bool = False
    effective_state: dict[str, Any] = Field(default_factory=dict)
    object_alias: str | None = None


class RevisionOutcome(HistoryModel):
    task_key: str
    decision: Literal["prepared", "ambiguous", "not_found"]
    question_ref: QuestionRef | None = None


class RevisionBatch(HistoryModel):
    questions: list[QuestionVersionDraft] = Field(default_factory=list)
    objects: list[ObjectVersionDraft] = Field(default_factory=list)
    outcomes: list[RevisionOutcome] = Field(default_factory=list)


class ScopedHistoryStore(Protocol):
    def current_questions_at(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        as_of_revision: int,
    ) -> list[QuestionVersionRecord]: ...

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

    def next_question_no(
        self, session_id: UUID, binding: DatabaseBinding, *, as_of_revision: int
    ) -> int: ...

    def current_object_at(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        as_of_revision: int,
        object_id: UUID,
    ) -> ObjectVersionRecord: ...


class HistoryResolutionPipeline:
    """Resolve only verified ledger identities, then stage append-only revisions."""

    def __init__(self, store: ScopedHistoryStore) -> None:
        self._store = store

    def resolve(self, scope: HistoryScope, lookup: HistoryLookup) -> HistoryResolution:
        if not scope.use_history:
            raise HistoryResolutionError("HISTORY_DISABLED", "History access is disabled")
        candidates = self._store.read_questions_at(
            scope.session_id,
            scope.binding,
            as_of_revision=scope.as_of_revision,
            question_no=lookup.question_no,
            turn_sequence=lookup.turn_sequence,
            question_id=lookup.question_id,
            pending_only=lookup.pending_only,
        )
        if not candidates:
            return HistoryResolution(decision="not_found")
        if len(candidates) != 1:
            return HistoryResolution(
                decision="ambiguous",
                candidates=[
                    QuestionRef(question_id=q.question_id, version=q.version) for q in candidates
                ],
            )
        question = candidates[0]
        objects = self._store.read_objects_at(
            scope.session_id,
            scope.binding,
            as_of_revision=scope.as_of_revision,
            refs=question.object_refs,
        )
        return HistoryResolution(decision="matched", selected=question, objects=objects)

    def current_questions(self, scope: HistoryScope) -> list[QuestionVersionRecord]:
        if not scope.use_history:
            raise HistoryResolutionError("HISTORY_DISABLED", "History access is disabled")
        return self._store.current_questions_at(
            scope.session_id,
            scope.binding,
            as_of_revision=scope.as_of_revision,
        )

    def next_question_no(self, scope: HistoryScope) -> int:
        return self._store.next_question_no(
            scope.session_id,
            scope.binding,
            as_of_revision=scope.as_of_revision,
        )

    def objects_for(self, scope: HistoryScope, refs: list[ObjectRef]) -> list[ObjectVersionRecord]:
        if not scope.use_history:
            raise HistoryResolutionError("HISTORY_DISABLED", "History access is disabled")
        if not refs:
            return []
        return self._store.read_objects_at(
            scope.session_id,
            scope.binding,
            as_of_revision=scope.as_of_revision,
            refs=refs,
        )

    def prepare_batch(
        self,
        scope: HistoryScope,
        request: RequestRecord,
        payload: RequestPayload,
        specs: list[RevisionSpec],
    ) -> RevisionBatch:
        if (
            request.session_id != scope.session_id
            or request.expected_revision != scope.as_of_revision
            or payload.use_history != scope.use_history
            or request.request_digest != request_digest(payload, scope.as_of_revision)
            or request.status not in {"queued", "running"}
        ):
            raise HistoryResolutionError("REQUEST_MISMATCH", "Request and scope do not match")
        if not scope.use_history and any(
            spec.kind not in {"new", "ambiguous", "not_found"} for spec in specs
        ):
            raise HistoryResolutionError(
                "HISTORY_DISABLED", "History-dependent revisions are disabled"
            )
        if len({s.task_key for s in specs}) != len(specs):
            raise HistoryResolutionError("DUPLICATE_TASK", "Task keys must be unique")
        next_no: int | None = None
        edited: set[UUID] = set()
        updated_objects: dict[UUID, ObjectRef] = {}
        questions: list[QuestionVersionDraft] = []
        objects: list[ObjectVersionDraft] = []
        new_aliases: dict[str, ObjectRef] = {}
        outcomes: list[RevisionOutcome] = []
        for spec in specs:
            self._source(payload.message, spec.source_excerpt)
            if spec.kind in {"ambiguous", "not_found"}:
                outcomes.append(RevisionOutcome(task_key=spec.task_key, decision=spec.kind))
                continue
            if spec.kind == "new":
                if spec.lookup is not None or spec.resolved_gaps:
                    raise HistoryResolutionError("INVALID_REVISION", "New task cannot use history")
                resolution = None
            else:
                if spec.lookup is None:
                    raise HistoryResolutionError("INVALID_REVISION", "History lookup is required")
                resolution = self.resolve(scope, spec.lookup)
                if resolution.decision != "matched":
                    outcomes.append(
                        RevisionOutcome(
                            task_key=spec.task_key,
                            decision=resolution.decision,
                        )
                    )
                    continue
            previous = resolution.selected if resolution else None
            branching = spec.kind in {"new", "extend", "compare"}
            if previous and not branching and previous.question_id in edited:
                raise HistoryResolutionError("DUPLICATE_TARGET", "A question changes twice")
            if previous and not branching:
                edited.add(previous.question_id)
            if spec.kind in {"format", "cancel"} and (spec.patches or spec.resolved_gaps):
                raise HistoryResolutionError(
                    "INVALID_REVISION", "This revision cannot change facts"
                )
            if spec.kind == "new" and spec.patches and not spec.new_object_kind:
                raise HistoryResolutionError("INVALID_REVISION", "New object kind is required")
            if next_no is None and spec.kind in {"new", "extend", "compare"}:
                next_no = self._store.next_question_no(
                    scope.session_id, scope.binding, as_of_revision=scope.as_of_revision
                )
            question_id = uuid4() if branching else previous.question_id
            question_no = next_no if branching else previous.question_no
            if branching:
                next_no += 1
            version = 1 if branching else previous.version + 1
            object_refs = list(previous.object_refs) if previous else []
            object_refs = [updated_objects.get(ref.object_id, ref) for ref in object_refs]
            if spec.kind == "new" and spec.object_alias in new_aliases:
                object_refs = [new_aliases[spec.object_alias]]
            create_empty = bool(
                spec.new_object_kind and not object_refs
                and spec.object_alias not in new_aliases
            )
            if spec.patches or create_empty:
                if resolution and any(
                    record.object_id in updated_objects
                    for record in resolution.objects
                    if spec.target_object_id in (None, record.object_id)
                ):
                    raise HistoryResolutionError(
                        "DUPLICATE_OBJECT", "An object changes twice in one request"
                    )
                if resolution:
                    matched_objects = [
                        record
                        for record in resolution.objects
                        if spec.target_object_id in (None, record.object_id)
                    ]
                    if len(matched_objects) == 1:
                        current = self._store.current_object_at(
                            scope.session_id,
                            scope.binding,
                            as_of_revision=scope.as_of_revision,
                            object_id=matched_objects[0].object_id,
                        )
                        if current.version != matched_objects[0].version:
                            raise HistoryResolutionError(
                                "STALE_OBJECT_REF", "Question refers to an old object version"
                            )
                draft, object_refs = self._prepare_object(
                    spec,
                    resolution.objects if resolution else [],
                    object_refs,
                    request.turn_id,
                    payload.message,
                    branch=branching,
                )
                objects.append(draft)
                if spec.kind == "new" and spec.object_alias:
                    new_aliases[spec.object_alias] = ObjectRef(
                        object_id=draft.object_id, version=draft.version
                    )
                if not branching:
                    updated_objects[draft.object_id] = ObjectRef(
                        object_id=draft.object_id, version=draft.version
                    )
            missing = list(previous.missing_inputs) if previous else []
            for gap in spec.resolved_gaps:
                self._source(payload.message, gap.source_excerpt)
                if gap.missing_input not in missing:
                    raise HistoryResolutionError("INVALID_GAP", "Unknown missing input")
                missing.remove(gap.missing_input)
            carry_requirements = (
                previous.requirements
                if previous
                and spec.kind in {"supplement", "extend", "compare", "format"}
                and not spec.resolved_gaps
                and not spec.requirements_complete
                else []
            )
            requirements = list(
                dict.fromkeys(
                    [
                        *carry_requirements,
                        *spec.requirements,
                    ]
                )
            )
            question = QuestionVersionDraft(
                question_id=question_id,
                question_no=question_no,
                version=version,
                question_text=spec.question_text,
                requirements=requirements,
                effective_state=spec.effective_state,
                status=(
                    "cancelled"
                    if spec.kind == "cancel"
                    else "waiting_input"
                    if missing
                    else "ready"
                ),
                relation=(
                    QuestionRelation(
                        kind=spec.kind,
                        target=QuestionRef(
                            question_id=previous.question_id,
                            version=previous.version,
                        ),
                    )
                    if previous
                    else None
                ),
                object_refs=object_refs,
                missing_inputs=missing,
                source_excerpt=spec.source_excerpt,
                revision_kind=spec.kind,
            )
            questions.append(question)
            outcomes.append(
                RevisionOutcome(
                    task_key=spec.task_key,
                    decision="prepared",
                    question_ref=QuestionRef(question_id=question_id, version=version),
                )
            )
        for question in questions:
            if question.revision_kind != "compare":
                question.object_refs = [
                    updated_objects.get(ref.object_id, ref) for ref in question.object_refs
                ]
        return RevisionBatch(questions=questions, objects=objects, outcomes=outcomes)

    @staticmethod
    def _source(message: str, excerpt: str) -> None:
        if excerpt not in message:
            raise HistoryResolutionError("INVALID_SOURCE", "Source excerpt is not in request")

    def _prepare_object(
        self,
        spec: RevisionSpec,
        records: list[ObjectVersionRecord],
        refs: list[ObjectRef],
        turn_id: UUID,
        message: str,
        *,
        branch: bool,
    ) -> tuple[ObjectVersionDraft, list[ObjectRef]]:
        if spec.kind in {"new"} and records:
            raise HistoryResolutionError("INVALID_REVISION", "New task has old objects")
        if spec.kind != "new" and records:
            matches = [r for r in records if spec.target_object_id in (None, r.object_id)]
            if len(matches) != 1:
                raise HistoryResolutionError("AMBIGUOUS_OBJECT", "Select one verified object")
            base = matches[0]
            before = dict(base.conditions)
            derived_from = (
                ObjectRef(object_id=base.object_id, version=base.version) if branch else None
            )
            object_id = uuid4() if branch else base.object_id
            version = 1 if branch else base.version + 1
            from_version = base.version
            kind = base.kind
        else:
            if not spec.new_object_kind:
                raise HistoryResolutionError("INVALID_REVISION", "A new object requires a kind")
            before = {}
            derived_from = None
            object_id = uuid4()
            version = 1
            from_version = 0
            kind = spec.new_object_kind
        after = dict(before)
        changes: list[ConditionChange] = []
        for patch in spec.patches:
            self._source(message, patch.source_excerpt)
            if (
                patch.op != "remove"
                and not literal_in_source(patch.value, patch.source_excerpt)
            ):
                raise HistoryResolutionError("UNVERIFIED_VALUE", "Value is not in source excerpt")
            if patch.unit and patch.unit.casefold() not in patch.source_excerpt.casefold():
                raise HistoryResolutionError("UNVERIFIED_VALUE", "Unit is not in source excerpt")
            exists = patch.field in after
            if (patch.op in {"replace", "remove"}) != exists:
                raise HistoryResolutionError(
                    "INVALID_CHANGE", "Patch operation does not match field"
                )
            changes.append(
                ConditionChange(
                    field=patch.field,
                    op=patch.op,
                    before=after.get(patch.field),
                    after=patch.value,
                    unit=patch.unit,
                    source_turn_id=turn_id,
                    source_excerpt=patch.source_excerpt,
                    from_version=from_version,
                    to_version=version,
                )
            )
            if patch.op == "remove":
                del after[patch.field]
            else:
                after[patch.field] = patch.value
        if len({change.field for change in changes}) != len(changes):
            raise HistoryResolutionError("INVALID_CHANGE", "A field changes more than once")
        draft = ObjectVersionDraft(
            object_id=object_id,
            version=version,
            kind=kind,
            conditions=after,
            changes=changes,
            derived_from=derived_from,
            source_excerpt=spec.source_excerpt,
        )
        return draft, [
            *(ref for ref in refs if not records or ref.object_id != base.object_id),
            ObjectRef(object_id=object_id, version=version),
        ]
