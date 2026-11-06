from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from openai import AuthenticationError
from pydantic import BaseModel, ConfigDict, Field

from story_rag.adapters.conversation_diagnostics import (
    DiagnosticArchive,
    ModelCallContext,
    model_call_context,
)
from story_rag.conversation_domain import (
    DatabaseBinding,
    HistoryContentQuery,
    LedgerError,
    QuestionRef,
    QuestionVersionDraft,
    RequestPayload,
    RequestRecord,
    UsageMetadata,
)
from story_rag.pipelines.calculation_lookup import resolve_answer_mode
from story_rag.pipelines.conversation_generation import (
    AnswerItemResult,
    ConversationAnswerPipeline,
    ConversationGenerationError,
    ConversationGenerationResult,
    ConversationTaskInput,
    answer_item_spans,
    merge_answer_items,
)
from story_rag.pipelines.evidence_mapping import (
    EvidenceBatch,
    EvidenceMappingError,
    EvidenceTaskInput,
    UserProvidedFact,
)
from story_rag.pipelines.history_content import HistoryContentPipeline, HistoryContentResult
from story_rag.pipelines.history_resolution import (
    ConditionPatch,
    GapResolution,
    HistoryLookup,
    HistoryResolutionError,
    HistoryResolutionPipeline,
    HistoryScope,
    RevisionSpec,
)
from story_rag.pipelines.task_planning import (
    InformationGap,
    PlannedTask,
    SourceBackedText,
    TaskPlan,
    TaskPlanningError,
    TaskPlanningPipeline,
)
from story_rag.pipelines.token_budget import BudgetExceededError, TokenBudget

logger = logging.getLogger(__name__)
_HISTORY_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "assistant",
    "current",
    "earlier",
    "explanation",
    "for",
    "from",
    "given",
    "in",
    "of",
    "prior",
    "previous",
    "previously",
    "principle",
    "process",
    "the",
    "to",
    "user",
    "users",
    "with",
    "workflow",
    "之前",
    "刚才",
    "前面",
    "流程",
}


class ConversationWorkflowError(RuntimeError):
    def __init__(self, code: str, message: str, *, outcome_unknown: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.outcome_unknown = outcome_unknown


class ConversationWorkflowModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationTurnResult(ConversationWorkflowModel):
    status: Literal["succeeded", "failed", "unknown"]
    request: RequestRecord
    revision: int
    answer: str
    items: list[AnswerItemResult] = Field(default_factory=list)
    completion_status: (
        Literal["complete", "partial", "waiting_input", "no_evidence", "cancelled", "needs_refresh"]
        | None
    ) = None
    error_code: str | None = None
    answer_not_retained: bool = False
    trace: dict[str, Any] = Field(default_factory=dict)


class ConversationTurnStarted(StartEvent):
    session_id: UUID
    binding: DatabaseBinding
    request_id: str
    payload: RequestPayload
    expected_revision: int


class ConversationWorkflow(Workflow):
    """The production conversation boundary around phases one through five."""

    def __init__(
        self,
        *,
        store: Any,
        planner: TaskPlanningPipeline,
        history: HistoryResolutionPipeline,
        evidence: Any,
        generator: ConversationAnswerPipeline,
        budget: TokenBudget | None = None,
        timeout: float = 900,
        diagnostic_directory: Path | None = None,
    ) -> None:
        super().__init__(timeout=timeout, verbose=False)
        self._store = store
        self._planner = planner
        self._history = history
        self._history_content = HistoryContentPipeline(store)
        self._evidence = evidence
        self._generator = generator
        self._budget = budget or TokenBudget()
        self._diagnostic_directory = diagnostic_directory

    @step
    async def execute_turn(self, ev: ConversationTurnStarted) -> StopEvent:
        result = await asyncio.to_thread(
            self.answer_turn,
            ev.session_id,
            ev.binding,
            request_id=ev.request_id,
            payload=ev.payload,
            expected_revision=ev.expected_revision,
        )
        return StopEvent(result=result)

    async def answer(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        request_id: str,
        payload: RequestPayload,
        expected_revision: int,
    ) -> ConversationTurnResult:
        async def finish() -> ConversationTurnResult:
            result = await self.run(
                session_id=session_id,
                binding=binding,
                request_id=request_id,
                payload=payload,
                expected_revision=expected_revision,
            )
            if not isinstance(result, ConversationTurnResult):
                raise TypeError("Unexpected conversation workflow result")
            return result

        running = asyncio.create_task(finish())
        try:
            return await asyncio.shield(running)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(running)
            except Exception:
                logger.exception("Cancelled caller's conversation work ended with an error")
            raise

    def answer_turn(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        request_id: str,
        payload: RequestPayload,
        expected_revision: int,
        plan: TaskPlan | None = None,
        diagnostic: bool = False,
    ) -> ConversationTurnResult:
        archive = (
            DiagnosticArchive(self._diagnostic_directory, str(session_id), request_id, uuid4().hex)
            if diagnostic and self._diagnostic_directory
            else None
        )
        token = model_call_context.set(
            ModelCallContext(
                before_call=lambda stage: self._store.mark_model_attempt(
                    session_id, binding, request_id, stage
                ),
                archive=archive,
            )
        )
        try:
            return self._answer_turn(
                session_id,
                binding,
                request_id=request_id,
                payload=payload,
                expected_revision=expected_revision,
                plan=plan,
                diagnostic=diagnostic,
            )
        finally:
            model_call_context.reset(token)

    def _answer_turn(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        *,
        request_id: str,
        payload: RequestPayload,
        expected_revision: int,
        plan: TaskPlan | None = None,
        diagnostic: bool = False,
    ) -> ConversationTurnResult:
        started = time.perf_counter()
        request = self._store.claim_request(
            session_id,
            binding,
            request_id=request_id,
            payload=payload,
            expected_revision=expected_revision,
        )
        if request.status == "succeeded":
            return self._replay(request, "succeeded")
        if request.status == "failed":
            return self._replay(request, "failed")
        if request.status == "unknown":
            return self._replay(request, "unknown")

        request = self._store.start_request(session_id, binding, request_id)
        metrics: list[Any] = []
        plan_result = None
        planning_input_upper = 0
        raw_task_count = 0
        try:
            scope = HistoryScope(
                session_id=session_id,
                binding=binding,
                as_of_revision=expected_revision,
                use_history=payload.use_history,
            )
            if payload.reply_to_question_id is not None:
                reference = self._history.resolve(
                    scope, HistoryLookup(question_id=payload.reply_to_question_id)
                )
                if reference.decision != "matched":
                    raise ConversationWorkflowError(
                        "INVALID_HISTORY_REFERENCE",
                        "The referenced question is not accessible in this conversation",
                    )
            if plan is None:
                try:
                    history_catalog = self._planning_history_catalog(scope, request=request)
                    planning_input_upper = self._planner.estimate_input_upper(
                        payload.message,
                        self._budget,
                        history_catalog=history_catalog,
                    )
                    self._budget.check(
                        "planning", planning_input_upper, self._budget.planning_output
                    )
                    plan_result = self._planner.plan(
                        payload.message,
                        llm_kwargs={
                            "max_tokens": self._budget.planning_output,
                            "thinking": {"type": "disabled"},
                        },
                        history_catalog=history_catalog,
                    )
                    plan = plan_result.plan
                    metrics.extend(plan_result.metrics)
                except TaskPlanningError as exc:
                    return self._fail(
                        session_id,
                        binding,
                        request,
                        failure_code=exc.code,
                        metrics=exc.metrics,
                        started=started,
                        planning_calls=exc.attempts,
                        planning_input_upper=planning_input_upper,
                        failure_details=exc.validation_errors if diagnostic else None,
                    )
                except BudgetExceededError:
                    return self._fail(
                        session_id,
                        binding,
                        request,
                        failure_code="PLANNING_BUDGET_EXCEEDED",
                        started=started,
                    )
                except AuthenticationError:
                    logger.warning("Planning request rejected by LLM authentication (HTTP 401)")
                    return self._fail(
                        session_id,
                        binding,
                        request,
                        failure_code="LLM_AUTHENTICATION_FAILED",
                        started=started,
                        planning_calls=1,
                        planning_input_upper=planning_input_upper,
                    )
                except Exception as exc:
                    logger.warning("Planning request outcome unknown (%s)", type(exc).__name__)
                    return self._fail(
                        session_id,
                        binding,
                        request,
                        failure_code="PLANNING_CALL_UNKNOWN",
                        started=started,
                        outcome_unknown=True,
                        planning_calls=1,
                        planning_input_upper=planning_input_upper,
                    )

            else:
                validated = self._planner.validate_candidate(payload.message, plan)
                if not validated.valid:
                    return self._fail(
                        session_id, binding, request, failure_code="INVALID_PLAN",
                        started=started,
                        failure_details=validated.errors if diagnostic else None,
                    )
                plan = validated.plan

            if plan.plan_kind == "interaction":
                visible_answer = plan.interaction_message or "你好，请告诉我想了解的问题。"
                committed = self._store.commit_request(
                    session_id,
                    binding,
                    request_id=request_id,
                    questions=[],
                    objects=[],
                    usage=self._usage(
                        plan_result, [], started, planning_input_upper=planning_input_upper
                    ),
                    payload=payload,
                    visible_answer=visible_answer,
                )
                return ConversationTurnResult(
                    status="succeeded",
                    request=committed,
                    revision=committed.turn_sequence or expected_revision,
                    answer=visible_answer,
                    trace={"plan_kind": "interaction", "task_count": 0},
                )
            raw_task_count = len(plan.tasks)
            plan = self._history_content_defaults(plan)
            plan = self._fold_constraint_only_goals(plan)
            plan = self._coalesce_explanation_revisions(scope, plan)
            if len(plan.tasks) > self._budget.max_tasks_per_turn:
                return self._fail(
                    session_id,
                    binding,
                    request,
                    failure_code="TOO_MANY_TASKS",
                    started=started,
                    metrics=metrics,
                    planning_calls=plan_result.llm_calls if plan_result else 0,
                    planning_input_upper=planning_input_upper,
                )
            history_content = self._read_task_content(scope, request, plan, payload)
            plan = self._bind_content_followups(scope, plan, history_content)
            specs, task_modes = self._revision_specs(scope, payload, plan)
            batch = self._history.prepare_batch(scope, request, payload, specs)
            questions, task_question_refs = self._complete_batch(
                scope, payload, plan, batch, task_modes
            )
            evidence_inputs = self._evidence_inputs(
                scope, request, payload, plan, questions, batch, task_question_refs
            )
            content_only = {
                task.task_key for task in plan.tasks
                if task.history_content is not None and (
                    task.history_content.operation != "followup"
                    or history_content[task.task_key].status != "matched"
                )
            }
            evidence_inputs = [
                item for item in evidence_inputs if item.task_key not in content_only
            ]
            mapped = (
                self._evidence.map_evidence(scope, request, payload, evidence_inputs)
                if evidence_inputs else EvidenceBatch(
                    session_id=scope.session_id, binding=scope.binding,
                    as_of_revision=scope.as_of_revision, tasks=[], evidence_pool={},
                )
            )
            generation_inputs = self._generation_inputs(
                plan, questions, mapped, task_question_refs, task_modes, evidence_inputs,
                history_content=history_content,
            )
            generated = self._generator.answer(generation_inputs)
            metrics.extend(generated.metrics)
            if any(item.status == "technical_failure" for item in generated.items) and not any(
                item.status in {"answered", "partial", "no_evidence"} for item in generated.items
            ):
                logger.warning(
                    "Conversation validation failed request_id=%s failures=%s",
                    request_id,
                    [(item.task_key, item.error_code) for item in generated.items],
                )
                return self._fail(
                    session_id,
                    binding,
                    request,
                    failure_code="ALL_ITEMS_TECHNICAL_FAILURE",
                    started=started,
                    metrics=metrics,
                    planning_calls=plan_result.llm_calls if plan_result else 0,
                    planning_input_upper=planning_input_upper,
                    failure_trace={
                        "generation_failures": [
                            {
                                "task_key": item.task_key,
                                "status": item.status,
                                "error_code": item.error_code,
                            }
                            for item in generated.items
                        ],
                        "generation_budget": generated.budget,
                        "generation_metrics": [
                            metric.model_dump(mode="json") for metric in generated.metrics
                        ],
                    }
                )
            stale = (
                self._evidence.validate_batch(scope, request, payload, evidence_inputs, mapped)
                if evidence_inputs else set()
            )
            if stale:
                items = [
                    item.model_copy(
                        update={
                            "status": "needs_refresh",
                            "answer": "",
                            "evidence_refs": [],
                            "error_code": "STALE_EVIDENCE",
                        }
                    )
                    if item.task_key in stale
                    else item
                    for item in generated.items
                ]
                generated = generated.model_copy(
                    update={
                        "items": items,
                        "answer": merge_answer_items(items),
                    }
                )
            committed_questions = self._apply_results(
                questions, generated, task_question_refs, turn_id=request.turn_id
            )
            usage = self._usage(
                plan_result, metrics, started, planning_input_upper=planning_input_upper
            )
            committed = self._store.commit_request(
                session_id,
                binding,
                request_id,
                questions=committed_questions,
                objects=batch.objects,
                usage=usage,
                payload=payload,
                visible_answer=generated.answer,
            )
            index_pending = False
            try:
                self._store.process_history_index(session_id, binding)
                index_pending = bool(self._store.pending_history_index(session_id, binding))
            except Exception:
                index_pending = True
                logger.exception("History index update deferred for request_id=%s", request_id)
            trace = {
                "raw_task_count": raw_task_count,
                "task_count": len(plan.tasks),
                "result_count": len(generated.items),
                "history_content": {
                    key: {
                        "status": value.status, "error_code": value.error_code,
                        "source": value.ref.model_dump(mode="json") if value.ref else None,
                    }
                    for key, value in history_content.items()
                },
                "history_index_pending": index_pending,
                "budget": generated.budget,
                "planning_budget": {
                    "input_upper_bound": planning_input_upper,
                    "output_reserve": (self._budget.planning_output if plan_result else 0),
                    "context_headroom": (
                        self._budget.input_limit(self._budget.planning_output)
                        - planning_input_upper
                        if plan_result
                        else None
                    ),
                    "usage_known": (
                        all(item.usage_known for item in plan_result.metrics)
                        if plan_result and plan_result.metrics
                        else None
                    ),
                },
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
            if diagnostic:
                generation_preview = self._generator.preview(
                    generation_inputs, include_messages=True
                )
                trace.update(
                    {
                        "plan": plan.model_dump(mode="json"),
                        "history": {
                            "outcomes": [
                                outcome.model_dump(mode="json") for outcome in batch.outcomes
                            ],
                            "task_modes": task_modes,
                        },
                        "evidence": mapped.model_dump(mode="json"),
                        "generation": generated.model_dump(mode="json"),
                        "messages": generation_preview["generation_messages"],
                        "calculation_lookups": generation_preview["calculation_lookups"],
                    }
                )
            return ConversationTurnResult(
                status="succeeded",
                request=committed,
                revision=committed.turn_sequence or expected_revision,
                answer=generated.answer,
                items=generated.items,
                completion_status=(
                    "complete"
                    if all(item.status == "answered" for item in generated.items)
                    else "waiting_input"
                    if all(
                        item.status in {"waiting_input", "ambiguous"} for item in generated.items
                    )
                    else generated.items[0].status
                    if all(item.status == generated.items[0].status for item in generated.items)
                    and generated.items[0].status in {"no_evidence", "cancelled", "needs_refresh"}
                    else "partial"
                ),
                trace=trace,
            )
        except ConversationGenerationError as exc:
            metrics.extend(exc.metrics)
            return self._fail(
                session_id,
                binding,
                request,
                failure_code=exc.code,
                started=started,
                metrics=metrics,
                outcome_unknown=exc.outcome_unknown,
                planning_calls=plan_result.llm_calls if plan_result else 0,
                planning_input_upper=planning_input_upper,
            )
        except (
            LedgerError,
            BudgetExceededError,
            HistoryResolutionError,
            EvidenceMappingError,
            ConversationWorkflowError,
        ) as exc:
            return self._fail(
                session_id,
                binding,
                request,
                failure_code=getattr(exc, "code", "WORKFLOW_FAILED"),
                started=started,
                metrics=metrics,
                planning_calls=plan_result.llm_calls if plan_result else 0,
                planning_input_upper=planning_input_upper,
            )
        except Exception:
            logger.exception("Conversation request ended with an unknown outcome")
            return self._fail(
                session_id,
                binding,
                request,
                failure_code="COMMIT_OR_WORKFLOW_UNKNOWN",
                started=started,
                metrics=metrics,
                outcome_unknown=True,
                planning_calls=plan_result.llm_calls if plan_result else 0,
                planning_input_upper=planning_input_upper,
            )

    def preview_turn(
        self,
        *,
        payload: RequestPayload,
        plan: TaskPlan,
        task_inputs: list[ConversationTaskInput],
    ) -> dict[str, Any]:
        """Build a no-commit diagnostic for budget and answer-count checks."""
        if [task.task_key for task in plan.tasks] != [task.task_key for task in task_inputs]:
            raise ConversationWorkflowError(
                "TASK_MAPPING_CONFLICT", "Preview tasks must match the complete plan in order"
            )
        preview = self._generator.preview(task_inputs, include_messages=True)
        return {
            "plan": plan.model_dump(mode="json"),
            "generation": preview,
            "message_bytes": len(payload.message.encode("utf-8")),
        }

    @staticmethod
    def _history_content_defaults(plan: TaskPlan) -> TaskPlan:
        tasks = []
        for task in plan.tasks:
            if (
                task.history_content is None and task.history_need != "none"
                and (task.intent == "format" or task.edit_mode == "exact_edit")
                and not task.conditions and not task.resolved_inputs
            ):
                task = task.model_copy(update={"history_content": HistoryContentQuery(
                    operation="transform",
                    selection="question" if task.history_question_no else "previous_turn",
                    question_no=task.history_question_no, version_selection="latest",
                )})
            tasks.append(task)
        return plan.model_copy(update={"tasks": tasks})

    def _read_task_content(
        self, scope: HistoryScope, request: RequestRecord, plan: TaskPlan, payload: RequestPayload
    ) -> dict[str, HistoryContentResult]:
        results = {}
        for task in plan.tasks:
            if task.history_content is not None:
                query = task.history_content
                if (
                    query.operation == "followup"
                    and query.selection == "previous_turn"
                    and task.history_question_no is not None
                ):
                    query = query.model_copy(update={
                        "selection": "question",
                        "question_no": task.history_question_no,
                        "version_selection": "latest",
                    })
                if (
                    payload.reply_to_question_id is not None
                    and query.selection == "previous_turn" and query.question_order is None
                ):
                    target = self._history.resolve(
                        scope, HistoryLookup(question_id=payload.reply_to_question_id)
                    ).selected
                    if target is not None:
                        query = query.model_copy(update={
                            "selection": "question", "question_no": target.question_no,
                            "version_selection": "latest",
                        })
                results[task.task_key] = self._history_content.read(
                    scope, request, query
                )
                value = results[task.task_key]
                logger.info(
                    "History content request_id=%s task=%s operation=%s status=%s error_code=%s",
                    request.request_id, task.task_key, query.operation,
                    value.status, value.error_code,
                )
        return results

    def _bind_content_followups(
        self, scope: HistoryScope, plan: TaskPlan, content: dict[str, HistoryContentResult]
    ) -> TaskPlan:
        tasks = []
        for task in plan.tasks:
            value = content.get(task.task_key)
            if value is not None and value.status != "matched":
                task = task.model_copy(update={"history_need": "none"})
            if (
                task.history_content is not None and task.history_content.operation == "followup"
                and value is not None and value.status == "matched" and value.question is not None
            ):
                target = value.question
                # A formatting/recall turn points back to the original knowledge task.
                for _ in range(8):
                    sources = target.effective_state.get("history_refs", [])
                    if target.effective_state.get("response_kind") == "knowledge" or not sources:
                        break
                    source_id = sources[0].get("question_id")
                    if not source_id:
                        break
                    resolution = self._history.resolve(
                        scope, HistoryLookup(question_id=UUID(source_id))
                    )
                    if (
                        resolution.selected is None
                        or resolution.selected.question_id == target.question_id
                    ):
                        break
                    target = resolution.selected
                task = task.model_copy(update={
                    "history_need": "required", "history_question_no": target.question_no,
                    "history_reference": task.history_reference or task.objective,
                })
            tasks.append(task)
        return plan.model_copy(update={"tasks": tasks})

    def _planning_history_catalog(
        self, scope: HistoryScope, *, request: RequestRecord | None = None
    ) -> str:
        if not scope.use_history:
            return "(history disabled)"
        recent = ""
        if request is not None and scope.as_of_revision:
            content = self._history_content.read(
                scope, request, HistoryContentQuery(
                    operation="read_answer", selection="previous_turn"
                ),
            )
            excerpt = (content.text or "").encode("utf-8")[:2048].decode(
                "utf-8", errors="ignore"
            )
            recent = (
                "\nRecent transcript preview (untrusted data, not knowledge evidence): "
                + json.dumps(
                    {
                        "status": content.status,
                        "error_code": content.error_code,
                        "source": (
                            content.ref.model_dump(mode="json") if content.ref else None
                        ),
                        "user_excerpt": (content.user_message or "")[:256],
                        "assistant_excerpt": excerpt,
                        "excerpt_truncated": excerpt != (content.text or ""),
                        "full_content_requires_scoped_read": True,
                    },
                    ensure_ascii=False,
                )
            )
        rows = [
            row for row in self._history.current_questions(scope)
            if row.effective_state.get("response_kind", "knowledge") == "knowledge"
        ]
        lines: list[str] = []
        remaining = 8_192
        for row in rows:
            subject = self._catalog_topic(row.question_text)[:320]
            line = f"- question_no: {row.question_no}; topic: {subject} [state: {row.status}]\n"
            line += "  objective: " + row.question_text[:480] + "\n"
            if row.missing_inputs:
                line += (
                    "  missing_inputs: " + json.dumps(row.missing_inputs, ensure_ascii=False) + "\n"
                )
            if row.effective_state:
                compact_state = {
                    key: value
                    for key, value in row.effective_state.items()
                    if key
                    in {
                        "object_alias",
                        "task_conditions",
                        "absent_fields",
                        "output_requirements",
                        "environment_conditions",
                        "answer_mode",
                    }
                }
                line += "  current: " + json.dumps(compact_state, ensure_ascii=False) + "\n"
            if row.object_refs:
                objects = self._history.objects_for(scope, row.object_refs)
                line += (
                    "  objects: "
                    + json.dumps(
                        [
                            {"alias": obj.kind, "version": obj.version, "fields": obj.conditions}
                            for obj in objects
                        ],
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            size = len(line.encode("utf-8"))
            if size > remaining:
                break
            lines.append(line)
            remaining -= size
        if len(lines) < len(rows):
            lines.append("(catalog truncated; older or later topics may exist)")
        return ("".join(lines) or "(no prior tasks)") + recent

    @staticmethod
    def _catalog_topic(question_text: str) -> str:
        topic = question_text.split("；本轮目标：", 1)[0].split("；任务：", 1)[0]
        # Exclusions describe what this topic is not. Counting their tokens as
        # positive evidence can map a draft follow-up to the generic policy.
        return re.split(
            r"\bnot\b|而非|不是|不包括|不涉及|并非",
            topic,
            maxsplit=1,
            flags=re.I,
        )[0].rstrip(" (,（")

    def _coalesce_explanation_revisions(
        self,
        scope: HistoryScope,
        plan: TaskPlan,
    ) -> TaskPlan:
        if (
            not scope.use_history or len(plan.tasks) < 2
            or any(task.history_content is not None for task in plan.tasks)
        ):
            return plan

        def unique(items: list[Any]) -> list[Any]:
            result = []
            seen: set[str] = set()
            for item in items:
                key = item.model_dump_json()
                if key not in seen:
                    result.append(item)
                    seen.add(key)
            return result

        candidates = self._history.current_questions(scope)
        current = plan
        while True:
            pair = None
            for later_index, later in enumerate(current.tasks):
                if later.intent != "revise":
                    continue
                earlier_candidates = current.tasks[:later_index]
                for earlier in earlier_candidates:
                    if (
                        earlier is None
                        or earlier.intent != "answer"
                        or earlier.history_need != "required"
                        or later.history_need != "required"
                    ):
                        continue

                    def selected(task):
                        if task.history_question_no is not None:
                            return next(
                                (
                                    row
                                    for row in candidates
                                    if row.question_no == task.history_question_no
                                ),
                                None,
                            )
                        return self._unique_history_candidate(task, candidates)

                    left, right = selected(earlier), selected(later)
                    if left is not None and right is not None:
                        answer_modes = {
                            resolve_answer_mode(
                                task.answer_mode,
                                task.objective.text,
                                row.effective_state.get("answer_mode")
                                or resolve_answer_mode(None, row.question_text),
                            )
                            for task, row in ((earlier, left), (later, right))
                        }
                        if len(answer_modes) != 1:
                            continue
                    changes = {
                        (c.scope, c.field): (c.op, c.value, c.unit) for c in earlier.conditions
                    }
                    compatible = all(
                        (c.scope, c.field) not in changes
                        or changes[c.scope, c.field] == (c.op, c.value, c.unit)
                        for c in later.conditions
                    )
                    if (
                        left is not None
                        and right is not None
                        and left.question_id == right.question_id
                        and compatible
                    ):
                        pair = earlier, later
                        break
                if pair is not None:
                    break
            if pair is None:
                return current
            earlier, later = pair
            objective = earlier.objective.text + "；" + later.objective.text
            requirements = unique(
                [
                    *earlier.output_requirements,
                    *later.output_requirements,
                    later.objective,
                ]
            )
            constraints = unique([*earlier.constraints, *later.constraints])
            prohibited = unique(
                [
                    *earlier.prohibited_actions,
                    *later.prohibited_actions,
                ]
            )
            gaps = unique(
                [
                    *earlier.information_gaps,
                    *later.information_gaps,
                ]
            )
            if (
                len(objective) > 1200
                or len(requirements) > 32
                or len(constraints) > 32
                or len(prohibited) > 32
                or len(gaps) > 32
            ):
                return current
            merged = earlier.model_copy(
                update={
                    "intent": "revise",
                    "objective": earlier.objective.model_copy(update={"text": objective}),
                    "output_requirements": requirements,
                    "constraints": constraints,
                    "prohibited_actions": prohibited,
                    "information_gaps": gaps,
                    "conditions": list(
                        {
                            (c.scope, c.field): c for c in [*later.conditions, *earlier.conditions]
                        }.values()
                    ),
                    "resolved_inputs": unique([*earlier.resolved_inputs, *later.resolved_inputs]),
                    "dependencies": list(
                        dict.fromkeys(
                            key
                            for key in [*earlier.dependencies, *later.dependencies]
                            if key not in {earlier.task_key, later.task_key}
                        )
                    ),
                    "relation_candidates": [
                        relation
                        for relation in [*earlier.relation_candidates, *later.relation_candidates]
                        if relation.target_task_key not in {earlier.task_key, later.task_key}
                    ],
                }
            )
            next_tasks = []
            for task in current.tasks:
                if task.task_key == later.task_key:
                    continue
                if task.task_key == earlier.task_key:
                    next_tasks.append(merged)
                    continue
                dependencies = list(
                    dict.fromkeys(
                        earlier.task_key if key == later.task_key else key
                        for key in task.dependencies
                    )
                )
                relations = [
                    relation.model_copy(
                        update={
                            "target_task_key": (
                                earlier.task_key
                                if relation.target_task_key == later.task_key
                                else relation.target_task_key
                            )
                        }
                    )
                    for relation in task.relation_candidates
                    if relation.target_task_key != task.task_key
                ]
                next_tasks.append(
                    task.model_copy(
                        update={
                            "dependencies": dependencies,
                            "relation_candidates": relations,
                        }
                    )
                )
            next_goals = [
                goal.model_copy(
                    update={
                        "task_keys": list(
                            dict.fromkeys(
                                earlier.task_key if key == later.task_key else key
                                for key in goal.task_keys
                            )
                        )
                    }
                )
                for goal in current.goals
            ]
            current = TaskPlan.model_validate(
                current.model_copy(
                    update={
                        "tasks": next_tasks,
                        "goals": next_goals,
                    }
                ).model_dump()
            )
            logger.info("Combined explanation and revision for one historical object")

    @staticmethod
    def _fold_constraint_only_goals(plan: TaskPlan) -> TaskPlan:
        if any(task.history_content is not None for task in plan.tasks):
            return plan
        """Keep a negative scope instruction without creating a spurious answer."""
        current = plan
        for goal in list(current.goals):
            if len(current.tasks) < 2 or len(goal.task_keys) != 1:
                continue
            key = goal.task_keys[0]
            task = next((item for item in current.tasks if item.task_key == key), None)
            if task is None or task.intent not in {"answer", "clarify"}:
                continue
            if not re.search(
                r"不改变|保持不变|不得改变|do not change|unchanged",
                goal.source_excerpt,
                re.I,
            ):
                continue
            if re.search(
                r"[?？]|为什么|如何|请说明|请解释|\bwhy\b|\bhow\b",
                goal.source_excerpt,
                re.I,
            ):
                continue
            if not re.match(
                r"acknowledge|record and preserve|preserve|keep|maintain|confirm|"
                r"确保|保持|确认",
                task.objective.text,
                re.I,
            ):
                continue
            if any(
                key in other.dependencies
                or any(relation.target_task_key == key for relation in other.relation_candidates)
                for other in current.tasks
                if other.task_key != key
            ) or any(
                key in other.task_keys for other in current.goals if other.goal_key != goal.goal_key
            ):
                continue
            constraint = SourceBackedText(text=goal.goal_text, source_excerpt=goal.source_excerpt)
            survivors = []
            for other in current.tasks:
                if other.task_key == key:
                    continue
                constraints = [*other.constraints, constraint]
                if len(constraints) > 32:
                    return current
                survivors.append(other.model_copy(update={"constraints": constraints}))
            current = TaskPlan.model_validate(
                current.model_copy(
                    update={
                        "tasks": survivors,
                        "goals": [item for item in current.goals if item.goal_key != goal.goal_key],
                    }
                ).model_dump()
            )
            logger.info("Folded a constraint-only goal into the remaining tasks")
        return current

    def _revision_specs(
        self,
        scope: HistoryScope,
        payload: RequestPayload,
        plan: TaskPlan,
    ) -> tuple[list[RevisionSpec], dict[str, str]]:
        candidates = [
            row for row in self._history.current_questions(scope)
            if row.effective_state.get("response_kind", "knowledge") == "knowledge"
        ] if payload.use_history else []
        by_id = {question.question_id: question for question in candidates}
        history_tasks = [
            task
            for task in plan.tasks
            if task.history_need in {"required", "optional", "ambiguous"}
        ]
        explicit = payload.reply_to_question_id
        explicit_target = by_id.get(explicit) if explicit else None
        specs: list[RevisionSpec] = []
        modes: dict[str, str] = {}
        used_targets: set[UUID] = set()
        for task in plan.tasks:
            target = None
            mode = "new"
            content_only = (
                task.history_content is not None and task.history_content.operation != "followup"
            )
            if (
                not content_only and payload.use_history
                and task.history_need in {"required", "ambiguous", "optional"}
            ):
                if explicit and len(history_tasks) == 1:
                    target = explicit_target
                elif task.history_need == "ambiguous":
                    target = None
                elif task.history_question_no is not None:
                    target = next(
                        (row for row in candidates if row.question_no == task.history_question_no),
                        None,
                    )
                elif task.history_reference is not None:
                    target = self._unique_history_candidate(task, candidates)
                if (
                    not explicit
                    and task.history_content is None
                    and task.history_reference is not None
                    and re.search(
                        r"第二[个项点]|the second|2nd", task.history_reference.source_excerpt, re.I
                    )
                    and len(candidates) > 1
                ):
                    target = None
                if target is None and task.resolved_inputs and task.history_need != "ambiguous":
                    pending = [
                        row
                        for row in candidates
                        if row.missing_inputs
                        and all(
                            gap.missing_input in row.missing_inputs for gap in task.resolved_inputs
                        )
                    ]
                    if len(pending) == 1:
                        target = pending[0]
                if target is not None:
                    mode = self._revision_kind(task)
                    if target.question_id in used_targets and mode not in {"extend", "compare"}:
                        mode = "extend"
                    used_targets.add(target.question_id)
                elif task.history_need in {"required", "ambiguous"}:
                    mode = "ambiguous" if candidates else "not_found"
            if not payload.use_history and task.history_need in {"required", "ambiguous"}:
                mode = "ambiguous"
            modes[task.task_key] = mode
            root_topic = self._catalog_topic(target.question_text) if target else ""
            current_topic = task.object_scope.text if task.object_scope else ""
            resolved_history_gap = bool(
                target
                and target.missing_inputs
                and (
                    task.intent == "diagnose"
                    or any(
                        dependency.intent == "diagnose"
                        for dependency in plan.tasks
                        if dependency.task_key in task.dependencies
                    )
                )
            )
            topic = root_topic if target else current_topic
            question_text = (
                f"{topic}；任务：{task.objective.text}" if topic else task.objective.text
            )
            if len(question_text) > 4000:
                raise ConversationWorkflowError(
                    "HISTORY_CONTEXT_TOO_LONG",
                    "Effective question history exceeds the stored context limit",
                )
            requirements = [
                *self._requirements(task),
                *self._shared_change_requirements(plan, task),
            ]
            current_goals = [goal for goal in plan.goals if task.task_key in goal.task_keys]
            goal_requirements = [
                goal.goal_text for goal in current_goals if len(goal.task_keys) == 1
            ]
            if target and mode not in {"new", "ambiguous", "cancel"} and not resolved_history_gap:
                requirements = [
                    *self._retained_requirements(target.requirements, task),
                    *requirements,
                ]
            requirements = list(dict.fromkeys(requirements))
            if len(requirements) > 64:
                raise ConversationWorkflowError(
                    "TOO_MANY_REQUIREMENTS", "Effective requirements exceed the task limit"
                )
            prior_state = dict(target.effective_state) if target else {}
            answer_mode = resolve_answer_mode(
                task.answer_mode,
                " ".join([task.objective.text, *(item.text for item in task.output_requirements)]),
                prior_state.get("answer_mode") or (
                    resolve_answer_mode(None, target.question_text) if target else None
                ),
            )
            changed_fields = {
                change.field for change in task.conditions if change.op in {"replace", "remove"}
            }
            prior_links = prior_state.get("condition_requirements", {})
            obsolete = {
                text for text, fields in prior_links.items() if changed_fields.intersection(fields)
            }
            if prior_state.get("fallback_objective"):
                obsolete.add(prior_state["fallback_objective"])
            obsolete.update(prior_state.get("goal_requirements", []))
            requirements = [text for text in requirements if text not in obsolete]
            requirements = list(dict.fromkeys([*requirements, *goal_requirements]))
            if len(requirements) > 64:
                raise ConversationWorkflowError(
                    "TOO_MANY_REQUIREMENTS", "Effective requirements exceed the task limit"
                )
            current_links = {
                text: fields for text, fields in prior_links.items() if text not in obsolete
            }
            for constraint in task.constraints:
                fields = [
                    change.field
                    for change in task.conditions
                    if change.op not in {"inherit", "remove"}
                    and (
                        change.source_excerpt in constraint.source_excerpt
                        or constraint.source_excerpt in change.source_excerpt
                    )
                ]
                if fields:
                    current_links[constraint.text] = fields
            environment = dict(prior_state.get("environment_conditions", {}))
            if plan.schema_version == 1:
                # Bounded compatibility for old plans without structured condition fields.
                grounded = " ".join(
                    [
                        task.objective.source_excerpt,
                        *(item.source_excerpt for item in task.constraints),
                        *(
                            goal.source_excerpt
                            for goal in plan.goals
                            if task.task_key in goal.task_keys
                        ),
                        *self._shared_change_requirements(plan, task),
                    ]
                )
                for key, pattern, value in (
                    ("tenancy", r"多租户|multi[- ]tenant", "多租户"),
                    ("dispatch", r"后台队列|排队|进入.*队列|background queue", "后台队列异步执行"),
                    ("executor", r"共享服务账号|shared service account", "共享服务账号"),
                ):
                    if re.search(pattern, grounded, re.I):
                        environment[key] = value
            requirements = list(dict.fromkeys([*requirements, *environment.values()]))
            previous_fields = dict(prior_state.get("task_conditions", {}))
            task_changes = []
            patches = []
            old_objects = self._history.objects_for(scope, target.object_refs) if target else []
            old_fields = old_objects[0].conditions if len(old_objects) == 1 else {}
            absent_fields = dict(prior_state.get("absent_fields", {}))
            legacy_context = prior_state.get("legacy_context")
            if target and not target.object_refs and not target.effective_state:
                legacy_context = {
                    "text": target.source_excerpt,
                    "question_id": str(target.question_id),
                    "version": target.version,
                }
            for change in task.conditions:
                if change.scope == "object":
                    if change.op == "remove" and mode == "new" and change.field not in old_fields:
                        absent_fields[change.field] = change.source_excerpt
                        continue
                    if change.op in {"supplement", "replace"}:
                        absent_fields.pop(change.field, None)
                    if change.op == "inherit" and change.field not in old_fields:
                        if legacy_context:
                            # Keep the original user text, without inventing a typed value.
                            continue
                        task.information_gaps.append(
                            InformationGap(
                                description=f"历史中未保留字段 {change.field} 的可信值，请补充该值",
                                state="not_provided",
                                basis_excerpt=change.source_excerpt,
                            )
                        )
                    if change.op != "inherit":
                        patches.append(
                            ConditionPatch(
                                field=change.field,
                                op=(
                                    change.op
                                    if change.op == "remove"
                                    else "replace"
                                    if change.field in old_fields
                                    else "supplement"
                                ),
                                value=change.value,
                                unit=change.unit,
                                source_excerpt=change.source_excerpt,
                            )
                        )
                    continue
                if change.op == "inherit":
                    if change.field not in previous_fields:
                        raise ConversationWorkflowError(
                            "MISSING_INHERITED_FIELD", "Cannot inherit an unknown field"
                        )
                    continue
                if (change.op in {"replace", "remove"}) != (change.field in previous_fields):
                    raise ConversationWorkflowError(
                        "INVALID_TASK_CONDITION", "Task condition operation is invalid"
                    )
                before = previous_fields.get(change.field)
                if change.op == "remove":
                    del previous_fields[change.field]
                else:
                    previous_fields[change.field] = change.value
                task_changes.append({**change.model_dump(), "before": before})
            state = {
                "schema_version": 1,
                "answer_mode": answer_mode,
                "condition_units": {
                    **prior_state.get("condition_units", {}),
                    **{change.field: change.unit for change in task.conditions
                       if change.op != "inherit"},
                },
                "environment_conditions": environment,
                "object_alias": task.object_alias or prior_state.get("object_alias"),
                "task_conditions": previous_fields,
                "task_changes": task_changes,
                "absent_fields": absent_fields,
                "condition_requirements": current_links,
                "fallback_objective": (
                    task.objective.text
                    if not (task.constraints or task.output_requirements or task.prohibited_actions)
                    else None
                ),
                "output_requirements": [item.text for item in task.output_requirements],
                "edit_mode": task.edit_mode,
                "legacy_context": legacy_context,
                "goal_requirements": goal_requirements,
                "format_source_excerpts": (
                    list(dict.fromkeys(g.source_excerpt for g in current_goals))
                    if task.intent == "format" and mode == "new"
                    else prior_state.get("format_source_excerpts", [])
                ),
                "format_source_ref": prior_state.get("format_source_ref"),
            }
            if task_changes:
                removed = [
                    str(change["before"])
                    for change in task_changes
                    if change["op"] in {"remove", "replace"}
                ]
                requirements = [
                    text for text in requirements if not any(value in text for value in removed)
                ]
            specs.append(
                RevisionSpec(
                    task_key=task.task_key,
                    kind=mode,
                    source_excerpt=task.objective.source_excerpt,
                    question_text=question_text,
                    lookup=HistoryLookup(question_id=target.question_id) if target else None,
                    requirements=requirements,
                    requirements_complete=True,
                    effective_state=state,
                    patches=patches,
                    object_alias=state["object_alias"],
                    new_object_kind=(
                        task.object_alias
                        if mode == "new" or (target and not target.object_refs and patches)
                        else None
                    ),
                    target_object_id=(
                        target.object_refs[0].object_id
                        if target and len(target.object_refs) == 1
                        else None
                    ),
                    resolved_gaps=self._resolved_gaps(target, task, payload.message),
                )
            )
        resolved_by_target = {
            spec.lookup.question_id: spec.resolved_gaps
            for spec in specs
            if spec.lookup is not None and spec.resolved_gaps
        }
        specs = [
            spec.model_copy(update={"resolved_gaps": resolved_by_target[spec.lookup.question_id]})
            if spec.lookup is not None
            and not spec.resolved_gaps
            and spec.lookup.question_id in resolved_by_target
            else spec
            for spec in specs
        ]
        # Merge compatible changes once, then all affected questions share the version.
        owners: dict[Any, RevisionSpec] = {}
        for spec in specs:
            if spec.kind in {"compare", "extend"}:
                continue
            key = spec.target_object_id or spec.object_alias
            if key is None:
                continue
            if key not in owners:
                owners[key] = spec
                continue
            owner = owners[key]
            existing = {patch.field: patch for patch in owner.patches}
            for patch in spec.patches:
                if patch.field in existing:
                    prior = existing[patch.field]
                    if (prior.op, prior.value, prior.unit) != (patch.op, patch.value, patch.unit):
                        raise ConversationWorkflowError(
                            "CONFLICTING_OBJECT_CHANGES", "One object has conflicting field changes"
                        )
                else:
                    owner.patches.append(patch)
            spec.patches = []
        return specs, modes

    @staticmethod
    def _resolved_gaps(target: Any | None, task: PlannedTask, message: str) -> list[GapResolution]:
        if target is None or not target.missing_inputs:
            return []
        if task.resolved_inputs:
            return [
                GapResolution(missing_input=gap.missing_input, source_excerpt=gap.source_excerpt)
                for gap in task.resolved_inputs
                if gap.missing_input in target.missing_inputs and gap.source_excerpt in message
            ]
        supplied = [task.objective.source_excerpt] + [
            fact.source_excerpt for fact in task.constraints
        ]
        if (
            task.intent == "diagnose"
            and not any(gap.state == "explicitly_missing" for gap in task.information_gaps)
            and len(supplied) >= 2
        ):
            outcome = next(
                (
                    quote
                    for quote in supplied
                    if re.search(
                        r"deny|denied|refus|rejected|拒绝|失败|错误|error|status", quote, re.I
                    )
                ),
                None,
            )
            if outcome is not None:
                return [
                    GapResolution(missing_input=gap, source_excerpt=outcome)
                    for gap in target.missing_inputs
                ]
        if not supplied:
            return []
        excerpt = supplied[0]
        source_terms = _terms(excerpt)
        return [
            GapResolution(missing_input=gap, source_excerpt=excerpt)
            for gap in target.missing_inputs
            if len(_terms(gap) & source_terms) >= 2
        ]

    @staticmethod
    def _unique_history_candidate(task: PlannedTask, candidates: list[Any]) -> Any | None:
        if not candidates:
            return None
        scores = [
            ConversationWorkflow._scope_match_score(task, question) for question in candidates
        ]
        if task.object_scope is not None:
            primary_scope = re.split(
                r"[;；]|\bnot\b|而非|不是|不包括|不涉及|并非",
                task.object_scope.text,
                maxsplit=1,
                flags=re.I,
            )[0]
            topics = [
                _terms(ConversationWorkflow._catalog_topic(question.question_text))
                - _HISTORY_STOPWORDS
                for question in candidates
            ]
            unique_scope_terms = {
                term
                for term in _terms(primary_scope) - _HISTORY_STOPWORDS
                if sum(term in topic for topic in topics) == 1
            }
            scores = [
                score + 30 * len(unique_scope_terms & topic)
                for score, topic in zip(scores, topics, strict=True)
            ]
        best = max(scores)
        top = [
            question for question, score in zip(candidates, scores, strict=True) if score == best
        ]
        scope_winner = top[0] if best > 0 and len(top) == 1 else None
        temporal = " ".join(
            [
                task.history_reference.source_excerpt if task.history_reference else "",
                task.objective.source_excerpt,
            ]
        )
        if (
            best > 0
            and len(top) > 1
            and re.search(r"刚才|上一轮|之前|前面|earlier|previous|recent", temporal, re.I)
            and len({ConversationWorkflow._catalog_topic(q.question_text) for q in top}) == 1
        ):
            newest = max(getattr(q, "question_no", 0) for q in top)
            latest = [q for q in top if getattr(q, "question_no", 0) == newest]
            if len(latest) == 1:
                scope_winner = latest[0]
        hint = task.history_reference.text.strip() if task.history_reference else ""
        exact = [
            question
            for question in candidates
            if ConversationWorkflow._catalog_topic(question.question_text).strip(" .")
            == hint.strip(" .")
        ]
        if not exact:
            normalized_hint = hint.split("(", 1)[0].split("（", 1)[0].strip(" .")
            exact = [
                question
                for question in candidates
                if ConversationWorkflow._catalog_topic(question.question_text).strip(" .")
                == normalized_hint
            ]
        if len(exact) == 1:
            selected = exact[0]
            selected_score = scores[candidates.index(selected)]
            if selected_score > 0 and selected_score == best:
                return selected
            if scope_winner is not None:
                logger.warning("History topic alias disagreed with unique object match")
                return scope_winner
            return None
        if exact:
            return scope_winner if scope_winner in exact else None
        if scope_winner is not None:
            return scope_winner
        reference = (
            f"{task.history_reference.text} {task.history_reference.source_excerpt}"
            if task.history_reference
            else ""
        )
        if task.object_scope is not None:
            reference += " " + task.object_scope.text
        terms = _terms(reference)
        if not terms:
            return None
        scored = []
        for question in candidates:
            candidate_text = ConversationWorkflow._catalog_topic(question.question_text)
            overlap = len(terms & _terms(candidate_text))
            if overlap >= 3:
                scored.append((overlap, question))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored or (len(scored) > 1 and scored[0][0] - scored[1][0] < 2):
            return None
        return scored[0][1]

    @staticmethod
    def _scope_match_score(task: PlannedTask, question: Any) -> int:
        def positive(text: str) -> str:
            return re.split(
                r"\bnot\b|而非|不是|不包括|不涉及|并非",
                text,
                maxsplit=1,
                flags=re.I,
            )[0]

        scope = (
            _terms(positive(task.object_scope.text)) - _HISTORY_STOPWORDS
            if task.object_scope is not None
            else set()
        )
        reference = (
            _terms(task.history_reference.text) - _HISTORY_STOPWORDS
            if task.history_reference is not None
            else set()
        )
        source = (
            _terms(positive(task.history_reference.source_excerpt)) - _HISTORY_STOPWORDS
            if task.history_reference is not None
            else set()
        )
        topic = (
            _terms(ConversationWorkflow._catalog_topic(question.question_text)) - _HISTORY_STOPWORDS
        )
        recent_excerpt = _terms(question.source_excerpt or "") - _HISTORY_STOPWORDS
        # Negated exclusions are not positive targets; latest excerpts may
        # mention both objects and therefore serve only as a weak fallback.
        return (
            12 * len(scope & topic)
            + 3 * len(reference & topic)
            + min(
                2,
                max(
                    len(source & topic),
                    len(scope & recent_excerpt),
                    len(source & recent_excerpt),
                ),
            )
        )

    @staticmethod
    def _revision_kind(task: PlannedTask) -> str:
        return {
            "revise": "replace",
            "format": "format",
            "compare": "compare",
            "cancel": "cancel",
        }.get(task.intent, "supplement")

    @staticmethod
    def _requirements(task: PlannedTask) -> list[str]:
        requirements = [
            item.text
            for item in [
                *(
                    item
                    for item in task.output_requirements
                    if not ConversationWorkflow._cross_task_output_requirement(item.text)
                ),
                *task.constraints,
                *task.prohibited_actions,
            ]
        ]
        return list(dict.fromkeys(requirements or [task.objective.text]))

    @staticmethod
    def _cross_task_output_requirement(text: str) -> bool:
        return bool(
            re.search(r"\b(?:together with|alongside)\b", text, re.I)
            and re.search(r"\bgeneral\b", text, re.I)
            and re.search(r"\bspecific\b", text, re.I)
            or re.search(r"同步.*通用.*具体", text)
        )

    @staticmethod
    def _shared_change_requirements(plan: TaskPlan, task: PlannedTask) -> list[str]:
        # A goal saying that the same changes apply to another task carries the
        # source task's explicit conditions into that task's durable requirements.
        # Shared goals alone do not authorize copying unrelated task conditions.
        result: list[str] = []
        for index, goal in enumerate(plan.goals):
            if task.task_key not in goal.task_keys or not re.search(
                r"这些变化也适用于|同样适用于|same changes (?:also )?apply",
                goal.source_excerpt,
                re.I,
            ):
                continue
            earlier_source_keys = {
                earlier.task_keys[0]
                for earlier in plan.goals[:index]
                if len(earlier.task_keys) == 1
            }
            for source in plan.tasks:
                if (
                    source.task_key != task.task_key
                    and source.task_key in goal.task_keys
                    and source.task_key in earlier_source_keys
                    and task.task_key not in earlier_source_keys
                ):
                    result.append(source.objective.text)
                    result.extend(item.text for item in source.constraints)
        return list(dict.fromkeys(result))

    @staticmethod
    def _retained_requirements(previous: list[str], task: PlannedTask) -> list[str]:
        current = " ".join(
            [
                task.objective.text,
                *(item.text for item in task.constraints),
                *(item.text for item in task.output_requirements),
            ]
        )
        language = bool(re.search(r"中文|英文|Chinese|English", current, re.I))
        presentation = bool(re.search(r"表格|摘要|一句|table|summary|one sentence", current, re.I))
        multi_tenant = bool(re.search(r"multi[- ]?tenant|多租户", current, re.I))
        queued = bool(re.search(r"\bqueu(?:e|ed|ing)\b|background queue|队列|排队", current, re.I))
        return [
            text
            for text in previous
            if not (
                multi_tenant
                and re.search(r"single[- ]tenant|单租户", text, re.I)
                or queued
                and re.search(r"synchron(?:ous|ously)|同步执行", text, re.I)
                or language
                and re.search(r"中文|英文|Chinese|English", text, re.I)
                or presentation
                and re.search(r"表格|摘要|一句|table|summary|one sentence", text, re.I)
            )
        ]

    def _complete_batch(
        self,
        scope: HistoryScope,
        payload: RequestPayload,
        plan: TaskPlan,
        batch: Any,
        modes: dict[str, str],
    ) -> tuple[list[QuestionVersionDraft], dict[str, Any]]:
        task_by_key = {task.task_key: task for task in plan.tasks}
        questions = list(batch.questions)
        refs: dict[str, Any] = {
            outcome.task_key: outcome.question_ref
            for outcome in batch.outcomes
            if outcome.decision == "prepared" and outcome.question_ref is not None
        }
        next_no = max(
            [
                self._history.next_question_no(scope),
                *(item.question_no + 1 for item in questions),
            ]
        )
        for outcome in batch.outcomes:
            if outcome.decision not in {"ambiguous", "not_found"}:
                continue
            task = task_by_key[outcome.task_key]
            question = QuestionVersionDraft(
                question_id=uuid4(),
                question_no=next_no,
                version=1,
                question_text=task.objective.text,
                requirements=self._requirements(task),
                status="waiting_input",
                missing_inputs=[
                    "当前会话中没有找到该历史问题，请提供原问题或正确引用"
                    if outcome.decision == "not_found"
                    else "请明确是哪个对象或哪组列表；可提供题号或原文片段"
                ],
                source_excerpt=task.objective.source_excerpt,
                revision_kind="new",
            )
            next_no += 1
            questions.append(question)
            refs[outcome.task_key] = QuestionRef(question_id=question.question_id, version=1)

        if set(refs) != {task.task_key for task in plan.tasks}:
            raise ConversationWorkflowError("TASK_NOT_MAPPED", "Not every task was mapped")
        identities = [(ref.question_id, ref.version) for ref in refs.values()]
        if len(identities) != len(set(identities)) or len(questions) != len(refs):
            raise ConversationWorkflowError(
                "QUESTION_MAPPING_CONFLICT", "Each task needs one distinct question version"
            )

        for _index, task in enumerate(plan.tasks):
            if task.task_key not in refs:
                raise ConversationWorkflowError(
                    "TASK_NOT_MAPPED", f"Task {task.task_key} has no question mapping"
                )
            ref = refs[task.task_key]
            question = next(
                item
                for item in questions
                if item.question_id == ref.question_id and item.version == ref.version
            )
            if question.effective_state.get(
                "format_source_excerpts"
            ) and not question.effective_state.get("format_source_ref"):
                question.effective_state["format_source_ref"] = {
                    "question_id": str(question.question_id),
                    "version": question.version,
                }
            lookup_only = question.effective_state.get("answer_mode") == "calculation_lookup"
            gaps = [
                gap.description
                for gap in task.information_gaps
                if gap.necessary
                and (plan.schema_version == 2 or gap.state == "explicitly_missing")
                and gap.origin != "knowledge_evidence"
                and (not lookup_only or gap.origin == "history_reference")
                and not re.search(
                    r"知识库|检索资料|检索证据|knowledge base|retrieved evidence|knowledge corpus",
                    gap.description,
                    re.I,
                )
            ]
            if lookup_only and modes[task.task_key] not in {"ambiguous", "not_found"}:
                # Numeric-parameter gaps no longer block lookup; unresolved identity still does.
                question.missing_inputs = []
                if question.status == "waiting_input":
                    question.status = "ready"
            if task.edit_mode == "exact_edit" and task.history_content is None and not re.search(
                r"原文[:：]|原句[:：]|source text[:：]", payload.message, re.I
            ):
                gaps = ["请指定要读取的历史轮次或提供要逐字修改的原文片段。"]
            if gaps:
                question = question.model_copy(
                    update={"status": "waiting_input", "missing_inputs": list(dict.fromkeys(gaps))}
                )
                existing_index = next(
                    index
                    for index, item in enumerate(questions)
                    if item.question_id == question.question_id and item.version == question.version
                )
                questions[existing_index] = question
            if not question.missing_inputs:
                inactive = [
                    text
                    for text in question.requirements
                    if re.search(
                        r"缺[少失].{0,25}[时就].{0,20}[问补]|if.{0,30}missing.{0,30}ask",
                        text,
                        re.I,
                    )
                ]
                if inactive:
                    question.requirements = [
                        text for text in question.requirements if text not in inactive
                    ] or [task.objective.text]
                    question.effective_state["inactive_clarification_requirements"] = inactive
            if modes[task.task_key] not in {"new", "ambiguous", "not_found"}:
                try:
                    old = self._history.resolve(
                        scope, HistoryLookup(question_id=question.question_id)
                    ).selected
                    if old and old.evidence_refs:
                        replacement = question.model_copy(
                            update={"evidence_refs": old.evidence_refs}
                        )
                        questions[questions.index(question)] = replacement
                except HistoryResolutionError:
                    pass
        return questions, refs

    def _evidence_inputs(
        self,
        scope: HistoryScope,
        request: RequestRecord,
        payload: RequestPayload,
        plan: TaskPlan,
        questions: list[QuestionVersionDraft],
        batch: Any,
        refs: dict[str, Any],
    ) -> list[EvidenceTaskInput]:
        objects = {item.object_id: item for item in batch.objects}
        inputs: list[EvidenceTaskInput] = []
        task_by_ref = {(ref.question_id, ref.version): key for key, ref in refs.items()}
        planned = {task.task_key: task for task in plan.tasks}
        for question in questions:
            task_key = task_by_ref[(question.question_id, question.version)]
            task = planned[task_key]
            fact_sources = [
                item.source_excerpt
                for item in [
                    *task.constraints,
                    *task.prohibited_actions,
                    *([task.object_scope] if task.object_scope else []),
                ]
            ]
            for dependency in task.dependencies:
                fact_sources.extend(item.source_excerpt for item in planned[dependency].constraints)
            fact_sources = list(dict.fromkeys(fact_sources))
            if task.intent == "diagnose" or any(
                planned[dependency].intent == "diagnose" for dependency in task.dependencies
            ):
                fact_sources.extend(
                    payload.message[offset : offset + 512]
                    for offset in range(0, len(payload.message), 512)
                )
                fact_sources = list(dict.fromkeys(fact_sources))
            if len(fact_sources) > 32:
                raise ConversationWorkflowError(
                    "TOO_MANY_USER_FACTS", "Current request exceeds the user-fact limit"
                )
            task_objects = [
                objects[ref.object_id] for ref in question.object_refs if ref.object_id in objects
            ]
            if scope.use_history and question.object_refs:
                prepared = {
                    obj.object_id: obj
                    for obj in task_objects
                    if any(
                        ref.object_id == obj.object_id and ref.version == obj.version
                        for ref in question.object_refs
                    )
                }
                remaining = [ref for ref in question.object_refs if ref.object_id not in prepared]
                existing = self._history.objects_for(scope, remaining)
                by_id = {item.object_id: item for item in existing}
                task_objects = [
                    prepared.get(ref.object_id, by_id.get(ref.object_id))
                    for ref in question.object_refs
                ]
                task_objects = [item for item in task_objects if item is not None]
            inputs.append(
                EvidenceTaskInput(
                    task_key=task_key,
                    question=question,
                    objects=task_objects,
                    prior_evidence_refs=question.evidence_refs,
                    prior_object_refs=question.object_refs,
                    user_facts=[
                        UserProvidedFact(field=f"current_fact_{index}", source_excerpt=source)
                        for index, source in enumerate(fact_sources, start=1)
                    ],
                )
            )
        return inputs

    def _generation_inputs(
        self,
        plan: TaskPlan,
        questions: list[QuestionVersionDraft],
        mapped: Any,
        refs: dict[str, Any],
        modes: dict[str, str],
        evidence_inputs: list[EvidenceTaskInput],
        *,
        history_content: dict[str, HistoryContentResult] | None = None,
    ) -> list[ConversationTaskInput]:
        evidence_by_key = {item.task_key: item for item in mapped.tasks}
        pool = mapped.evidence_pool
        by_question = {str(question.question_id): question for question in questions}
        input_by_key = {item.task_key: item for item in evidence_inputs}
        result: list[ConversationTaskInput] = []
        for task in plan.tasks:
            question = by_question[str(refs[task.task_key].question_id)]
            content = (history_content or {}).get(task.task_key)
            if content is not None and task.history_content is not None and (
                task.history_content.operation != "followup" or content.status != "matched"
            ):
                result.append(ConversationTaskInput(
                    task_key=task.task_key, question_id=question.question_id,
                    question_version=question.version, question_text=task.objective.text,
                    requirements=self._requirements(task), evidence=[], answer_mode="standard",
                    response_kind=(
                        "history_transform" if task.history_content.operation == "transform"
                        else "history_quote"
                    ),
                    history_text=content.text,
                    history_refs=[content.ref] if content.ref else [],
                    evidence_decision=(
                        "candidates" if content.status == "matched" else content.status
                    ),
                    error_code=content.error_code,
                    missing_inputs=[content.explanation] if content.explanation else [],
                    constraints=[item.text for item in task.constraints],
                ))
                continue
            if modes[task.task_key] in {"ambiguous", "not_found"}:
                result.append(
                    ConversationTaskInput(
                        task_key=task.task_key,
                        question_id=question.question_id,
                        question_version=question.version,
                        question_text=question.question_text,
                        requirements=question.requirements or [question.question_text],
                        evidence=[],
                        evidence_decision=(
                            "ambiguous" if modes[task.task_key] == "ambiguous" else "waiting_input"
                        ),
                        missing_inputs=question.missing_inputs,
                    )
                )
                continue
            evidence = evidence_by_key[task.task_key]
            task_input = input_by_key[task.task_key]
            conditions = {
                f"{obj.kind}.{key}": value
                for obj in task_input.objects
                for key, value in obj.conditions.items()
            }
            conditions.update(question.effective_state.get("task_conditions", {}))
            units = question.effective_state.get("condition_units", {})
            conditions = {
                key: f"{value} {units[key.rsplit('.', 1)[-1]]}"
                if units.get(key.rsplit('.', 1)[-1]) else value
                for key, value in conditions.items()
            }
            if question.effective_state.get("legacy_context"):
                conditions["unstructured_historical_user_context"] = json.dumps(
                    question.effective_state["legacy_context"], ensure_ascii=False
                )
            format_sources = question.effective_state.get("format_source_excerpts", [])
            facts = [fact.model_dump() for fact in evidence.user_facts]
            if task.intent == "format":
                facts.extend(
                    {
                        "field": f"user_supplied_format_source_{index}",
                        "source_excerpt": text,
                        "source_ref": json.dumps(question.effective_state.get("format_source_ref")),
                    }
                    for index, text in enumerate(format_sources)
                )
            result.append(
                ConversationTaskInput.from_evidence(
                    evidence,
                    question.question_text,
                    question.requirements,
                    pool,
                    answer_mode=question.effective_state.get("answer_mode", "standard"),
                    direct_fact_reasoning=(
                        task.intent == "diagnose"
                        or (task.intent == "format" and bool(format_sources))
                        or any(
                            dependency.intent == "diagnose"
                            for dependency in plan.tasks
                            if dependency.task_key in task.dependencies
                        )
                    ),
                    prior_conditions=conditions,
                    user_facts=facts,
                    constraints=[item.text for item in task.constraints],
                    prohibited_actions=list(
                        dict.fromkeys(
                            [
                                *(item.text for item in task.prohibited_actions),
                                *(
                                    requirement
                                    for requirement in question.requirements
                                    if re.search(
                                        r"must not|do not|prohibit|forbid|cannot|禁止|不得|不允许",
                                        requirement,
                                        re.I,
                                    )
                                    and re.search(
                                        r"publish|delet|notif|发布|删除|通知|business api|业务接口",
                                        requirement,
                                        re.I,
                                    )
                                ),
                            ]
                        )
                    ),
                )
            )
            if content is not None:
                result[-1] = result[-1].model_copy(update={
                    "history_text": content.text,
                    "history_refs": [content.ref] if content.ref else [],
                })
        return result

    @staticmethod
    def _apply_results(
        questions: list[QuestionVersionDraft],
        generated: ConversationGenerationResult,
        refs: dict[str, QuestionRef],
        *,
        turn_id: UUID | None = None,
    ) -> list[QuestionVersionDraft]:
        expected = {(ref.question_id, ref.version): key for key, ref in refs.items()}
        actual = {(item.question_id, item.question_version): item for item in generated.items}
        if (
            len(expected) != len(refs)
            or len(actual) != len(generated.items)
            or set(expected) != set(actual)
            or len(questions) != len(refs)
            or any(actual[identity].task_key != key for identity, key in expected.items())
        ):
            raise ConversationWorkflowError(
                "RESULT_MAPPING_CONFLICT", "Generated items do not match planned questions"
            )
        updated: list[QuestionVersionDraft] = []
        spans = answer_item_spans(generated.items)
        for question in questions:
            result = actual[(question.question_id, question.version)]
            answer_hash = (
                hashlib.sha256(result.answer.encode("utf-8")).hexdigest()
                if result.answer and result.status in {"answered", "partial"}
                else None
            )
            state = dict(question.effective_state)
            state.pop("display_answer_span", None)
            state["response_kind"] = result.response_kind
            state["history_refs"] = [ref.model_dump(mode="json") for ref in result.history_refs]
            if turn_id is not None and result.task_key in spans:
                start, end = spans[result.task_key]
                state["display_answer_span"] = {
                    "turn_id": str(turn_id), "start": start, "end": end,
                    "unit": "unicode_codepoints",
                    "sha256": hashlib.sha256(
                        generated.answer[start:end].encode("utf-8")
                    ).hexdigest(),
                }
            updated.append(
                question.model_copy(
                    update={
                        "status": (
                            "waiting_input" if result.status == "ambiguous" else result.status
                        ),
                        "evidence_refs": result.evidence_refs,
                        "answer_hash": answer_hash,
                        "effective_state": state,
                    }
                )
            )
        return updated

    def _usage(
        self,
        plan_result: Any,
        metrics: list[Any],
        started: float,
        *,
        planning_input_upper: int = 0,
        planning_calls: int = 0,
    ) -> UsageMetadata:
        planning_calls = plan_result.llm_calls if plan_result else planning_calls
        metrics = list(metrics)
        if plan_result and not any(item.stage == "planning" for item in metrics):
            metrics = [*plan_result.metrics, *metrics]
        planning_calls = max(0, planning_calls - sum(item.stage == "planning" for item in metrics))
        known_prompt = [item.reported_prompt_tokens for item in metrics]
        known_completion = [item.reported_completion_tokens for item in metrics]
        settled = sum(
            item.reported_prompt_tokens + item.reported_completion_tokens
            for item in metrics
            if item.usage_known
        )
        unknown_reserve = sum(
            item.input_upper_bound + item.max_output_tokens
            for item in metrics
            if not item.usage_known
        )
        return UsageMetadata(
            prompt_tokens=(
                sum(known_prompt)
                if metrics
                and planning_calls == 0
                and all(value is not None for value in known_prompt)
                else None
            ),
            completion_tokens=(
                sum(known_completion)
                if metrics
                and planning_calls == 0
                and all(value is not None for value in known_completion)
                else None
            ),
            estimated_prompt_tokens=(
                planning_calls * planning_input_upper
                + sum(item.input_upper_bound for item in metrics)
            ),
            reserved_output_tokens=(
                planning_calls * self._budget.planning_output
                + sum(item.max_output_tokens for item in metrics)
            ),
            unknown_usage_calls=planning_calls + sum(not item.usage_known for item in metrics),
            settled_tokens=settled,
            accounted_tokens=(
                settled
                + unknown_reserve
                + planning_calls * (planning_input_upper + self._budget.planning_output)
            ),
            llm_calls=planning_calls + len(metrics),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    def _fail(
        self,
        session_id: UUID,
        binding: DatabaseBinding,
        request: RequestRecord,
        *,
        failure_code: str,
        started: float,
        metrics: list[Any] | None = None,
        outcome_unknown: bool = False,
        planning_calls: int = 0,
        planning_input_upper: int = 0,
        failure_details: list[str] | None = None,
        failure_trace: dict[str, Any] | None = None,
    ) -> ConversationTurnResult:
        metric_list = metrics or []
        usage = self._usage(
            None,
            metric_list,
            started,
            planning_calls=planning_calls,
            planning_input_upper=planning_input_upper,
        ).model_copy(update={"failure_code": failure_code, "outcome_unknown": outcome_unknown})
        failed = self._store.fail_request(
            session_id,
            binding,
            request.request_id,
            failure_code=failure_code,
            usage=usage,
            outcome_unknown=outcome_unknown,
        )
        return ConversationTurnResult(
            status="unknown" if outcome_unknown else "failed",
            request=failed,
            revision=request.expected_revision,
            answer="",
            error_code=failure_code,
            trace={
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "planning_input_upper_bound": planning_input_upper,
                **({"validation_errors": failure_details} if failure_details else {}),
                **(failure_trace or {}),
            },
        )

    @staticmethod
    def _replay(request: RequestRecord, status: Literal["succeeded", "failed", "unknown"]):
        return ConversationTurnResult(
            status=status,
            request=request,
            revision=request.turn_sequence or request.expected_revision,
            answer="",
            error_code=request.usage.failure_code,
            answer_not_retained=status == "succeeded",
        )


def _terms(text: str) -> set[str]:
    terms: set[str] = set()
    for part in re.findall(r"[a-z0-9_.]+|[\u3400-\u9fff]+", text.casefold()):
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            terms.update(part[index : index + 2] for index in range(len(part) - 1))
        else:
            terms.add(part)
    return terms
