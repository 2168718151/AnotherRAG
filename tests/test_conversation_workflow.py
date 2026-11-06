from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from openai import APITimeoutError, AuthenticationError

from story_rag.adapters.session_store import PostgresSessionStore
from story_rag.conversation_domain import (
    DatabaseBinding,
    EvidenceRef,
    LedgerError,
    RequestPayload,
)
from story_rag.conversation_workflow import ConversationWorkflow
from story_rag.pipelines.conversation_generation import (
    ConversationAnswerPipeline,
    ConversationGenerationError,
)
from story_rag.pipelines.evidence_mapping import (
    EvidenceBatch,
    EvidencePoolEntry,
    TaskEvidence,
)
from story_rag.pipelines.history_resolution import HistoryResolutionPipeline, HistoryScope
from story_rag.pipelines.task_planning import (
    SourceBackedText,
    TaskPlan,
    TaskPlanningPipeline,
)
from tests.conftest import ScriptedLLM
from tests.test_conversation_generation import BatchLLM
from tests.test_conversation_ledger import _case_messages, postgres_dsn
from tests.test_task_planning import _frozen_plan

pytestmark = pytest.mark.contract
__all__ = ["postgres_dsn"]


class FactsOnlyEvidence:
    def validate_batch(self, _scope, _request, _payload, _inputs, _batch):
        return set()

    def map_evidence(self, scope, request, payload, inputs):
        policy = EvidencePoolEntry(
            evidence_id="phase0-mock-policy",
            ref=EvidenceRef(document_id="phase0-mock-policy", node_id="policy-node"),
            text=(
                "三种写操作控制机制：一，提示词约束只能约束模型意图和输出，"
                "不能授予权限且可能被绕过；二，用户确认表达针对指定动作和"
                "资源的当次意愿，不等于服务端授权，目标或参数变化后确认失效；"
                "三，服务端鉴权强制检查主体、租户、目标资源、动作与执行时"
                "权限，不能替代用户确认。"
                + (
                    "草稿标题变更仅能提交待审批变更单，人工审批平台负责审批和执行，"
                    "最终以平台回执确认执行成功。"
                    if "只能生成待审批的变更单" in payload.message else ""
                )
            ),
            source="方案零结构测试固定证据",
        )
        ticket_record = "现在补充刚才那个独立工单测试环境的调用记录" in payload.message
        return EvidenceBatch(
            session_id=scope.session_id, binding=scope.binding,
            as_of_revision=scope.as_of_revision,
            evidence_pool={} if ticket_record else {policy.evidence_id: policy},
            tasks=[
                TaskEvidence(
                    task_key=item.task_key, question_id=item.question.question_id,
                    question_version=item.question.version,
                    decision=("waiting_input" if item.question.status == "waiting_input"
                              else "no_candidates" if ticket_record else "candidates"),
                    allowed_evidence_ids=(
                        [] if item.question.status == "waiting_input" or ticket_record
                        else [policy.evidence_id]
                    ),
                    user_facts=item.user_facts, current_fact_turn_id=request.turn_id,
                    missing_inputs=item.question.missing_inputs, latency_ms=0,
                )
                for item in inputs
            ],
        )


def _workflow(dsn: str):
    store = PostgresSessionStore(dsn, schema=f"workflow_{uuid4().hex[:12]}")
    assert store.migrate() == 5
    binding = DatabaseBinding(database_id=uuid4(), storage_fingerprint="a" * 64)
    llm = BatchLLM()
    workflow = ConversationWorkflow(
        store=store, planner=TaskPlanningPipeline(ScriptedLLM()),
        history=HistoryResolutionPipeline(store), evidence=FactsOnlyEvidence(),
        generator=ConversationAnswerPipeline(llm),
    )
    return store, binding, llm, workflow


def test_phase0_five_turns_keep_ids_answers_and_scope(postgres_dsn: str) -> None:
    assert ConversationWorkflow._cross_task_output_requirement(
        "Synchronously revise the general permission control scheme together with "
        "the specific business workflow."
    )
    assert not ConversationWorkflow._cross_task_output_requirement(
        "Account for the executor using a shared service account."
    )
    store, binding, llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("方案零连续对话", binding)
    planner = TaskPlanningPipeline(ScriptedLLM())
    messages = _case_messages()
    results = []

    for index, message in enumerate(messages, start=1):
        plan = planner.validate_candidate(message, _frozen_plan(message, index)).plan
        assert plan is not None
        result = workflow.answer_turn(
            session.session_id, binding, request_id=f"phase0-{index}",
            expected_revision=index - 1, payload=RequestPayload(message=message),
            plan=plan, diagnostic=True,
        )
        assert result.status == "succeeded", (index, result.error_code)
        assert result.revision == index
        assert store.get_display_answer(session.session_id, binding, result.request.turn_id) == (
            result.answer
        )
        assert len(result.items) == len(plan.tasks)
        assert result.trace["budget"]["expected_item_count"] == len(plan.tasks)
        assert result.trace["budget"]["produced_item_count"] == len(plan.tasks)
        for budget in result.trace["budget"]["batches"]:
            assert budget["generation_output_headroom"] >= 1536
            assert budget["gate_output_headroom"] >= 2048
            assert budget["generation_context_headroom"] >= 4096
            assert budget["gate_context_headroom"] >= 4096
        assert [item.task_key for item in result.items] == [
            task.task_key for task in plan.tasks
        ]
        assert result.trace["history_index_pending"] is False
        results.append(result)

    assert results[0].items[2].status == "waiting_input"
    for turn in (1, 2):
        assert [item.status for item in results[turn].items] == [
            "answered", "answered"
        ], results[turn].trace["history"]
    assert [item.status for item in results[3].items] == [
        "answered"
    ], results[3].trace["history"]
    assert results[4].items[0].status == "answered"
    assert results[4].items[1].status == "answered"
    assert len(results[4].items) == 2
    assert results[4].items[0].question_id != results[4].items[1].question_id
    assert results[3].items[0].question_id == results[2].items[1].question_id
    assert results[3].items[0].question_id != results[2].items[0].question_id
    assert store.pending_history_index(session.session_id, binding) == 0
    draft_id = results[3].items[0].question_id
    for turn in (3, 4):
        prompt = json.dumps(results[turn - 1].trace["messages"], ensure_ascii=False)
        assert "多租户" in prompt, f"Turn {turn} lost the active tenant revision"
        assert "异步" in prompt, f"Turn {turn} lost queued execution"
        assert "the later goal overrides conflicting earlier conditions" in prompt
        draft = next(
            question for question in store.current_questions_at(
                session.session_id, binding, as_of_revision=turn
            ) if question.question_id == draft_id
        )
        assert draft.effective_state["environment_conditions"]["tenancy"] == "多租户"
        assert "异步" in draft.effective_state["environment_conditions"]["dispatch"]
        assert "；本轮目标：" not in draft.question_text
        assert len(draft.question_text) < 300
        assert "single-tenant application" not in draft.requirements
        message = results[turn - 1].trace["messages"][0][0]["blocks"][0]["text"]
        payload = json.JSONDecoder().raw_decode(
            message[message.index('{"evidence_pool"'):]
        )[0]
        draft_task = next(
            task for task in payload["tasks"]
            if task["question_id"] == str(draft_id)
        )
        assert "shared service account" in draft_task["explicit_required_concepts"]
        assert any(
            "notif" in ban.lower() or "通知" in ban
            for ban in draft_task["prohibited_actions"]
        )
    fifth_prompts = llm.prompts[-2:]
    assert all(term in "\n".join(fifth_prompts) for term in (
        "ticket.read", "ticket.create", "missing_write_scope"
    ))
    assert "missing_write_scope" not in "\n".join(llm.prompts[:2])
    assert "调用记录和错误日志我还没提供" not in "\n".join(fifth_prompts)
    assert "说明需要的调用记录和错误日志" not in "\n".join(fifth_prompts)
    fifth_payload = json.JSONDecoder().raw_decode(
        fifth_prompts[0][fifth_prompts[0].index('{"evidence_pool"'):]
    )[0]
    assert any(
        "missing_write_scope" in fact["source_excerpt"]
        and "ticket.create" in fact["source_excerpt"]
        for row in fifth_payload["tasks"]
        for fact in row["user_provided_facts"]
    )

    replay = workflow.answer_turn(
        session.session_id, binding, request_id="phase0-5",
        expected_revision=4, payload=RequestPayload(message=messages[4]),
    )
    assert replay.status == "succeeded"
    assert replay.answer_not_retained
    assert replay.revision == 5


def test_phase0_no_history_creates_only_new_task_without_reading_history(
    postgres_dsn: str,
) -> None:
    store, binding, _llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("关闭历史", binding)
    message = _case_messages()[0]
    plan = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(
        message, _frozen_plan(message, 1)
    ).plan
    assert plan is not None
    result = workflow.answer_turn(
        session.session_id, binding, request_id="no-history-1",
        expected_revision=0, payload=RequestPayload(message=message, use_history=False),
        plan=plan,
    )
    assert result.status == "succeeded", result.error_code
    assert len(result.items) == 3


def test_phase0_history_topic_alias_must_match_the_same_object() -> None:
    message = _case_messages()[2]
    task = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(
        message, _frozen_plan(message, 3)
    ).plan.tasks[0]
    policy = SimpleNamespace(
        question_text="General Agent write-operation permission control.；任务：Explain controls",
        source_excerpt="Agent 写操作权限怎么控制", requirements=[],
    )
    draft = SimpleNamespace(
        question_text="Internal assistant for the user's draft title；任务：Design workflow",
        source_excerpt="How would you design the approval and execution workflow",
        requirements=[],
    )
    task = task.model_copy(update={
        "object_scope": SourceBackedText(
            text="The draft-title modification workflow",
            source_excerpt="草稿修改流程",
        ),
        "history_reference": SourceBackedText(
            text="General Agent write-operation permission control.",
            source_excerpt="草稿修改流程",
        ),
    })
    assert ConversationWorkflow._unique_history_candidate(task, [policy, draft]) is draft
    task.history_reference.text = "Internal assistant for the user's draft title"
    assert ConversationWorkflow._unique_history_candidate(task, [policy, draft]) is draft

    fourth_message = _case_messages()[3]
    fourth = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(
        fourth_message, _frozen_plan(fourth_message, 4)
    ).plan.tasks[0]
    first = _frozen_plan(_case_messages()[0], 1)["tasks"]
    policy.question_text = first[0]["objective"]["text"]
    policy.source_excerpt = fourth_message
    draft.question_text = first[1]["objective"]["text"]
    draft.source_excerpt = first[1]["objective"]["source_excerpt"]
    assert ConversationWorkflow._unique_history_candidate(
        fourth, [policy, draft]
    ) is draft
    policy.question_text = (
        "General Agent write-operation permission control "
        "(not tied to the specific draft-title workflow).；任务：Explain controls"
    )
    draft.question_text = (
        "Internal assistant in a single-tenant application, synchronous requests "
        "within the authenticated user's session, allowed to change only the current "
        "user's draft title.；任务：Design workflow"
    )
    fourth = fourth.model_copy(update={
        "object_scope": SourceBackedText(
            text=(
                "The specific draft-title modification business workflow, "
                "not the general principle"
            ),
            source_excerpt="只调整草稿业务的实现流程",
        ),
        "history_reference": SourceBackedText(
            text=ConversationWorkflow._catalog_topic(draft.question_text),
            source_excerpt="刚才那个修改草稿标题的业务",
        ),
    })
    assert ConversationWorkflow._unique_history_candidate(fourth, [policy, draft]) is draft
    draft.question_no = 2
    later_draft = SimpleNamespace(
        question_text=draft.question_text,
        source_excerpt="修改草稿流程",
        requirements=[], question_no=4,
    )
    assert ConversationWorkflow._unique_history_candidate(
        fourth, [policy, draft, later_draft]
    ) is later_draft
    fifth_message = _case_messages()[4]
    fifth = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(
        fifth_message, _frozen_plan(fifth_message, 5)
    ).plan.tasks[1].model_copy(update={
        "object_scope": SourceBackedText(
            text="independent ticket environment ticket.create authorization handling",
            source_excerpt="独立工单环境",
        ),
        "history_reference": SourceBackedText(
            text="General Agent write-operation permission control",
            source_excerpt="权限控制",
        ),
    })
    ticket = SimpleNamespace(
        question_text="Ticket submission failure in independent test environment",
        source_excerpt="独立工单环境", requirements=[], question_no=3,
    )
    assert ConversationWorkflow._unique_history_candidate(
        fifth, [policy, draft, ticket]
    ) is ticket


def test_phase0_ticket_record_replaces_old_diagnostic_input_gap() -> None:
    fifth_message = _case_messages()[4]
    fifth = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(
        fifth_message, _frozen_plan(fifth_message, 5)
    ).plan.tasks[0]
    old = SimpleNamespace(missing_inputs=[
        "Call records for the failed ticket submission were not provided",
        "Error logs for the failed ticket submission were not provided",
    ])
    resolved = ConversationWorkflow._resolved_gaps(old, fifth, fifth_message)
    assert len(resolved) == len(old.missing_inputs)
    assert all(gap.source_excerpt in fifth_message for gap in resolved)
    assert not ConversationWorkflow._resolved_gaps(
        old, fifth.model_copy(update={"constraints": []}), fifth_message
    )


@pytest.mark.parametrize("intent", ["answer", "clarify"])
@pytest.mark.parametrize("verb", ["Acknowledge and preserve", "Record and preserve"])
def test_phase0_preserve_draft_scope_is_not_an_extra_answer(
    intent: str, verb: str,
) -> None:
    message = _case_messages()[4]
    source = message[message.index("不改变"):]
    plan_data = _frozen_plan(message, 5)
    plan_data["goals"].append({
        "goal_key": f"g{len(plan_data['goals']) + 1}",
        "goal_text": "Preserve the draft approval rule",
        "source_excerpt": source, "task_keys": ["t3"],
    })
    plan_data["tasks"].append({
        "task_key": "t3", "intent": intent,
        "objective": {"text": f"{verb} the draft approval rule",
                      "source_excerpt": source},
        "constraints": [{"text": "Do not alter the draft approval rule",
                         "source_excerpt": source}],
    })
    plan = TaskPlan.model_validate(plan_data)
    folded = ConversationWorkflow._fold_constraint_only_goals(plan)
    assert [task.task_key for task in folded.tasks] == ["t1", "t2"]
    assert len(folded.goals) == len(plan_data["goals"]) - 1
    assert all(source in [item.source_excerpt for item in task.constraints]
               for task in folded.tasks)


def test_phase0_explain_then_revise_same_history_object_is_one_task(
    postgres_dsn: str,
) -> None:
    store, binding, _llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("方案零去重", binding)
    messages = _case_messages()
    planner = TaskPlanningPipeline(ScriptedLLM())
    for index in (1, 2):
        plan = planner.validate_candidate(
            messages[index - 1], _frozen_plan(messages[index - 1], index)
        ).plan
        assert plan is not None
        result = workflow.answer_turn(
            session.session_id, binding, request_id=f"dedup-{index}",
            expected_revision=index - 1,
            payload=RequestPayload(message=messages[index - 1]), plan=plan,
        )
        assert result.status == "succeeded"

    plan = planner.validate_candidate(messages[2], _frozen_plan(messages[2], 3)).plan
    assert plan is not None
    plan = plan.model_copy(deep=True)
    for task in plan.tasks:
        task.intent = "answer"
    extras = []
    for index, task in enumerate(plan.tasks, start=3):
        duplicate = task.model_copy(deep=True)
        duplicate.task_key = f"t{index}"
        duplicate.intent = "revise"
        duplicate.dependencies = [task.task_key]
        extras.append(duplicate)
        for goal in plan.goals:
            if task.task_key in goal.task_keys:
                goal.task_keys.append(duplicate.task_key)
    plan.tasks.extend(extras)
    plan = TaskPlan.model_validate(plan.model_dump())
    scope = HistoryScope(
        session_id=session.session_id, binding=binding, as_of_revision=2,
    )
    normalized = workflow._coalesce_explanation_revisions(scope, plan)
    assert [task.task_key for task in normalized.tasks] == ["t1", "t2"]
    assert all(task.intent == "revise" for task in normalized.tasks)
    assert all(goal.task_keys for goal in normalized.goals)
    assert all(set(goal.task_keys) <= {"t1", "t2"} for goal in normalized.goals)
    no_dependencies = plan.model_copy(deep=True)
    for task in no_dependencies.tasks:
        if task.intent == "revise":
            task.dependencies = []
    normalized_without_advisory_links = workflow._coalesce_explanation_revisions(
        scope, no_dependencies
    )
    assert [task.task_key for task in normalized_without_advisory_links.tasks] == [
        "t1", "t2"
    ]


def test_phase0_stale_evidence_after_generation_affects_only_one_task(
    postgres_dsn: str,
) -> None:
    store, binding, _llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("生成后证据复验", binding)
    message = _case_messages()[0]
    plan = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(
        message, _frozen_plan(message, 1)
    ).plan
    assert plan is not None

    class StaleOneEvidence(FactsOnlyEvidence):
        def validate_batch(self, scope, request, payload, inputs, batch):
            return {inputs[0].task_key}

    workflow._evidence = StaleOneEvidence()
    result = workflow.answer_turn(
        session.session_id, binding, request_id="stale-one",
        expected_revision=0, payload=RequestPayload(message=message), plan=plan,
    )
    assert result.status == "succeeded"
    assert [item.status for item in result.items] == [
        "needs_refresh", "answered", "waiting_input"
    ]
    assert result.items[0].answer == ""
    assert not result.items[0].evidence_refs
    assert "本轮任务 t1 已回答" not in result.answer
    assert "本轮任务 t2 已回答" in result.answer
    assert [question.status for question in store.current_questions_at(
        session.session_id, binding, as_of_revision=1
    )] == ["needs_refresh", "answered", "waiting_input"]


def test_phase0_deterministic_failure_does_not_commit_or_repeat(
    postgres_dsn: str,
) -> None:
    store, binding, llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("确定失败恢复", binding)
    message = _case_messages()[0]
    plan = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(
        message, _frozen_plan(message, 1)
    ).plan
    assert plan is not None
    payload = RequestPayload(message=message)
    good = workflow._generator

    class FailedGenerator:
        def answer(self, _tasks):
            raise ConversationGenerationError("INVALID_GENERATED_ITEM", "Mock failure")

    workflow._generator = FailedGenerator()
    failed = workflow.answer_turn(
        session.session_id, binding, request_id="failure", expected_revision=0,
        payload=payload, plan=plan,
    )
    assert failed.status == "failed"
    assert failed.revision == 0
    assert store.get_session(session.session_id, binding).revision == 0
    workflow._generator = good
    replay = workflow.answer_turn(
        session.session_id, binding, request_id="failure", expected_revision=0,
        payload=payload, plan=plan,
    )
    assert replay.status == "failed"
    assert not llm.calls
    recovered = workflow.answer_turn(
        session.session_id, binding, request_id="new-request", expected_revision=0,
        payload=payload, plan=plan,
    )
    assert recovered.status == "succeeded"
    assert recovered.revision == 1


def test_phase0_all_generated_items_failed_does_not_commit(postgres_dsn: str) -> None:
    store, binding, llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("全题技术失败", binding)
    message = _case_messages()[0]
    plan = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(
        message, _frozen_plan(message, 1)
    ).plan
    assert plan is not None
    llm.mode = "always_truncated"
    result = workflow.answer_turn(
        session.session_id, binding, request_id="all-truncated",
        expected_revision=0, payload=RequestPayload(message=message), plan=plan,
        diagnostic=True,
    )
    assert result.status == "failed"
    assert result.trace["generation_failures"]
    assert [item["status"] for item in result.trace["generation_failures"]] == [
        "technical_failure", "technical_failure", "waiting_input"
    ]
    assert result.trace["generation_budget"]["expected_item_count"] == len(plan.tasks)
    assert result.trace["generation_metrics"]
    assert result.revision == 0
    assert store.get_session(session.session_id, binding).revision == 0
    assert store.current_questions_at(
        session.session_id, binding, as_of_revision=0
    ) == []


def test_phase0_index_lag_recovers_without_regenerating(
    postgres_dsn: str, monkeypatch,
) -> None:
    store, binding, llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("索引滞后恢复", binding)
    message = _case_messages()[0]
    plan = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(
        message, _frozen_plan(message, 1)
    ).plan
    assert plan is not None
    original = store.process_history_index
    monkeypatch.setattr(store, "process_history_index", lambda *_args, **_kwargs: 1 / 0)
    result = workflow.answer_turn(
        session.session_id, binding, request_id="index-lag", expected_revision=0,
        payload=RequestPayload(message=message), plan=plan,
    )
    assert result.status == "succeeded"
    assert result.trace["history_index_pending"] is True
    assert store.pending_history_index(session.session_id, binding) == 1
    calls = len(llm.calls)
    monkeypatch.setattr(store, "process_history_index", original)
    assert store.process_history_index(session.session_id, binding) == 1
    assert store.process_history_index(session.session_id, binding) == 0
    assert store.pending_history_index(session.session_id, binding) == 0
    replay = workflow.answer_turn(
        session.session_id, binding, request_id="index-lag", expected_revision=0,
        payload=RequestPayload(message=message), plan=plan,
    )
    assert replay.status == "succeeded"
    assert len(llm.calls) == calls


@pytest.mark.parametrize("failure", ["authentication", "timeout"])
def test_planning_provider_failure_classification_and_retry_boundary(
    postgres_dsn: str, caplog, failure: str,
) -> None:
    store, binding, generator_llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("规划鉴权失败分类", binding)
    message = _case_messages()[0]
    private_details = "unit-test-private-provider-details"
    request = httpx.Request("POST", "https://model.invalid/chat/completions")
    error = (
        AuthenticationError(
            private_details, response=httpx.Response(401, request=request),
            body={"message": private_details},
        )
        if failure == "authentication" else APITimeoutError(request=request)
    )

    class RejectedLLM(ScriptedLLM):
        def complete(self, prompt, formatted=False, **kwargs):
            self.prompts.append(prompt)
            raise error

    planner_llm = RejectedLLM()
    workflow._planner = TaskPlanningPipeline(planner_llm)
    payload = RequestPayload(message=message)
    result = workflow.answer_turn(
        session.session_id, binding, request_id="provider-failure",
        expected_revision=0, payload=payload,
    )
    is_unknown = failure == "timeout"
    assert result.status == ("unknown" if is_unknown else "failed")
    assert result.error_code == (
        "PLANNING_CALL_UNKNOWN" if is_unknown else "LLM_AUTHENTICATION_FAILED"
    )
    assert result.request.usage.outcome_unknown is is_unknown
    assert result.request.usage.llm_calls == 1
    assert result.answer == "" and result.items == []
    assert result.revision == 0
    assert store.get_session(session.session_id, binding).revision == 0
    assert len(planner_llm.prompts) == 1
    assert not generator_llm.calls
    assert private_details not in caplog.text
    assert private_details not in result.model_dump_json()

    replay = workflow.answer_turn(
        session.session_id, binding, request_id="provider-failure",
        expected_revision=0, payload=payload,
    )
    assert replay.status == result.status
    assert len(planner_llm.prompts) == 1
    plan = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(
        message, _frozen_plan(message, 1),
    ).plan
    assert plan is not None
    if is_unknown:
        with pytest.raises(LedgerError) as blocked:
            workflow.answer_turn(
                session.session_id, binding, request_id="next",
                expected_revision=0, payload=payload, plan=plan,
            )
        assert blocked.value.code == "SESSION_UNCERTAIN"
    else:
        retried = workflow.answer_turn(
            session.session_id, binding, request_id="next",
            expected_revision=0, payload=payload, plan=plan,
        )
        assert retried.status == "succeeded"
        assert retried.revision == 1


def test_phase0_unknown_needs_explicit_recovery_before_new_request(
    postgres_dsn: str,
) -> None:
    store, binding, llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("未知结果保护", binding)
    message = _case_messages()[0]
    plan = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(
        message, _frozen_plan(message, 1)
    ).plan
    assert plan is not None
    payload = RequestPayload(message=message)
    good = workflow._generator

    class UnknownGenerator:
        def answer(self, _tasks):
            raise ConversationGenerationError(
                "MODEL_CALL_UNKNOWN", "Remote outcome unknown", outcome_unknown=True
            )

    workflow._generator = UnknownGenerator()
    unknown = workflow.answer_turn(
        session.session_id, binding, request_id="unknown", expected_revision=0,
        payload=payload, plan=plan,
    )
    assert unknown.status == "unknown"
    assert store.get_session(session.session_id, binding).revision == 0
    workflow._generator = good
    replay = workflow.answer_turn(
        session.session_id, binding, request_id="unknown", expected_revision=0,
        payload=payload, plan=plan,
    )
    assert replay.status == "unknown"
    assert not llm.calls
    with pytest.raises(LedgerError, match="unknown request") as blocked:
        workflow.answer_turn(
            session.session_id, binding, request_id="next", expected_revision=0,
            payload=payload, plan=plan,
        )
    assert blocked.value.code == "SESSION_UNCERTAIN"
    with pytest.raises(LedgerError, match="acknowledged"):
        store.acknowledge_unknown_request(
            session.session_id, binding, "unknown", acknowledge_result_unknown=False
        )
    acknowledged = store.acknowledge_unknown_request(
        session.session_id, binding, "unknown", acknowledge_result_unknown=True
    )
    assert acknowledged.status == "failed"
    assert acknowledged.usage.outcome_unknown
    recovered = workflow.answer_turn(
        session.session_id, binding, request_id="next", expected_revision=0,
        payload=payload, plan=plan,
    )
    assert recovered.status == "succeeded"


def test_phase0_native_workflow_entry_runs_the_same_turn(postgres_dsn: str) -> None:
    store, binding, _llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("原生工作流", binding)
    message = _case_messages()[0]
    workflow._planner = TaskPlanningPipeline(ScriptedLLM(responses=[
        json.dumps(_frozen_plan(message, 1), ensure_ascii=False)
    ]))
    result = asyncio.run(workflow.answer(
        session.session_id, binding, request_id="native", expected_revision=0,
        payload=RequestPayload(message=message),
    ))
    assert result.status == "succeeded", result.error_code
    assert result.revision == 1
    assert len(result.items) == 3
    assert result.request.usage.llm_calls >= 1
    assert result.request.usage.unknown_usage_calls >= 1
    assert result.request.usage.accounted_tokens >= result.request.usage.settled_tokens


def test_phase0_cancelled_caller_waits_for_workflow_cleanup(postgres_dsn: str) -> None:
    store, binding, _llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("取消收尾", binding)
    message = _case_messages()[0]
    workflow._planner = TaskPlanningPipeline(ScriptedLLM(responses=[
        json.dumps(_frozen_plan(message, 1), ensure_ascii=False)
    ]))
    entered = threading.Event()
    released = threading.Event()
    generator = workflow._generator

    class SlowGenerator:
        def answer(self, tasks):
            entered.set()
            assert released.wait(5)
            return generator.answer(tasks)

    workflow._generator = SlowGenerator()

    async def run_and_cancel() -> None:
        running = asyncio.create_task(workflow.answer(
            session.session_id, binding, request_id="cancelled-caller",
            expected_revision=0, payload=RequestPayload(message=message),
        ))
        assert await asyncio.to_thread(entered.wait, 5)
        running.cancel()
        released.set()
        with pytest.raises(asyncio.CancelledError):
            await running

    asyncio.run(run_and_cancel())
    assert store.get_session(session.session_id, binding).revision == 1
    assert store.get_request(session.session_id, binding, "cancelled-caller").status == "succeeded"
