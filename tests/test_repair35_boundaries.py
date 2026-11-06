from __future__ import annotations

from uuid import uuid4

import pytest

from story_rag.conversation_domain import DatabaseBinding, LedgerError, RequestPayload
from story_rag.pipelines.task_planning import TaskPlan
from tests.test_conversation_ledger import postgres_dsn
from tests.test_conversation_repairs import plan
from tests.test_conversation_workflow import _workflow
from tests.test_databases import databases, opened

__all__ = ["postgres_dsn", "databases"]


def combined(*plans):
    tasks, goals = [], []
    for index, candidate in enumerate(plans, 1):
        key = f"t{index}"
        tasks.append(candidate.tasks[0].model_copy(update={"task_key": key}))
        goals.append(
            candidate.goals[0].model_copy(update={"goal_key": f"g{index}", "task_keys": [key]})
        )
    return TaskPlan(schema_version=2, tasks=tasks, goals=goals)


def condition(field, value, quote, op="supplement"):
    return {"field": field, "value": value, "source_excerpt": quote, "op": op}


def turn(workflow, session, binding, revision, message, candidate, target=None):
    result = workflow.answer_turn(
        session.session_id,
        binding,
        request_id=f"boundary-{revision}",
        expected_revision=revision,
        plan=candidate,
        diagnostic=True,
        payload=RequestPayload(message=message, reply_to_question_id=target),
    )
    assert result.status == "succeeded", result.model_dump()
    return result


def test_G01_G06_shared_training_state_and_independent_inference(postgres_dsn):
    store, binding, _, workflow = _workflow(postgres_dsn)
    session = store.create_session("shared training", binding)
    message = "训练方案A参数70亿，梯度BF16，8卡，分别解释原理和计算；推理方案B的batch=4。"
    base = plan(
        message,
        objective="训练方案A原理",
        alias="training_a",
        conditions=[
            condition("parameter_count", "70亿", message),
            condition("gradient_dtype", "BF16", message),
            condition("data_parallel_size", "8", message),
        ],
    )
    calculation = plan(
        message,
        objective="训练方案A计算",
        alias="training_a",
        conditions=[condition("gradient_dtype", "BF16", message)],
    )
    inference = plan(
        message,
        objective="推理方案B缓存",
        alias="inference_b",
        conditions=[condition("batch_size", "4", message)],
    )
    first = turn(workflow, session, binding, 0, message, combined(base, calculation, inference))
    rows = store.current_questions_at(session.session_id, binding, as_of_revision=1)
    assert rows[0].object_refs == rows[1].object_refs
    assert rows[2].object_refs != rows[0].object_refs
    objects = store.list_object_versions(session.session_id, binding)
    assert len(objects) == 2
    message = "训练方案A梯度改为FP32，原理与计算同步修订，推理方案B保持不变。"
    revise = plan(
        message,
        objective="训练方案A计算",
        alias="training_a",
        history="训练方案A计算",
        conditions=[condition("gradient_dtype", "FP32", message, "replace")],
    )
    turn(workflow, session, binding, 1, message, revise, first.items[1].question_id)
    latest = store.list_object_versions(session.session_id, binding)
    assert len(latest) == 3
    training = [obj for obj in latest if obj.kind == "training_a"]
    assert training[-1].conditions == {
        "parameter_count": "70亿",
        "gradient_dtype": "FP32",
        "data_parallel_size": "8",
    }
    assert [obj for obj in latest if obj.kind == "inference_b"][0].conditions == {"batch_size": "4"}


def test_C05_C06_inheritance_and_comparison_preserve_original_sources(postgres_dsn):
    store, binding, _, workflow = _workflow(postgres_dsn)
    session = store.create_session("branch", binding)
    message = "训练方案A参数70亿，梯度BF16，8卡。"
    first = turn(
        workflow,
        session,
        binding,
        0,
        message,
        plan(
            message,
            alias="training_a",
            conditions=[
                condition("parameter_count", "70亿", message),
                condition("gradient_dtype", "BF16", message),
                condition("data_parallel_size", "8", message),
            ],
        ),
    )
    original = store.list_object_versions(session.session_id, binding)[0]
    message = "训练方案A参数量和卡数保持不变，继续解释。"
    turn(
        workflow,
        session,
        binding,
        1,
        message,
        plan(
            message,
            alias="training_a",
            history="training_a",
            conditions=[
                condition("parameter_count", None, message, "inherit"),
                condition("data_parallel_size", None, message, "inherit"),
            ],
        ),
        first.items[0].question_id,
    )
    assert store.list_object_versions(session.session_id, binding) == [original]
    message = "另外比较FP32梯度，但不修改现在的BF16方案。"
    candidate = plan(
        message,
        alias="training_a",
        history="training_a",
        conditions=[condition("gradient_dtype", "FP32", message, "replace")],
    )
    candidate.tasks[0].intent = "compare"
    result = turn(workflow, session, binding, 2, message, candidate, first.items[0].question_id)
    assert result.items[0].question_id != first.items[0].question_id
    objects = store.list_object_versions(session.session_id, binding)
    assert len(objects) == 2
    branch = next(obj for obj in objects if obj.object_id != original.object_id)
    assert branch.derived_from.object_id == original.object_id
    assert branch.conditions["gradient_dtype"] == "FP32"
    assert original.conditions["gradient_dtype"] == "BF16"
    assert original.changes[0].source_turn_id == first.request.turn_id


@pytest.mark.parametrize("calculation", [False, True])
def test_H03_gap_resolution_obeys_calculation_boundary(postgres_dsn, calculation):
    store, binding, llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("gaps", binding)
    message = (
        "计算GQA，层数与长度均尚未提供。" if calculation
        else "定位工具调用失败，调用记录和错误日志均未提供。"
    )
    fields = ["层数", "长度"] if calculation else ["调用记录", "错误日志"]
    candidate = plan(
        message,
        alias="inference",
        gaps=[
            {"description": field, "state": "explicitly_missing", "basis_excerpt": message}
            for field in fields
        ],
    )
    first = turn(workflow, session, binding, 0, message, candidate)
    assert llm.calls == []
    message = "层数32，长度稍后补。" if calculation else "调用记录已提供，错误日志稍后补。"
    second_plan = plan(message, alias="inference", history="inference")
    from story_rag.pipelines.task_planning import PlannedGapResolution

    second_plan.tasks[0].resolved_inputs = [
        PlannedGapResolution(missing_input=fields[0], source_excerpt=message)
    ]
    second = turn(workflow, session, binding, 1, message, second_plan, first.items[0].question_id)
    assert second.items[0].status == ("no_evidence" if calculation else "waiting_input")
    assert second.items[0].missing_inputs == ([] if calculation else [fields[1]])
    assert llm.calls == []


@pytest.mark.parametrize("different_database", [False, True])
def test_H06_real_foreign_ids_are_rejected_without_model_calls(postgres_dsn, different_database):
    store, binding, llm, workflow = _workflow(postgres_dsn)
    foreign_binding = (
        DatabaseBinding(database_id=uuid4(), storage_fingerprint="c" * 64)
        if different_database
        else binding
    )
    foreign = store.create_session("foreign", foreign_binding)
    first = turn(workflow, foreign, foreign_binding, 0, "解释AOF", plan("解释AOF"))
    own = store.create_session("own", binding)
    before = len(llm.calls)
    result = workflow.answer_turn(
        own.session_id,
        binding,
        request_id="foreign-ref",
        expected_revision=0,
        payload=RequestPayload(message="继续", reply_to_question_id=first.items[0].question_id),
    )
    assert result.error_code == "INVALID_HISTORY_REFERENCE"
    assert result.revision == 0 and len(llm.calls) == before
    assert result.request.usage.llm_calls == 0


def test_O06_exact_edit_requires_original_body(postgres_dsn):
    store, binding, llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("exact", binding)
    first = turn(workflow, session, binding, 0, "解释AOF", plan("解释AOF"))
    before = len(llm.calls)
    message = "逐字修改上一条回答第二点的最后三个字"
    candidate = plan(message, history="AOF")
    candidate.tasks[0].intent = "format"
    candidate.tasks[0].edit_mode = "exact_edit"
    result = turn(workflow, session, binding, 1, message, candidate, first.items[0].question_id)
    assert result.items[0].status == "waiting_input" and "原文" in result.answer
    assert len(llm.calls) == before


def test_R03_recreated_same_name_database_cannot_bind_old_session(postgres_dsn, databases):
    store, _, _, _ = _workflow(postgres_dsn)
    record = databases.create("repair35-isolated")
    with opened(databases, record) as storage:
        old_binding = DatabaseBinding(
            database_id=record.db_id, storage_fingerprint=storage.database_id
        )
    session = store.create_session("old", old_binding)
    databases.drop(record, confirmation=record.db_id)
    recreated = databases.create("repair35-isolated")
    with opened(databases, recreated) as storage:
        new_binding = DatabaseBinding(
            database_id=recreated.db_id, storage_fingerprint=storage.database_id
        )
    assert old_binding.database_id != new_binding.database_id
    with pytest.raises(LedgerError):
        store.get_session(session.session_id, new_binding)
    fresh = store.create_session("fresh", new_binding)
    assert store.get_session(fresh.session_id, new_binding).revision == 0


def test_shared_object_is_attached_when_field_owner_comes_last(postgres_dsn):
    store, binding, _, workflow = _workflow(postgres_dsn)
    session = store.create_session("shared ordering", binding)
    message = "训练方案A的原理和计算共享参数：梯度FP32；推理配置稍后补。"
    candidate = combined(
        plan(message, objective="训练方案A原理", alias="training"),
        plan(
            message,
            objective="训练方案A计算",
            alias="training",
            conditions=[condition("gradient_dtype", "FP32", message)],
        ),
        plan(message, objective="推理缓存", alias="inference"),
    )
    turn(workflow, session, binding, 0, message, candidate)
    rows = store.current_questions_at(session.session_id, binding, as_of_revision=1)
    assert rows[0].object_refs and rows[0].object_refs == rows[1].object_refs
    assert rows[2].object_refs and rows[2].object_refs != rows[0].object_refs


def test_proposed_field_operation_uses_authoritative_existing_fields(postgres_dsn):
    store, binding, _, workflow = _workflow(postgres_dsn)
    session = store.create_session("field binding", binding)
    first = turn(workflow, session, binding, 0, "草稿助手", plan("草稿助手", alias="draft"))
    message = "助手执行方式改成审批平台"
    candidate = plan(
        message,
        alias="draft",
        history="draft",
        conditions=[condition("execution_mode", "审批平台", message, "replace")],
    )
    turn(workflow, session, binding, 1, message, candidate, first.items[0].question_id)
    rows = store.list_object_versions(session.session_id, binding)
    assert rows[-1].changes[0].op == "supplement"
    assert rows[-1].changes[0].before is None
    assert rows[-1].conditions == {"execution_mode": "审批平台"}


def test_R04_equal_text_near_scores_keep_document_identity(monkeypatch):
    from tests.test_conversation_ledger import _case_messages
    from tests.test_evidence_mapping import _inputs, _request, _runtime

    session, _, storage, docs, retriever, mapper = _runtime()
    body = "服务端鉴权独立于提示词和用户确认。"
    first = storage.add("allowed.txt", body)
    second = storage.add("forbidden.txt", body)
    original_retrieve = retriever.retrieve

    def nearby_scores(request):
        result = original_retrieve(request)
        for index, node in enumerate(result.nodes):
            node.score = node.rerank_score = 0.5 + index * 1e-9
        return result

    monkeypatch.setattr(retriever, "retrieve", nearby_scores)
    message = _case_messages()[0]
    scope, request, payload = _request(session, message)
    task = _inputs(1, message, docs, {})[0]
    task.allowed_document_ids = [first]
    result = mapper.map_evidence(scope, request, payload, [task])
    entries = [result.evidence_pool[eid] for eid in result.tasks[0].allowed_evidence_ids]
    assert len(entries) == 1 and entries[0].ref.document_id == first
    assert entries[0].ref.document_id != second
    original_id = entries[0].ref.node_id
    storage.docstore.docs[first].metadata["content_hash"] = "f" * 64
    stale = mapper.map_evidence(scope, request, payload, [task])
    assert stale.tasks[0].decision == "needs_refresh"
    assert original_id in stale.tasks[0].stale_candidate_ids
    assert stale.tasks[0].allowed_evidence_ids == []


def test_R01_deleted_point_invalidates_saved_evidence():
    from tests.test_conversation_ledger import _case_messages
    from tests.test_evidence_mapping import _inputs, _request, _runtime

    session, _, storage, docs, _, mapper = _runtime()
    message = _case_messages()[0]
    scope, request, payload = _request(session, message)
    task = _inputs(1, message, docs, {})[0]
    task.allowed_document_ids = [docs["policy"]]
    result = mapper.map_evidence(scope, request, payload, [task])
    assert result.tasks[0].allowed_evidence_ids
    storage.points[docs["policy"]].clear()
    assert task.task_key in mapper.validate_batch(scope, request, payload, [task], result)


def test_E03_E05_O05_partial_and_literal_none_do_not_lose_status():
    from story_rag.pipelines.conversation_generation import (
        ConversationAnswerPipeline,
        GateItem,
        GeneratedItem,
        merge_answer_items,
    )
    from tests.test_conversation_generation import _tasks

    task = _tasks(1)[0].model_copy(update={"requirements": ["优点", "缺点"]})
    candidate = GeneratedItem(
        task_key=task.task_key,
        question_id=task.question_id,
        question_version=1,
        answer="有依据的优点；缺点暂无依据。",
        evidence_ids=[task.evidence[0].evidence_id],
        answerability="partial",
    )
    gate = GateItem(
        task_key=task.task_key,
        question_id=task.question_id,
        question_version=1,
        scope_ok=True,
        evidence_supported=True,
        answer_relevant=True,
        covered_requirement_indexes=[0],
        missing_requirement_indexes=[1],
        evidence_missing_requirement_indexes=[1],
        answerability="partial",
    )
    result = ConversationAnswerPipeline._decide(task, candidate, gate)
    assert result.status == "partial" and result.uncovered_requirements == ["缺点"]
    assert "未覆盖要求：缺点" in merge_answer_items([result])
    candidate.answer = "无"
    candidate.answerability = gate.answerability = "full"
    gate.covered_requirement_indexes = [0, 1]
    gate.missing_requirement_indexes = gate.evidence_missing_requirement_indexes = []
    assert ConversationAnswerPipeline._decide(task, candidate, gate).status == "answered"


def test_diagnostic_disk_failure_does_not_discard_received_response(tmp_path):
    from story_rag.adapters.conversation_diagnostics import (
        ModelCallContext,
        after_model_call,
        model_call_context,
    )

    class BrokenArchive:
        def write(self, *args):
            raise OSError("disk full")

    token = model_call_context.set(
        ModelCallContext(before_call=lambda stage: None, archive=BrokenArchive())
    )
    try:
        after_model_call("generation", {"answer": "known response"})
    finally:
        model_call_context.reset(token)


def test_initial_absence_is_not_a_deletion_of_nonexistent_history(postgres_dsn):
    store, binding, _, workflow = _workflow(postgres_dsn)
    session = store.create_session("initial absence", binding)
    message = "初始参数FP32，没有额外master copy。"
    result = turn(
        workflow,
        session,
        binding,
        0,
        message,
        plan(
            message,
            alias="training",
            conditions=[
                condition("parameter_dtype", "FP32", message),
                condition("master_dtype", None, message, "remove"),
            ],
        ),
    )
    objects = store.list_object_versions(session.session_id, binding)
    assert result.revision == 1 and "master_dtype" not in objects[0].conditions
    assert [change.field for change in objects[0].changes] == ["parameter_dtype"]
    questions = store.current_questions_at(session.session_id, binding, as_of_revision=1)
    assert "master_dtype" in questions[0].effective_state["absent_fields"]


def test_v3_ledger_upgrade_preserves_existing_ids_and_revision(postgres_dsn):
    store, binding, _, workflow = _workflow(postgres_dsn)
    session = store.create_session("upgrade", binding)
    result = turn(workflow, session, binding, 0, "解释AOF", plan("解释AOF"))
    before = store.current_questions_at(session.session_id, binding, as_of_revision=1)
    # Only this random test schema is reduced to the old v3 layout.
    with store._connect() as conn:
        conn.execute("DROP TABLE conversation_display_turns")
        conn.execute("DROP INDEX turn_requests_display_page_idx")
        conn.execute("ALTER TABLE question_versions DROP COLUMN effective_state")
        conn.execute("ALTER TABLE turn_requests DROP COLUMN execution_owner")
        conn.execute("DELETE FROM schema_migrations WHERE version IN (4, 5)")
    assert store.migrate() == 5
    after = store.current_questions_at(session.session_id, binding, as_of_revision=1)
    assert after[0].question_id == before[0].question_id == result.items[0].question_id
    assert after[0].question_text == before[0].question_text
    assert after[0].effective_state == {}
    assert store.get_session(session.session_id, binding).revision == 1


def test_B04_dispatch_persists_conservative_reservation_until_settlement(postgres_dsn):
    store, binding, _, _ = _workflow(postgres_dsn)
    session = store.create_session("inflight usage", binding)
    store.claim_request(
        session.session_id,
        binding,
        request_id="usage",
        expected_revision=0,
        payload=RequestPayload(message="test"),
    )
    store.start_request(session.session_id, binding, "usage")
    store.mark_model_attempt(session.session_id, binding, "usage", "planning")
    request = store.get_request(session.session_id, binding, "usage")
    assert request.usage.llm_calls == request.usage.unknown_usage_calls == 1
    assert request.usage.accounted_tokens > 0
    assert request.usage.prompt_tokens is None and request.usage.completion_tokens is None


def test_D05_shared_prerequisite_does_not_duplicate_training_revision(postgres_dsn):
    store, binding, _, workflow = _workflow(postgres_dsn)
    session = store.create_session("overlapping goals", binding)
    message = "训练原理与训练计算均使用BF16梯度。"
    first = combined(
        plan(
            message,
            objective="训练原理",
            alias="principle",
            conditions=[condition("gradient_dtype", "BF16", message)],
        ),
        plan(
            message,
            objective="训练计算",
            alias="training",
            conditions=[condition("gradient_dtype", "BF16", message)],
        ),
    )
    turn(workflow, session, binding, 0, message, first)
    message = "梯度改FP32；统一原理公式后给出训练分项、总计，并同步修订训练计算。"
    candidate = combined(
        plan(
            message,
            objective="给出训练分项和总计",
            alias="training",
            history="训练计算",
            conditions=[condition("gradient_dtype", "FP32", message, "replace")],
        ),
        plan(
            message,
            objective="修订原理公式",
            alias="principle",
            history="训练原理",
            conditions=[condition("gradient_dtype", "FP32", message, "replace")],
        ),
        plan(
            message,
            objective="修订训练计算",
            alias="training",
            history="训练计算",
            conditions=[condition("gradient_dtype", "FP32", message, "replace")],
        ),
    )
    candidate.tasks[0].intent = "answer"
    for task, number in zip(candidate.tasks, [2, 1, 2], strict=True):
        task.history_question_no = number
    candidate.tasks[0].dependencies = candidate.tasks[2].dependencies = ["t2"]
    result = turn(workflow, session, binding, 1, message, candidate)
    assert len(result.items) == 2
    assert len(result.trace["plan"]["goals"]) == 3
    objects = store.list_object_versions(session.session_id, binding)
    assert len(objects) == 4
    assert all(obj.conditions["gradient_dtype"] == "FP32" for obj in objects if obj.version == 2)


def test_H03_ambiguous_model_candidate_does_not_modify_either_object(postgres_dsn):
    store, binding, _, workflow = _workflow(postgres_dsn)
    session = store.create_session("same named objects", binding)
    message = "第一服务A使用AOF，第二服务A使用RDB。"
    turn(
        workflow,
        session,
        binding,
        0,
        message,
        combined(
            plan(
                message,
                objective="第一服务A",
                alias="first",
                conditions=[condition("persistence_mode", "AOF", message)],
            ),
            plan(
                message,
                objective="第二服务A",
                alias="second",
                conditions=[condition("persistence_mode", "RDB", message)],
            ),
        ),
    )
    original = store.list_object_versions(session.session_id, binding)
    message = "请把服务A改成RDB，我还没有指定哪一个，请先澄清。"
    candidate = plan(
        message,
        alias="first",
        history="第一服务A",
        conditions=[condition("persistence_mode", "RDB", message, "replace")],
    )
    candidate.tasks[0].history_need = "ambiguous"
    candidate.tasks[0].history_question_no = 1
    result = turn(workflow, session, binding, 1, message, candidate)
    assert result.items[0].status == "ambiguous"
    assert store.list_object_versions(session.session_id, binding) == original


def test_H03_legacy_history_keeps_original_text_without_inventing_fields(postgres_dsn):
    store, binding, _, workflow = _workflow(postgres_dsn)
    session = store.create_session("legacy conditions", binding)
    message = "服务A采用AOF，每秒fsync。"
    first = turn(workflow, session, binding, 0, message, plan(message, objective=message))
    # Model an existing v3 row; do not manufacture typed fields from its prose.
    with store._connect() as conn:
        conn.execute(
            "UPDATE question_versions SET effective_state='{}'::jsonb WHERE question_id=%s",
            (first.items[0].question_id,),
        )
    message = "解释这个服务的恢复步骤，原有持久化策略保持不变。"
    candidate = plan(
        message,
        history="服务A",
        conditions=[condition("persistence_mode", None, message, "inherit")],
    )
    result = turn(workflow, session, binding, 1, message, candidate, first.items[0].question_id)
    assert result.items[0].status == "answered"
    assert store.list_object_versions(session.session_id, binding) == []


def test_HG04_format_goals_retain_user_source_for_next_summary(postgres_dsn):
    from story_rag.pipelines.task_planning import PlannedGoal

    store, binding, llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("format sources", binding)
    message = "转成表格：提示词不能授予权限。用户确认不等于鉴权。"
    candidate = plan(message, objective="转成表格")
    candidate.tasks[0].intent = "format"
    candidate.goals.extend(
        [
            PlannedGoal(goal_key=f"g{i}", goal_text=text, source_excerpt=text, task_keys=["t1"])
            for i, text in enumerate(["提示词不能授予权限", "用户确认不等于鉴权"], 2)
        ]
    )
    first = turn(workflow, session, binding, 0, message, candidate)
    message = "把刚才两个原则压缩为一句摘要，不增加事实。"
    candidate = plan(message, history="两个原则")
    candidate.tasks[0].intent = "format"
    second = turn(workflow, session, binding, 1, message, candidate, first.items[0].question_id)
    assert first.completion_status == second.completion_status == "complete"
    assert "提示词不能授予权限" in llm.prompts[-1]
    assert "用户确认不等于鉴权" in llm.prompts[-1]
    rows = store.current_questions_at(session.session_id, binding, as_of_revision=2)
    assert rows[0].effective_state["format_source_ref"]["version"] == 1


def test_E05_abstention_coverage_is_conservative_but_invalid_indexes_rejected(monkeypatch):
    import json

    from story_rag.pipelines.conversation_generation import (
        ConversationAnswerPipeline,
        GateItem,
        GeneratedItem,
        ModelCallMetric,
    )
    from tests.test_conversation_generation import BatchLLM, _tasks

    task = _tasks(1)[0]
    candidate = GeneratedItem(
        task_key=task.task_key,
        question_id=task.question_id,
        question_version=1,
        answer="无",
        evidence_ids=[],
        answerability="no_evidence",
    )
    gate = GateItem(
        task_key=task.task_key,
        question_id=task.question_id,
        question_version=1,
        scope_ok=True,
        evidence_supported=True,
        answer_relevant=True,
        answerability="no_evidence",
        covered_requirement_indexes=[0],
        missing_requirement_indexes=[],
        evidence_missing_requirement_indexes=[0],
    )
    pipeline = ConversationAnswerPipeline(BatchLLM())
    metric = ModelCallMetric(
        stage="gate",
        input_upper_bound=1,
        max_output_tokens=4096,
        response_bytes=1,
        elapsed_ms=1,
        usage_known=False,
    )
    monkeypatch.setattr(
        pipeline,
        "_call",
        lambda *a, **kw: (json.dumps({"items": [gate.model_dump(mode="json")]}), metric),
    )
    gates, invalid, _ = pipeline._gate([task], {task.task_key: candidate}, [], "gate")
    assert not invalid and gates[task.task_key].covered_requirement_indexes == []
    assert pipeline._decide(task, candidate, gates[task.task_key]).status == "no_evidence"
    gate.evidence_missing_requirement_indexes = [9]
    _, invalid, _ = pipeline._gate([task], {task.task_key: candidate}, [], "gate")
    assert invalid[task.task_key] == "ANSWERABILITY_COVERAGE_INVALID"


def test_planning_repair_feedback_stays_inside_preflight_allowance():
    from story_rag.pipelines.task_planning import PlanningValidationError, TaskPlanningPipeline

    text = TaskPlanningPipeline._validation_feedback(
        PlanningValidationError("INVALID_SOURCE", "来源" * 5000)
    )
    assert len(text.encode("utf-8")) <= 4096
    assert "INVALID_SOURCE" in text


def test_H03_resolved_inputs_retire_conditional_clarification_requirement(postgres_dsn):
    from story_rag.pipelines.task_planning import PlannedGapResolution, SourceBackedText

    store, binding, _, workflow = _workflow(postgres_dsn)
    session = store.create_session("conditional clarification", binding)
    message = "计算缓存，层数未提供，缺少必要参数时向用户询问。"
    candidate = plan(
        message,
        gaps=[
            {"description": "层数", "state": "explicitly_missing", "basis_excerpt": "层数未提供"}
        ],
    )
    candidate.tasks[0].output_requirements = [
        SourceBackedText(text="缺少必要参数时向用户询问", source_excerpt="缺少必要参数时向用户询问")
    ]
    first = turn(workflow, session, binding, 0, message, candidate)
    message = "层数32，给出结果。"
    candidate = plan(message, history="计算缓存")
    candidate.tasks[0].resolved_inputs = [
        PlannedGapResolution(missing_input="层数", source_excerpt="层数32")
    ]
    turn(workflow, session, binding, 1, message, candidate, first.items[0].question_id)
    current = store.current_questions_at(session.session_id, binding, as_of_revision=2)[0]
    assert not current.missing_inputs
    assert "缺少必要参数时向用户询问" not in current.requirements


def test_HG06_evaluator_preserves_saved_conditions_without_regeneration(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from acceptance_evaluation import evaluation_inputs

    payload = {
        "evidence_pool": {},
        "tasks": [
            {
                "task_key": "t1",
                "question": "计算缓存",
                "allowed_evidence_ids": [],
                "current_conditions": {"layers": 32, "batch": 4},
                "user_provided_facts": [{"source_excerpt": "层数32"}],
                "requirements": ["以GiB报告"],
                "prohibited_actions": [],
            }
        ],
    }
    record = {
        "group": "B03",
        "position": 1,
        "response": {
            "items": [{"task_key": "t1", "status": "answered", "answer": "4GiB"}],
            "trace": {"messages": [[{"content": json.dumps(payload)}]]},
        },
    }
    (tmp_path / "C2").mkdir()
    (tmp_path / "C2/results.json").write_text(json.dumps([record]), encoding="utf-8")
    rows = evaluation_inputs(tmp_path, include_requirements=True)
    assert len(rows) == 1 and rows[0]["answer"] == "4GiB"
    assert '"layers": 32' in rows[0]["reference"]
    assert "以GiB报告" in rows[0]["query"]


def test_planning_repair_reports_bad_values_and_quotes_together():
    from story_rag.pipelines.task_planning import PlanningValidationError, TaskPlanningPipeline

    message = (
        "Use data parallelism only on 8 GPUs, with no offload, "
        "tensor parallelism or pipeline parallelism."
    )
    candidate = plan(
        message,
        conditions=[
            condition("data_parallel_size", "8", "Use data parallelism only"),
            condition("tensor_parallel_size", "no tensor parallelism", "no tensor parallelism"),
        ],
    )
    with pytest.raises(PlanningValidationError) as error:
        TaskPlanningPipeline._validate_candidate(message, candidate)
    feedback = TaskPlanningPipeline._validation_feedback(error.value)
    assert "data_parallel_size" in feedback
    assert "no tensor parallelism" in feedback


@pytest.mark.parametrize(
    "value,quote,valid",
    [
        ("2", "梯度FP32", False),
        ("2", "batch=-2", False),
        ("2", "batch=2.5", False),
        ("2", "batch=2", True),
        ("16", "预算16GiB", True),
        ("7,000,000,000", "7,000,000,000 parameters", True),
    ],
)
def test_M03_numbers_must_be_numeric_literals_not_dtype_substrings(value, quote, valid):
    from story_rag.pipelines.task_planning import TaskPlanningPipeline
    from tests.conftest import ScriptedLLM

    candidate = plan(quote, conditions=[condition("batch_size", value, quote)])
    assert TaskPlanningPipeline(ScriptedLLM()).validate_candidate(quote, candidate).valid == valid


@pytest.mark.parametrize("status,blocked", [(200, False), (402, True), (401, True), (403, True)])
def test_HG06_provider_rejection_stops_remaining_live_cases(tmp_path, monkeypatch, status, blocked):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from system_acceptance import Audit

    audit = Audit(tmp_path)
    audit.calls.append({"id": 9, "status_code": status})
    reason = audit.terminal_provider_rejection()
    assert bool(reason) == blocked
    if blocked:
        assert reason == f"PROVIDER_HTTP_{status}_CALL_9"


def test_C06_explicit_variant_cannot_be_only_a_textual_answer(postgres_dsn):
    from story_rag.pipelines.task_planning import PlannedCondition, TaskPlanningPipeline
    from tests.conftest import ScriptedLLM

    store, binding, llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("compare missing delta", binding)
    message = "服务A梯度FP32。"
    first = turn(
        workflow, session, binding, 0, message,
        plan(message, alias="service_a", conditions=[condition("gradient_dtype", "FP32", message)]),
    )
    message = "另外比较梯度BF16的独立分支，不修改服务A当前FP32梯度方案。"
    candidate = plan(
        message, history="service_a", alias="service_a",
        conditions=[condition("gradient_dtype", None, "不修改服务A当前FP32梯度方案", "inherit")],
    )
    candidate.tasks[0].intent = "compare"
    validator = TaskPlanningPipeline(ScriptedLLM())
    invalid = validator.validate_candidate(message, candidate)
    assert not invalid.valid and invalid.errors == ["MISSING_COMPARISON_BRANCH_CHANGE"]
    calls = len(llm.calls)
    rejected = workflow.answer_turn(
        session.session_id, binding, request_id="bad-branch", expected_revision=1,
        payload=RequestPayload(message=message, reply_to_question_id=first.items[0].question_id),
        plan=candidate, diagnostic=True,
    )
    assert rejected.status == "failed" and rejected.revision == 1
    assert len(llm.calls) == calls
    assert len(store.list_object_versions(session.session_id, binding)) == 1
    candidate.tasks[0].conditions = [PlannedCondition(
        **condition("gradient_dtype", "BF16", "另外比较梯度BF16的独立分支", "replace")
    )]
    assert validator.validate_candidate(message, candidate).valid
    result = turn(workflow, session, binding, 1, message, candidate, first.items[0].question_id)
    objects = store.list_object_versions(session.session_id, binding)
    assert len(objects) == 2
    original = next(obj for obj in objects if obj.derived_from is None)
    branch = next(obj for obj in objects if obj.derived_from is not None)
    assert original.conditions["gradient_dtype"] == "FP32"
    assert branch.conditions["gradient_dtype"] == "BF16"
    assert branch.derived_from.object_id == original.object_id
    current = store.current_questions_at(session.session_id, binding, as_of_revision=2)
    compared = next(q for q in current if q.question_id == result.items[0].question_id)
    assert compared.object_refs[0].object_id == branch.object_id


@pytest.mark.parametrize("message", [
    "继续比较刚才AOF和RDB的优缺点",
    "继续比较刚才AOF和RDB的优缺点，不创建新分支",
])
def test_C06_ordinary_comparison_does_not_require_a_made_up_branch(message):
    from story_rag.pipelines.task_planning import TaskPlanningPipeline
    from tests.conftest import ScriptedLLM

    candidate = plan(message, history="AOF/RDB")
    candidate.tasks[0].intent = "compare"
    assert TaskPlanningPipeline(ScriptedLLM()).validate_candidate(message, candidate).valid


def test_C06_english_independent_variant_requires_a_condition_change():
    from story_rag.pipelines.task_planning import TaskPlanningPipeline
    from tests.conftest import ScriptedLLM

    message = "Compare an independent branch with BF16 gradients, keeping current FP32 unchanged."
    candidate = plan(message, history="training", alias="training")
    candidate.tasks[0].intent = "compare"
    result = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(message, candidate)
    assert not result.valid and result.errors == ["MISSING_COMPARISON_BRANCH_CHANGE"]
