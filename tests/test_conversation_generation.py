from __future__ import annotations

import json
from uuid import NAMESPACE_URL, uuid5

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from pydantic import Field

from story_rag.conversation_domain import EvidenceRef
from story_rag.pipelines.conversation_generation import (
    ConversationAnswerPipeline,
    ConversationGenerationError,
    ConversationTaskInput,
    GeneratedItem,
    _missing_required_concepts,
    _numeric_expression_error,
    _prohibited_action_error,
)
from story_rag.pipelines.evidence_mapping import EvidencePoolEntry
from story_rag.pipelines.token_budget import BudgetExceededError
from tests.conftest import ScriptedLLM
from tests.test_task_planning import _case_messages, _frozen_plan


class BatchLLM(ScriptedLLM):
    mode: str = "normal"
    calls: list[str] = Field(default_factory=list)
    repairs: int = 0

    def stream_chat(self, messages: list[ChatMessage], **kwargs):
        content = "\n".join(message.content or "" for message in messages)
        self.prompts.append(content)
        payload, _ = json.JSONDecoder().raw_decode(content[content.index('{"evidence_pool"'):])
        rows = payload["tasks"]
        gate = "candidate" in rows[0]
        repair = "correction" in rows[0]
        self.calls.append("gate" if gate else "repair" if repair else "generation")
        if repair:
            self.repairs += 1
        if gate:
            items = [
                {
                    "task_key": row["task_key"],
                    "question_id": row["question_id"],
                    "question_version": row["question_version"],
                    "scope_ok": True,
                    "covered_requirement_indexes": list(range(len(row["requirements"]))),
                    "missing_requirement_indexes": [],
                    "evidence_supported": self.mode != "oversize_then_gate_reject",
                    "answer_relevant": True,
                    "unsupported_claims": (
                        ["审批平台回执格式未提供，需确认"]
                        if self.mode == "caveat_gate" and not row.get("correction")
                        else []
                    ),
                    "prompt_injection_detected": self.mode == "injection_notice",
                }
                for row in rows
            ]
            if self.mode == "missing_gate_item" and len(items) > 1:
                items.pop()
            if self.mode == "off_by_one_gate" and not rows[0].get("correction"):
                items[0]["covered_requirement_indexes"].append(len(rows[0]["requirements"]))
        else:
            items = [
                {
                    "task_key": row["task_key"],
                    "question_id": row["question_id"],
                    "question_version": row["question_version"],
                    "answer": (
                        "失效后应通知发起人重新提交。"
                        if self.mode == "forbidden_notification" and not repair
                        else (
                            "The request enters a background queue; bind the execution identity."
                            if not repair else
                            "In the multi-tenant background queue, the shared service account "
                            "executes only after checking the initiating user's authorization."
                        ) if self.mode == "history_omission"
                        else "长" * 1500
                        if self.mode == "oversize_then_gate_reject" and not repair
                        else f"本轮任务 {row['task_key']} 已回答。审批平台回执格式未提供，需确认。"
                        if self.mode == "caveat_gate"
                        else (
                            f"本轮任务 {row['task_key']} 已回答。 "
                            + "、".join(row["explicit_required_concepts"])
                        )
                    ),
                    "evidence_ids": row["allowed_evidence_ids"],
                }
                for row in rows
            ]
            if self.mode == "numeric_retry":
                value = (
                    "2 × 32 × 11 × 8192 × 8 × 128 × 2 = 11,811,160,064 bytes; "
                    "GiB = 11,811,160,064 / 2^30 = 11 GiB"
                    if repair and self.repairs >= 2
                    else "2 × 32 × 11 × 8192 × 8 × 128 × 2 = "
                    "1,181,116,416 bytes; GiB = 1.1 GiB"
                )
                for item in items:
                    item["answer"] = value
            if self.mode == "cross_task_citation" and len(items) > 1 and not repair:
                items[-1]["evidence_ids"] = rows[0]["allowed_evidence_ids"]
            if self.mode == "missing_item" and len(items) > 1 and not repair:
                items.pop()
            if self.mode == "duplicate_item" and not repair:
                items.append(items[0].copy())
        body = json.dumps({"items": items}, ensure_ascii=False)
        raw = {
            "choices": [{"finish_reason": (
                "length" if self.mode == "always_truncated" else "stop"
            )}],
        }
        if self.mode != "unknown_usage":
            raw["usage"] = {"prompt_tokens": 100, "completion_tokens": 50}
        yield ChatResponse(
            message=ChatMessage(role="assistant", content=body),
            delta=body,
            raw=raw,
        )


def _tasks(count: int) -> list[ConversationTaskInput]:
    message = _case_messages()[0]
    plan = _frozen_plan(message, 1)
    first = plan["tasks"][0]
    evidence = EvidencePoolEntry(
        evidence_id="phase0-mock-policy",
        ref=EvidenceRef(document_id="phase0-mock-policy", node_id="policy-node"),
        text="服务端鉴权和用户确认是不同控制，执行前需校验动作、资源与权限。",
        source="方案零结构测试固定证据",
    )
    return [
        ConversationTaskInput(
            task_key=f"t{index}",
            question_id=uuid5(NAMESPACE_URL, f"phase0-budget-{index}"),
            question_version=1,
            question_text=first["objective"]["text"],
            requirements=[first["output_requirements"][0]["text"]],
            evidence=[evidence],
            user_facts=[{"field": "current_message", "source_excerpt": message[:60]}],
        )
        for index in range(1, count + 1)
    ]


@pytest.mark.unit
@pytest.mark.parametrize("count", [1, 4, 8, 12])
def test_phase0_budget_and_answer_count_preserve_headroom(count: int) -> None:
    llm = BatchLLM()
    pipeline = ConversationAnswerPipeline(llm)
    tasks = _tasks(count)
    preview = pipeline.preview(tasks)

    assert preview["task_count"] == preview["answerable_task_count"] == count
    assert len(preview["batches"]) == (count + 1) // 2
    assert sum(len(batch["task_keys"]) for batch in preview["batches"]) == count
    assert all(batch["generation_output_headroom"] >= 1536 for batch in preview["batches"])
    assert all(batch["gate_output_headroom"] >= 2048 for batch in preview["batches"])
    assert all(batch["generation_context_headroom"] >= 4096 for batch in preview["batches"])
    assert all(batch["gate_context_headroom"] >= 4096 for batch in preview["batches"])

    result = pipeline.answer(tasks)

    assert len(result.items) == count
    assert result.budget["expected_item_count"] == count
    assert result.budget["produced_item_count"] == count
    assert result.budget["status_counts"] == {"answered": count}
    assert result.budget["unknown_call_reservation"] == 0
    assert result.budget["settled_tokens"] == 150 * len(result.metrics)
    assert len({item.question_id for item in result.items}) == count
    assert [item.task_key for item in result.items] == [task.task_key for task in tasks]
    assert all(item.status == "answered" for item in result.items)
    assert len(llm.calls) == 2 * len(preview["batches"])
    assert all(metric.usage_known for metric in result.metrics)
    assert all(
        metric.reported_completion_tokens < metric.max_output_tokens
        for metric in result.metrics
    )
    if count >= 4:
        assert llm.prompts[0].count("服务端鉴权和用户确认是不同控制") == 1


@pytest.mark.unit
def test_numeric_answer_gets_one_extra_bounded_recalculation_then_gate() -> None:
    llm = BatchLLM(mode="numeric_retry")
    result = ConversationAnswerPipeline(llm).answer(_tasks(1))

    assert result.items[0].status == "answered"
    assert "11,811,160,064 bytes" in result.items[0].answer
    assert result.budget["produced_item_count"] == 1
    assert llm.calls == ["generation", "repair", "repair", "gate"]
    assert llm.repairs == 2


@pytest.mark.unit
def test_phase0_json_escaping_is_reserved_for_generation_and_gate() -> None:
    tasks = _tasks(2)
    pipeline = ConversationAnswerPipeline(BatchLLM())
    preview = pipeline.preview(tasks)
    assert sum(len(batch["task_keys"]) for batch in preview["batches"]) == 2
    for batch in preview["batches"]:
        selected = [task for task in tasks if task.task_key in batch["task_keys"]]
        worst_answers = {
            task.task_key: GeneratedItem(
                task_key=task.task_key,
                question_id=task.question_id,
                question_version=task.question_version,
                answer="\\" * pipeline._budget.per_task_generation_reserve,
                evidence_ids=[entry.evidence_id for entry in task.evidence],
            ) for task in selected
        }
        serialized = json.dumps(
            {"items": [item.model_dump(mode="json") for item in worst_answers.values()]},
            ensure_ascii=False, separators=(",", ":"),
        )
        assert len(serialized.encode("utf-8")) < pipeline._budget.generation_output
        assert (
            batch["gate_worst_case_input_upper_bound"]
            >= pipeline._estimate(
                "gate", selected, output=pipeline._budget.gate_output,
                answers=worst_answers,
            )
        )
        assert (
            batch["gate_repair_worst_case_input_upper_bound"]
            > batch["gate_worst_case_input_upper_bound"]
        )
        assert batch["generation_output_headroom"] >= 1536
        assert batch["gate_output_headroom"] >= 2048


@pytest.mark.unit
def test_phase0_thirteenth_task_fails_explicitly_without_model_call() -> None:
    llm = BatchLLM()
    with pytest.raises(ConversationGenerationError, match="13 tasks") as raised:
        ConversationAnswerPipeline(llm).preview(_tasks(13))
    assert raised.value.code == "TOO_MANY_TASKS"
    assert not llm.calls


@pytest.mark.unit
def test_phase0_one_repair_only_and_gate_rejection_is_not_no_evidence() -> None:
    llm = BatchLLM(mode="oversize_then_gate_reject")
    result = ConversationAnswerPipeline(llm).answer(_tasks(1))
    assert llm.repairs == 1
    assert llm.calls == ["generation", "repair", "gate"]
    assert result.items[0].status == "technical_failure"
    assert result.items[0].error_code == "REPAIR_EXHAUSTED"
    assert result.items[0].answer != "无"


@pytest.mark.unit
def test_phase0_cross_task_citation_repairs_only_offending_item() -> None:
    tasks = _tasks(2)
    distinct = tasks[1].evidence[0].model_copy(update={
        "evidence_id": "phase0-mock-draft",
        "ref": EvidenceRef(document_id="phase0-mock-draft", node_id="draft-node"),
    })
    tasks[1] = tasks[1].model_copy(update={"evidence": [distinct]})
    llm = BatchLLM(mode="cross_task_citation")
    result = ConversationAnswerPipeline(llm).answer(tasks)
    assert [item.status for item in result.items] == ["answered", "answered"]
    assert llm.calls == ["generation", "repair", "gate"]
    assert llm.repairs == 1
    assert [ref.document_id for ref in result.items[1].evidence_refs] == [
        "phase0-mock-draft"
    ]


@pytest.mark.unit
def test_phase0_unknown_usage_stays_reserved_and_injection_flag_does_not_blank_answer() -> None:
    unknown = ConversationAnswerPipeline(BatchLLM(mode="unknown_usage")).answer(_tasks(1))
    assert unknown.budget["settled_tokens"] == 0
    assert unknown.budget["unknown_call_reservation"] > 0
    assert all(not metric.usage_known for metric in unknown.metrics)

    injection = ConversationAnswerPipeline(BatchLLM(mode="injection_notice")).answer(_tasks(1))
    assert injection.items[0].status == "answered"
    assert injection.items[0].prompt_injection_detected


@pytest.mark.unit
def test_phase0_oversized_evidence_fails_before_any_generation() -> None:
    tasks = _tasks(1)
    huge = tasks[0].evidence[0].model_copy(update={"text": "授权检查" * 8000})
    tasks[0] = tasks[0].model_copy(update={"evidence": [huge]})
    llm = BatchLLM()
    with pytest.raises(BudgetExceededError):
        ConversationAnswerPipeline(llm).answer(tasks)
    assert not llm.calls


@pytest.mark.unit
def test_bug_ledger_numeric_rounding_guard_is_deterministic() -> None:
    assert _numeric_expression_error("15750000000 / 2^30 = 14.68 GiB") == (
        "NUMERIC_EXPRESSION_MISMATCH"
    )
    assert _numeric_expression_error("15750000000 / 2^30 = 14.67 GiB") is None


@pytest.mark.unit
def test_phase0_prohibited_notification_is_repaired_without_false_rejecting_ban() -> None:
    task = _tasks(1)[0].model_copy(update={
        "prohibited_actions": ["Publishing, deletion, or notification."],
    })
    assert _prohibited_action_error(task, "不得发布、删除或通知任何人。") is None
    assert _prohibited_action_error(
        task, "Publishing, deletion, and notification remain prohibited."
    ) is None
    assert _prohibited_action_error(
        task,
        "If the request involves publishing, deleting, or notifying anyone, it is refused. "
        "The server enforces the prohibition on notifying anyone; no notifications are sent.",
    ) is None
    assert _prohibited_action_error(
        task,
        "The server rejects any publish, delete, or notify action; it enforces the "
        "prohibitions on publishing, deleting, and notifying anyone."
    ) is None
    assert _prohibited_action_error(task, "失效后应通知发起人重新提交。") == (
        "PROHIBITED_ACTION:notification"
    )
    llm = BatchLLM(mode="forbidden_notification")
    result = ConversationAnswerPipeline(llm).answer([task])
    assert llm.calls == ["generation", "repair", "gate"]
    assert result.items[0].status == "answered"
    assert "通知" not in result.items[0].answer


@pytest.mark.unit
def test_phase0_retained_history_concepts_cannot_be_approved_by_a_generic_label() -> None:
    task = _tasks(1)[0].model_copy(update={
        "requirements": [
            "Revise the draft workflow for a multi-tenant, queued, "
            "shared-service-account permission model."
        ],
    })
    assert _missing_required_concepts(
        task, "The request enters a background queue; bind the execution identity."
    ) == ["multi-tenant", "shared service account"]

    llm = BatchLLM(mode="history_omission")
    result = ConversationAnswerPipeline(llm).answer([task])
    assert llm.calls == ["generation", "repair", "gate"]
    assert result.items[0].status == "answered"
    assert _missing_required_concepts(task, result.items[0].answer) == []
    assert result.budget["expected_item_count"] == result.budget["produced_item_count"] == 1


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["missing_item", "duplicate_item"])
def test_phase0_item_count_repairs_once_without_losing_other_task(mode: str) -> None:
    llm = BatchLLM(mode=mode)
    result = ConversationAnswerPipeline(llm).answer(_tasks(2))
    assert [item.status for item in result.items] == ["answered", "answered"]
    assert result.budget["expected_item_count"] == result.budget["produced_item_count"] == 2
    assert llm.repairs == 1


@pytest.mark.unit
def test_phase0_gate_missing_item_keeps_other_answer() -> None:
    llm = BatchLLM(mode="missing_gate_item")
    result = ConversationAnswerPipeline(llm).answer(_tasks(2))
    assert [item.status for item in result.items] == ["answered", "technical_failure"]
    assert result.items[1].error_code == "MISSING_GATE_ITEM"
    assert result.items[1].answer != "无"
    assert llm.repairs == 0


@pytest.mark.unit
def test_phase0_gate_off_by_one_index_gets_one_bounded_gate_retry() -> None:
    llm = BatchLLM(mode="off_by_one_gate")
    result = ConversationAnswerPipeline(llm).answer(_tasks(1))
    assert llm.calls == ["generation", "gate", "gate"]
    assert result.items[0].status == "answered"
    assert result.budget["expected_item_count"] == result.budget["produced_item_count"] == 1


@pytest.mark.unit
def test_phase0_gate_rechecks_unknown_format_caveat_without_answer_regeneration() -> None:
    llm = BatchLLM(mode="caveat_gate")
    result = ConversationAnswerPipeline(llm).answer(_tasks(1))
    assert llm.calls == ["generation", "gate", "gate"]
    assert result.items[0].status == "answered"
    assert [metric.stage for metric in result.metrics] == [
        "generation", "gate", "gate_after_repair"
    ]


@pytest.mark.unit
def test_phase0_truncated_response_never_becomes_answered() -> None:
    llm = BatchLLM(mode="always_truncated")
    result = ConversationAnswerPipeline(llm).answer(_tasks(1))
    assert result.items[0].status == "technical_failure"
    assert result.items[0].error_code == "INCOMPLETE_RESPONSE"
    assert llm.calls == ["generation", "repair"]
    assert all(metric.finish_reason == "length" for metric in result.metrics)


@pytest.mark.unit
def test_phase0_cumulative_stream_snapshots_count_only_new_bytes() -> None:
    class SnapshotLLM(ScriptedLLM):
        def stream_chat(self, _messages, **_kwargs):
            body = "x" * 70000
            for content in (body[:35000], body, body):
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content=content),
                    delta=None,
                    raw={"choices": [{"finish_reason": "stop"}]},
                )

    pipeline = ConversationAnswerPipeline(SnapshotLLM())
    body, metric = pipeline._call(
        [ChatMessage(role="user", content=_case_messages()[0][:60])],
        stage="generation", input_upper=100, max_output=8192,
    )
    assert len(body) == 70000
    assert metric.response_bytes == 70000


@pytest.mark.unit
def test_phase0_gate_budget_preflight_precedes_generation_for_each_batch() -> None:
    tasks = _tasks(2)
    tasks[1] = tasks[1].model_copy(update={
        "evidence": [tasks[1].evidence[0].model_copy(update={
            "text": "审批平台" * 3500,
        })],
    })
    llm = BatchLLM()
    pipeline = ConversationAnswerPipeline(llm)
    preview = pipeline.preview(tasks)
    assert sum(len(batch["task_keys"]) for batch in preview["batches"]) == 2
    assert all(batch["gate_context_headroom"] >= 0 for batch in preview["batches"])
    result = pipeline.answer(tasks)
    assert len(result.items) == 2
    assert result.budget["expected_item_count"] == result.budget["produced_item_count"]


@pytest.mark.unit
def test_phase0_gate_preflight_reserves_all_possible_citations() -> None:
    task = _tasks(1)[0]
    evidence = [
        task.evidence[0].model_copy(update={"evidence_id": f"node-{index}-" + "x" * 180})
        for index in range(12)
    ]
    task = task.model_copy(update={"evidence": evidence})
    pipeline = ConversationAnswerPipeline(BatchLLM())
    preview = pipeline.preview([task])
    candidate = GeneratedItem(
        task_key=task.task_key,
        question_id=task.question_id,
        question_version=task.question_version,
        answer="字" * 1024,
        evidence_ids=[item.evidence_id for item in evidence],
    )
    actual_gate_input = pipeline._estimate(
        "gate", [task], output=pipeline._budget.gate_output,
        answers={task.task_key: candidate},
    )
    assert preview["batches"][0]["gate_worst_case_input_upper_bound"] >= actual_gate_input
    assert preview["batches"][0]["gate_context_headroom"] >= 2048

    second = _tasks(2)[1].model_copy(update={"evidence": evidence})
    split = pipeline.preview([task, second])
    assert len(split["batches"]) == 2
    assert all(batch["generation_output_headroom"] >= 1024
               for batch in split["batches"])
