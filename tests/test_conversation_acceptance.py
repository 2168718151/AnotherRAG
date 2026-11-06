"""整体专项验收：禁止动作修复的成功、失败及账本闭环。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse

from story_rag.adapters.model_factory import BudgetedOpenAILike
from story_rag.conversation_domain import RequestPayload
from story_rag.pipelines.conversation_generation import (
    ConversationAnswerPipeline,
    ConversationGenerationError,
    _numeric_expression_error,
    _prohibited_action_error,
)
from story_rag.pipelines.task_planning import TaskPlanningPipeline
from tests.conftest import ScriptedLLM
from tests.test_conversation_generation import BatchLLM, _tasks
from tests.test_conversation_ledger import _case_messages, postgres_dsn
from tests.test_conversation_workflow import _workflow
from tests.test_task_planning import _frozen_plan

__all__ = ["postgres_dsn"]


class NotificationLLM(BatchLLM):
    behavior: str = "persistent"

    def stream_chat(self, messages: list[ChatMessage], **kwargs):
        for frame in super().stream_chat(messages, **kwargs):
            body = json.loads(frame.message.content or "{}")
            if self.calls[-1] != "gate":
                unsafe = self.behavior == "persistent" or (
                    self.behavior == "repaired" and self.calls[-1] == "generation"
                )
                for item in body["items"]:
                    item["answer"] = (
                        "审批完成后通知发起人。"
                        if unsafe
                        else "助手仅生成待审批变更单；人工审批平台执行并提供回执。不得通知任何人。"
                    )
            text = json.dumps(body, ensure_ascii=False)
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=text),
                delta=text,
                raw=frame.raw,
            )


def _record(name: str, value: dict) -> None:
    directory = os.getenv("ACCEPTANCE_REPORT_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )


@pytest.mark.contract
def test_live_negative_clause_must_not_allow_notification() -> None:
    task = _tasks(1)[0].model_copy(update={"prohibited_actions": ["must not notify anyone"]})
    candidate = (
        "On any invalidation, mark the queued job cancelled, do not call the business API, "
        "record the reason, and notify the requesting user that re-confirmation is needed."
    )
    observed = _prohibited_action_error(task, candidate)
    _record(
        "live_notification_false_negative",
        {
            "candidate": candidate,
            "observed": observed,
            "expected": "PROHIBITED_ACTION:notification",
            "source": "B/positions/Z00-03.json 的真实模型候选",
        },
    )
    assert observed == "PROHIBITED_ACTION:notification"


@pytest.mark.contract
def test_real_adapter_reasoning_content_and_usage_only_tail() -> None:
    packets = [
        {
            "id": "stream-case",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "visible-answer",
                        "reasoning_content": "private-reasoning",
                    },
                    "finish_reason": None,
                }
            ],
        },
        {"id": "stream-case", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            "id": "stream-case",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
    ]
    wire = "".join("data: " + json.dumps(p) + "\n\n" for p in packets) + "data: [DONE]\n\n"
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=wire.encode(),
            )
        )
    )
    llm = BudgetedOpenAILike(
        model="mock-model",
        api_base="http://127.0.0.1:9999/v1",
        api_key="test-only",
        is_chat_model=True,
        http_client=client,
        max_retries=0,
    )
    frames = list(llm.stream_chat([ChatMessage(role="user", content="test")]))
    assert frames[-1].message.content == "visible-answer"
    assert all("private-reasoning" not in (f.message.content or "") for f in frames)
    assert frames[-1].additional_kwargs.get("total_tokens") == 15


@pytest.mark.contract
@pytest.mark.parametrize("behavior", ["allowed", "repaired", "persistent"])
def test_prohibited_action_three_paths(behavior: str) -> None:
    task = _tasks(1)[0].model_copy(update={"prohibited_actions": ["不得通知任何人"]})
    llm = NotificationLLM(behavior=behavior)
    result = ConversationAnswerPipeline(llm).answer([task])
    _record(
        f"prohibited_{behavior}",
        {
            "calls": llm.calls,
            "result": result.model_dump(mode="json"),
        },
    )
    if behavior == "persistent":
        assert llm.calls == ["generation", "repair"]
        assert result.items[0].status == "technical_failure"
        assert result.items[0].error_code == "PROHIBITED_ACTION:notification"
        assert "通知发起人" not in result.answer
    else:
        assert result.items[0].status == "answered"
        assert "不得通知" in result.answer
        assert llm.repairs == (1 if behavior == "repaired" else 0)


@pytest.mark.contract
def test_persistent_notification_repair_never_commits_and_session_recovers(
    postgres_dsn: str,
) -> None:
    store, binding, _llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("禁止动作失败闭环", binding)
    planner = TaskPlanningPipeline(ScriptedLLM())
    messages = _case_messages()
    for turn, message in enumerate(messages[:3], 1):
        plan = planner.validate_candidate(message, _frozen_plan(message, turn)).plan
        result = workflow.answer_turn(
            session.session_id,
            binding,
            request_id=f"setup-{turn}",
            expected_revision=turn - 1,
            payload=RequestPayload(message=message),
            plan=plan,
        )
        assert result.status == "succeeded"
    good_generator = workflow._generator
    bad_llm = NotificationLLM(behavior="persistent")
    workflow._generator = ConversationAnswerPipeline(bad_llm)
    message = messages[3]
    plan = planner.validate_candidate(message, _frozen_plan(message, 4)).plan
    before = store.current_questions_at(session.session_id, binding, as_of_revision=3)
    failed = workflow.answer_turn(
        session.session_id,
        binding,
        request_id="unsafe-repair",
        expected_revision=3,
        payload=RequestPayload(message=message),
        plan=plan,
        diagnostic=True,
    )
    after = store.current_questions_at(session.session_id, binding, as_of_revision=3)
    _record(
        "prohibited_commit_failure",
        {
            "result": failed.model_dump(mode="json"),
            "before": [x.model_dump(mode="json") for x in before],
            "after": [x.model_dump(mode="json") for x in after],
            "calls": bad_llm.calls,
        },
    )
    assert failed.status == "failed"
    assert failed.revision == 3
    assert store.get_session(session.session_id, binding).revision == 3
    assert before == after
    assert "通知发起人" not in failed.answer
    workflow._generator = good_generator
    recovered = workflow.answer_turn(
        session.session_id,
        binding,
        request_id="safe-new-request",
        expected_revision=3,
        payload=RequestPayload(message=message),
        plan=plan,
    )
    assert recovered.status == "succeeded"
    assert recovered.revision == 4


@pytest.mark.contract
def test_empty_evidence_and_mixed_status_do_not_generate_missing_answers() -> None:
    tasks = _tasks(3)
    tasks[1] = tasks[1].model_copy(
        update={
            "evidence_decision": "waiting_input",
            "evidence": [],
            "user_facts": [],
        }
    )
    tasks[2] = tasks[2].model_copy(
        update={
            "evidence": [],
            "user_facts": [],
        }
    )
    llm = BatchLLM()
    pipeline = ConversationAnswerPipeline(llm)
    empty = pipeline.answer([tasks[2]])
    assert empty.items[0].status == "no_evidence"
    assert empty.answer == "1. 无"
    assert llm.calls == []
    mixed = pipeline.answer(tasks)
    assert [item.status for item in mixed.items] == [
        "answered",
        "waiting_input",
        "no_evidence",
    ]
    assert len(mixed.items) == 3
    assert "2. 信息不足" in mixed.answer and "3. 无" in mixed.answer
    assert llm.calls == ["generation", "gate"]


@pytest.mark.contract
@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("empty", "EMPTY_RESPONSE"),
        ("disconnect", "MODEL_CALL_UNKNOWN"),
    ],
)
def test_empty_or_interrupted_stream_and_next_attempt_buffers(
    scenario: str,
    expected: str,
) -> None:
    class StreamLLM(ScriptedLLM):
        attempts: int = 0
        closed: int = 0
        scenario: str = "empty"

        def stream_chat(self, _messages, **_kwargs):
            self.attempts += 1
            try:
                if self.attempts == 1 and self.scenario == "disconnect":
                    yield ChatResponse(
                        message=ChatMessage(role="assistant", content='{"items":['),
                        delta='{"items":[',
                        raw={},
                    )
                    raise httpx.ReadError("Injected stream interruption")
                body = "" if self.attempts == 1 else '{"items":[]}'
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content=body),
                    delta=body,
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            finally:
                self.closed += 1

    llm = StreamLLM(scenario=scenario)
    pipeline = ConversationAnswerPipeline(llm)
    args = dict(stage="generation", input_upper=100, max_output=2048)
    messages = [ChatMessage(role="user", content="stream boundary")]
    with pytest.raises(ConversationGenerationError) as failed:
        pipeline._call(messages, **args)
    assert failed.value.code == expected
    assert failed.value.outcome_unknown == (scenario == "disconnect")
    assert llm.closed == 1
    body, metric = pipeline._call(messages, **args)
    assert body == '{"items":[]}'
    assert metric.finish_reason == "stop"
    assert llm.closed == 2


@pytest.mark.contract
@pytest.mark.parametrize(
    "answer",
    [
        "1.5 * 2 = 3.0",
        "2.5 + 2.5 = 5.0",
    ],
)
def test_valid_decimal_arithmetic_is_not_false_rejected(answer: str) -> None:
    observed = _numeric_expression_error(answer)
    _record("valid_decimal_" + answer[:3], {"answer": answer, "observed": observed})
    assert observed is None


@pytest.mark.contract
def test_wrong_decimal_arithmetic_must_not_pass() -> None:
    answer = "2.5 + 2.5 = 6.0"
    observed = _numeric_expression_error(answer)
    _record("wrong_decimal", {"answer": answer, "observed": observed})
    assert observed == "NUMERIC_EXPRESSION_MISMATCH"


@pytest.mark.contract
def test_committed_request_replay_after_workflow_reconstruction(postgres_dsn: str) -> None:
    store, binding, llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("重建工作流后重放", binding)
    message = _case_messages()[0]
    plan = (
        TaskPlanningPipeline(ScriptedLLM())
        .validate_candidate(
            message,
            _frozen_plan(message, 1),
        )
        .plan
    )
    payload = RequestPayload(message=message)
    first = workflow.answer_turn(
        session.session_id,
        binding,
        request_id="committed",
        expected_revision=0,
        payload=payload,
        plan=plan,
    )
    assert first.status == "succeeded"
    llm.calls.clear()
    restored = type(workflow)(
        store=store,
        planner=workflow._planner,
        history=workflow._history,
        evidence=workflow._evidence,
        generator=ConversationAnswerPipeline(llm),
    )
    replay = restored.answer_turn(
        session.session_id,
        binding,
        request_id="committed",
        expected_revision=0,
        payload=payload,
        plan=plan,
    )
    assert replay.status == "succeeded" and replay.answer_not_retained
    assert replay.answer == "" and replay.revision == 1
    assert llm.calls == []
