from __future__ import annotations

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import pytest

from story_rag.conversation_domain import RequestPayload
from story_rag.pipelines.conversation_generation import (
    ConversationAnswerPipeline,
    ConversationGenerationError,
    GateItem,
    GeneratedItem,
    _numeric_expression_error,
    _prohibited_action_error,
)
from story_rag.pipelines.task_planning import TaskPlan, TaskPlanningPipeline
from tests.conftest import ScriptedLLM
from tests.test_conversation_generation import BatchLLM, _tasks
from tests.test_conversation_ledger import postgres_dsn
from tests.test_conversation_workflow import _workflow

__all__ = ["postgres_dsn"]


def plan(message, *, objective=None, history=None, conditions=(), alias=None, gaps=()):
    source = message[:120]
    task = {
        "task_key": "t1",
        "intent": "revise" if history else "answer",
        "objective": {"text": objective or message, "source_excerpt": source},
        "object_scope": {"text": alias or "方案A", "source_excerpt": source},
        "object_alias": alias,
        "conditions": list(conditions),
        "information_gaps": list(gaps),
        "history_need": "required" if history else "none",
        "history_reference": ({"text": history, "source_excerpt": source} if history else None),
    }
    return TaskPlan.model_validate(
        {
            "schema_version": 2,
            "goals": [
                {
                    "goal_key": "g1",
                    "goal_text": objective or message,
                    "source_excerpt": source,
                    "task_keys": ["t1"],
                }
            ],
            "tasks": [task],
        }
    )


@pytest.mark.parametrize(
    "answer,valid",
    [
        ("Do not call the business API and notify the requesting user.", False),
        ("Do not notify anyone, but notify the owner.", False),
        ("Do not publish, delete, or notify anyone.", True),
        ("The server rejects any publish, delete, or notify action.", True),
        ("不得调用业务接口，然后通知发起人。", False),
        ("不得发布、删除或通知任何人。", True),
    ],
)
def test_HG01_negation_is_scoped_to_the_action(answer, valid):
    task = _tasks(1)[0].model_copy(update={"prohibited_actions": ["must not notify anyone"]})
    assert (_prohibited_action_error(task, answer) is None) == valid


@pytest.mark.parametrize(
    "answer,valid",
    [
        ("1.5 * 2 = 3.0", True),
        ("2.5 + 2.5 = 6.0", False),
        ("15750000000 / 2^30 = 14.67 GiB", True),
        ("15750000000 / 2^30 = 14.68 GiB", False),
        ("(1.25 + 0.75) * 2 = 4.00", True),
        ("1e3 * 2 = 2001", False),
    ],
)
def test_HG02_decimal_and_exponent_arithmetic(answer, valid):
    assert (_numeric_expression_error(answer) is None) == valid


def test_HG02_long_numeric_chains_finish_without_backtracking():
    cases = [
        (
            "14,000,000,000 + (14,000,000,000+84,000,000,000)/8 = "
            "14,000,000,000 + 12,250,000,000 = 26,250,000,000 B",
            None,
        ),
        (
            "2 × 32 × 30 × 8192 × 8 × 128 × 2 = 32,212,254,720 bytes; "
            "GiB = 32,212,254,720 / 2^30 = 30 GiB",
            None,
        ),
        (
            "2 × 32 × 30 × 8192 × 8 × 128 × 2 = 3,221,225,472 bytes; "
            "GiB = 3,221,225,472 / 2^30 = 3 GiB",
            "NUMERIC_EXPRESSION_MISMATCH",
        ),
        (
            "2 × 32 × 11 × 8192 × 8 × 128 × 2 = 11,811,160,064 bytes; "
            "GiB = 11,811,160,064 / 2^30 = 11 GiB",
            None,
        ),
        (
            "2 × 32 × 11 × 8192 × 8 × 128 × 2 = 1,181,116,416 bytes; "
            "GiB = 1.1 GiB",
            "NUMERIC_EXPRESSION_MISMATCH",
        ),
        (
            "(14,000,000,000+14,000,000,000+84,000,000,000)/8 = "
            "112,000,000,000/8 = 14,000,000,000 B",
            None,
        ),
        ("14,000,000,000 + 12,250,000,000 = 26,250,000,001", "NUMERIC_EXPRESSION_MISMATCH"),
        ("7,000,000,000 + " * 128 + "bytes", None),
        ("1e3 * 2 = 2001", "NUMERIC_EXPRESSION_MISMATCH"),
    ]
    # 隔离进程限制回归耗时；只载入实际纯函数，避免模型/框架导入影响时限。
    source_file = sys.modules[_numeric_expression_error.__module__].__file__
    probe = """
import ast, json, re, sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from pathlib import Path
source = Path(sys.argv[1]).read_text(encoding="utf-8")
for node in ast.parse(source).body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "_numeric_expression_error", "_safe_arithmetic"
    }:
        exec(compile(ast.Module(body=[node], type_ignores=[]), sys.argv[1], "exec"))
for answer, expected in json.load(sys.stdin):
    observed = _numeric_expression_error(answer)
    assert observed == expected, (answer, observed, expected)
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, source_file],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr


def test_HG02_numeric_repair_feedback_requires_recalculation():
    source = Path(
        __import__("story_rag.pipelines.conversation_generation", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    assert "Recalculate every arithmetic equality from the supplied factors" in source
    assert "do not copy any byte or GiB value from it" in source


def test_HG03_unfulfillable_minimum_rejected_before_generation():
    llm = BatchLLM()
    tasks = _tasks(4)
    tasks = [task.model_copy(update={"requirements": ["每题至少1500汉字"]}) for task in tasks]
    with pytest.raises(ConversationGenerationError, match="cannot satisfy") as exc:
        ConversationAnswerPipeline(llm).answer(tasks)
    assert exc.value.code == "OUTPUT_REQUIREMENT_EXCEEDS_CAPACITY"
    assert llm.calls == []


def test_P02_E06_single_character_source_and_missing_object_are_valid_plans():
    planner = TaskPlanningPipeline(ScriptedLLM())
    assert planner.validate_candidate("2", plan("2")).valid
    candidate = plan(
        "它有什么缺点？",
        gaps=[
            {
                "description": "请说明‘它’指的对象",
                "state": "not_provided",
                "basis_excerpt": "它",
            }
        ],
    )
    assert planner.validate_candidate("它有什么缺点？", candidate).valid
    bad = candidate.model_dump()
    bad["tasks"][0]["objective"]["source_start"] = 1
    bad["tasks"][0]["objective"]["source_end"] = 3
    assert not planner.validate_candidate("它有什么缺点？", bad).valid


def test_M03_inherited_number_cannot_claim_a_new_source():
    message = "参数量保持不变"
    candidate = plan(
        message,
        conditions=[
            {
                "field": "parameter_count",
                "op": "replace",
                "value": 7000000000,
                "source_excerpt": message,
            }
        ],
    )
    result = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(message, candidate)
    assert not result.valid and "CONDITION_VALUE_NOT_IN_CURRENT_SOURCE" in result.errors


def test_P05_interaction_commits_no_knowledge_questions(postgres_dsn):
    store, binding, llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("interaction", binding)
    candidate = TaskPlan(
        schema_version=2,
        plan_kind="interaction",
        interaction_message="你好，请告诉我想了解的问题。",
    )
    result = workflow.answer_turn(
        session.session_id,
        binding,
        request_id="hello",
        payload=RequestPayload(message="你好"),
        expected_revision=0,
        plan=candidate,
    )
    assert result.status == "succeeded" and result.revision == 1 and not result.items
    assert store.list_question_versions(session.session_id, binding) == []
    assert llm.calls == []


def test_H06_explicit_foreign_reference_fails_before_planning(postgres_dsn):
    store, binding, llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("foreign", binding)
    result = workflow.answer_turn(
        session.session_id,
        binding,
        request_id="foreign",
        payload=RequestPayload(message="继续", reply_to_question_id=uuid4()),
        expected_revision=0,
    )
    assert result.status == "failed" and result.error_code == "INVALID_HISTORY_REFERENCE"
    assert result.revision == 0 and llm.calls == []
    assert result.request.usage.llm_calls == 0


def test_B03_thirty_mock_revisions_keep_effective_query_bounded(postgres_dsn):
    store, binding, llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("bounded", binding)
    question_id = None
    lengths = []
    for index in range(30):
        message = f"方案A的batch_size改成{index + 1}"
        candidate = plan(
            message,
            objective="解释当前方案A的批量设置",
            alias="training",
            history="training" if index else None,
            conditions=[
                {
                    "field": "batch_size",
                    "op": "replace" if index else "supplement",
                    "value": str(index + 1),
                    "source_excerpt": message,
                }
            ],
        )
        result = workflow.answer_turn(
            session.session_id,
            binding,
            request_id=f"r{index}",
            payload=RequestPayload(message=message, reply_to_question_id=question_id),
            expected_revision=index,
            plan=candidate,
            diagnostic=True,
        )
        assert result.status == "succeeded", result.model_dump()
        question_id = result.items[0].question_id
        questions = store.list_question_versions(session.session_id, binding)
        lengths.append(len(questions[-1].question_text))
    objects = store.list_object_versions(session.session_id, binding)
    assert objects[-1].conditions["batch_size"] == "30"
    assert len(objects) == 30 and len({obj.object_id for obj in objects}) == 1
    assert max(lengths) < 100 and max(lengths) - min(lengths) < 20
    assert all(len(obj.changes) == 1 for obj in objects)


def test_C03_remove_task_budget_keeps_question_and_records_change(postgres_dsn):
    store, binding, _, workflow = _workflow(postgres_dsn)
    session = store.create_session("remove", binding)
    message = "方案A仅数值题增加16GiB限制"
    first = workflow.answer_turn(
        session.session_id,
        binding,
        request_id="one",
        expected_revision=0,
        payload=RequestPayload(message=message),
        plan=plan(
            message,
            alias="training",
            conditions=[
                {
                    "field": "budget_gib",
                    "op": "supplement",
                    "value": "16",
                    "scope": "task",
                    "source_excerpt": message,
                }
            ],
        ),
    )
    message = "取消刚才的16GiB预算限制"
    second = workflow.answer_turn(
        session.session_id,
        binding,
        request_id="two",
        expected_revision=1,
        payload=RequestPayload(message=message, reply_to_question_id=first.items[0].question_id),
        plan=plan(
            message,
            history="training",
            alias="training",
            conditions=[
                {
                    "field": "budget_gib",
                    "op": "remove",
                    "value": None,
                    "scope": "task",
                    "source_excerpt": message,
                }
            ],
        ),
    )
    assert second.status == "succeeded" and second.items[0].status != "cancelled"
    versions = store.list_question_versions(session.session_id, binding)
    assert versions[0].effective_state["task_conditions"]["budget_gib"] == "16"
    assert versions[-1].effective_state["task_conditions"] == {}
    assert versions[-1].effective_state["task_changes"][0]["before"] == "16"


def test_E05_O05_no_evidence_status_requires_structured_agreement():
    task = _tasks(1)[0]
    answer = GeneratedItem(
        task_key=task.task_key,
        question_id=task.question_id,
        question_version=1,
        answer="无",
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
        missing_requirement_indexes=[0],
        evidence_missing_requirement_indexes=[0],
    )
    result = ConversationAnswerPipeline._decide(task, answer, gate)
    assert result.status == "no_evidence" and result.answer == "无"
    assert not result.evidence_refs
    gate.answerability = "full"
    assert ConversationAnswerPipeline._decide(task, answer, gate).status == "technical_failure"


def test_E06_waiting_input_is_readable_not_an_error_code():
    task = _tasks(1)[0].model_copy(
        update={
            "evidence_decision": "waiting_input",
            "missing_inputs": ["批量大小", "KV头数量"],
        }
    )
    result = ConversationAnswerPipeline(BatchLLM()).answer([task])
    assert "批量大小" in result.answer and "KV头数量" in result.answer
    assert "MISSING_INPUT" not in result.answer
    assert json.loads(result.model_dump_json())["items"][0]["status"] == "waiting_input"


def test_S06_concurrent_diagnostic_files_never_mix_and_conflicts_fail(tmp_path):
    from story_rag.adapters.conversation_diagnostics import DiagnosticArchive

    barrier = threading.Barrier(2)

    def emit(session):
        sink = DiagnosticArchive(tmp_path, session, "request", "attempt-1")
        barrier.wait(timeout=5)
        return sink.write("generation", {"unique": session}, snapshot_id="same-snapshot")

    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(emit, ["a/b", "ab"]))
    assert outputs[0] != outputs[1]
    for path, session in zip(outputs, ["a/b", "ab"], strict=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["session_id"] == record["payload"]["unique"] == session
    with pytest.raises(FileExistsError):
        DiagnosticArchive(tmp_path, "a/b", "request", "attempt-1").write(
            "generation", {"unique": "should-not-overwrite"}, snapshot_id="same-snapshot"
        )
    second = DiagnosticArchive(tmp_path, "a/b", "request", "attempt-2").write(
        "generation", {}, snapshot_id="same-snapshot"
    )
    assert second != outputs[0]
    assert not list(tmp_path.rglob(".pending-*"))


def test_B04_provider_usage_settles_even_when_above_estimate(postgres_dsn):
    import time

    from llama_index.core.base.llms.types import ChatMessage, ChatResponse
    from llama_index.core.callbacks.schema import CBEventType, EventPayload

    from story_rag.adapters.planning_metrics import PlanningMetricsHandler, planning_responses
    from story_rag.pipelines.conversation_generation import ModelCallMetric
    from story_rag.pipelines.task_planning import TaskPlanningResult

    observed = []
    token = planning_responses.set(observed)
    try:
        PlanningMetricsHandler().on_event_end(
            CBEventType.LLM,
            {
                EventPayload.RESPONSE: ChatResponse(
                    message=ChatMessage(role="assistant", content="{}"),
                    raw={
                        "usage": {"prompt_tokens": 3000, "completion_tokens": 500},
                        "choices": [{"finish_reason": "stop"}],
                    },
                )
            },
        )
    finally:
        planning_responses.reset(token)
    assert observed == [{"prompt_tokens": 3000, "completion_tokens": 500, "finish_reason": "stop"}]
    _, _, _, workflow = _workflow(postgres_dsn)
    metric = ModelCallMetric(
        stage="planning",
        input_upper_bound=100,
        max_output_tokens=1000,
        reported_prompt_tokens=3000,
        reported_completion_tokens=500,
        elapsed_ms=1.0,
        response_bytes=2,
        usage_known=True,
    )
    planned = TaskPlanningResult(
        plan=plan("测试"), llm_calls=1, attempts=1, elapsed_ms=1.0, metrics=[metric]
    )
    usage = workflow._usage(planned, [], time.perf_counter(), planning_input_upper=100)
    assert usage.llm_calls == 1 and usage.unknown_usage_calls == 0
    assert usage.settled_tokens == usage.accounted_tokens == 3500
    assert usage.prompt_tokens == 3000 and usage.completion_tokens == 500
    unknown = metric.model_copy(
        update={
            "usage_known": False,
            "reported_prompt_tokens": None,
            "reported_completion_tokens": None,
        }
    )
    planned.metrics = [unknown]
    usage = workflow._usage(planned, [], time.perf_counter(), planning_input_upper=100)
    assert usage.unknown_usage_calls == 1 and usage.accounted_tokens == 1100
    assert usage.prompt_tokens is None


def test_T05_real_child_crash_after_dispatch_becomes_unknown(postgres_dsn, tmp_path):
    from story_rag.conversation_domain import LedgerError

    store, binding, llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("crash", binding)
    arrived = threading.Event()
    release = threading.Event()
    calls = []

    class Provider(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            arrived.set()
            release.wait(timeout=20)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    child = tmp_path / "dispatch_child.py"
    child.write_text(
        """import sys, urllib.request
from uuid import UUID
from story_rag.adapters.session_store import PostgresSessionStore
from story_rag.conversation_domain import DatabaseBinding, RequestPayload
store = PostgresSessionStore(sys.argv[1], schema=sys.argv[2])
binding = DatabaseBinding.model_validate_json(sys.argv[3])
session = UUID(sys.argv[4])
payload = RequestPayload(message='crash-test')
store.claim_request(session, binding, request_id='crash', payload=payload, expected_revision=0)
store.start_request(session, binding, 'crash')
store.mark_model_attempt(session, binding, 'crash', 'generation')
urllib.request.urlopen(sys.argv[5], timeout=30).read()
""",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(child),
            postgres_dsn,
            store.schema,
            binding.model_dump_json(),
            str(session.session_id),
            f"http://127.0.0.1:{server.server_port}/model",
        ],
        cwd=Path.cwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        assert arrived.wait(timeout=15), "Child did not dispatch the HTTP request"
        with pytest.raises(LedgerError) as active:
            store.recover_orphaned_request(session.session_id, binding, "crash")
        assert active.value.code == "REQUEST_IN_PROGRESS"
        process.terminate()
        process.wait(timeout=10)
        recovered = store.recover_orphaned_request(session.session_id, binding, "crash")
        assert recovered.status == "unknown" and recovered.execution_owner["attempt"] == 1
        replay = workflow.answer_turn(
            session.session_id,
            binding,
            request_id="crash",
            payload=RequestPayload(message="crash-test"),
            expected_revision=0,
        )
        assert replay.status == "unknown" and replay.revision == 0
        assert len(calls) == 1 and llm.calls == []
        assert store.list_question_versions(session.session_id, binding) == []
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
