from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from story_rag.adapters.model_factory import build_conversation_llm
from story_rag.config import Settings
from story_rag.conversation_domain import RequestPayload
from story_rag.pipelines.conversation_generation import ConversationAnswerPipeline
from story_rag.pipelines.task_planning import TaskPlanningPipeline
from tests.conftest import ScriptedLLM
from tests.test_conversation_ledger import _case_messages, postgres_dsn
from tests.test_conversation_workflow import _workflow
from tests.test_task_planning import _frozen_plan

__all__ = ["postgres_dsn"]


@pytest.mark.integration
@pytest.mark.parametrize("planning_mode", ["frozen", "live"])
def test_phase0_five_turns_with_real_generation_in_isolated_ledger(
    postgres_dsn: str, planning_mode: str,
) -> None:
    enabled = (
        "STORY_RAG_RUN_LIVE_PHASE0" if planning_mode == "frozen"
        else "STORY_RAG_RUN_LIVE_PLANNING"
    )
    if os.getenv(enabled) != "1":
        pytest.skip(f"Real LLM calls require {enabled}=1")

    root = Path(__file__).resolve().parents[1]
    settings = Settings.from_env(root, require_reranker=False)
    store, binding, _, workflow = _workflow(postgres_dsn)
    llm = build_conversation_llm(settings)
    raw_calls = []
    workflow._generator = ConversationAnswerPipeline(
        llm, diagnostic_sink=lambda stage, body, metric: raw_calls.append({
            "stage": stage, "body": body,
            "metric": metric.model_dump(mode="json"),
        })
    )
    workflow._planner = TaskPlanningPipeline(llm)
    session = store.create_session("方案零真实生成", binding)
    planner = TaskPlanningPipeline(ScriptedLLM())
    records = []
    for index, message in enumerate(_case_messages(), start=1):
        call_start = len(raw_calls)
        reference_plan = planner.validate_candidate(
            message, _frozen_plan(message, index)
        ).plan
        assert reference_plan is not None
        plan = reference_plan if planning_mode == "frozen" else None
        result = workflow.answer_turn(
            session.session_id, binding,
            request_id=f"phase0-live-{index}-{uuid4().hex[:8]}",
            expected_revision=index - 1, payload=RequestPayload(message=message),
            plan=plan, diagnostic=True,
        )
        generation = result.trace.get("generation", {})
        records.append({
            "turn": index,
            "message": message,
            "status": result.status,
            "revision": result.revision,
            "error_code": result.error_code,
            "validation_errors": result.trace.get("validation_errors", []),
            "reference_items": len(reference_plan.tasks),
            "expected_items": (
                len(plan.tasks) if plan else result.trace.get("task_count", 0)
            ),
            "items": [item.model_dump(mode="json") for item in result.items],
            "budget": result.trace.get("budget", {}),
            "planning_budget": result.trace.get("planning_budget", {}),
            "metrics": generation.get("metrics", []),
            "generation_failures": result.trace.get("generation_failures", []),
            "failure_metrics": result.trace.get("generation_metrics", []),
            "plan": result.trace.get("plan"),
            "history": result.trace.get("history"),
            "generation_messages": result.trace.get("messages"),
            "raw_model_calls": raw_calls[call_start:],
            "usage": result.request.usage.model_dump(mode="json"),
        })
        if result.status != "succeeded":
            break

    output = root / "reports" / (
        "phase56_live_" + planning_mode + "_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "real_generation.json").write_text(
        json.dumps({
            "model": settings.llm_model,
            "base_url": settings.deepseek_base_url,
            "planning": (
                "frozen plan; no planning API calls" if planning_mode == "frozen"
                else "real API planning; no frozen plan injected"
            ),
            "evidence": "phase-zero fixed Mock evidence; no real retrieval",
            "turns": records,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assert len(records) == 5, f"Live trace: {output}"
    assert all(record["status"] == "succeeded" for record in records), output
    allowed_task_counts = {1: {3}, 2: {2}, 3: {2}, 4: {1}, 5: {1, 2}}
    for record in records:
        assert len(record["items"]) == record["expected_items"]
        assert record["budget"]["expected_item_count"] == record["expected_items"]
        assert record["budget"]["produced_item_count"] == record["expected_items"]
        assert sum(record["budget"]["status_counts"].values()) == record["expected_items"]
        assert len({item["task_key"] for item in record["items"]}) == record["expected_items"]
        assert len({item["question_id"] for item in record["items"]}) == record["expected_items"]
        for metric in record["metrics"]:
            assert metric["finish_reason"] == "stop", output
            if metric["usage_known"]:
                assert metric["reported_prompt_tokens"] <= metric["input_upper_bound"], output
                assert (
                    metric["max_output_tokens"] - metric["reported_completion_tokens"]
                    >= 1024
                ), output
        assert all(batch["generation_output_headroom"] >= 1536
                   and batch["gate_output_headroom"] >= 2048
                   and batch["generation_context_headroom"] >= 4096
                   and batch["gate_context_headroom"] >= 4096
                   for batch in record["budget"]["batches"])
        if planning_mode == "live":
            assert record["planning_budget"]["context_headroom"] >= 4096, output
            assert record["expected_items"] in allowed_task_counts[record["turn"]], output
            if record["turn"] > 1:
                assert all(item["status"] == "answered"
                           for item in record["items"]), output
    if planning_mode == "live":
        first_items = records[0]["items"]
        assert [item["status"] for item in first_items] == [
            "answered", "answered", "waiting_input"
        ], output
        for turn in (3, 4):
            draft_answer = records[turn - 1]["items"][0]["answer"].lower()
            assert re.search(r"multi[- ]tenan(?:t|cy)|多租户", draft_answer), output
            assert re.search(r"queu|队列|排队|入队", draft_answer), output
            assert re.search(r"shared[- ]service[- ]account|共享服务", draft_answer), output
        third_draft_id = records[2]["items"][0]["question_id"]
        assert records[3]["items"][0]["question_id"] == third_draft_id, output
        assert records[3]["items"][0]["question_id"] != first_items[0]["question_id"], output
        assert records[4]["items"][0]["question_id"] == first_items[2]["question_id"], output
        assert all(
            item["question_id"] not in {
                first_items[0]["question_id"], records[3]["items"][0]["question_id"]
            }
            for item in records[4]["items"][1:]
        ), output
        assert "missing_write_scope" in "\n".join(
            item["answer"] for item in records[4]["items"]
        ), output
