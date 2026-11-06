from __future__ import annotations

import json
from uuid import uuid4

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse

from story_rag.adapters.qa_parser import extract_qa_pairs
from story_rag.conversation_domain import EvidenceRef, RequestPayload
from story_rag.pipelines.calculation_lookup import literals_compatible, resolve_answer_mode
from story_rag.pipelines.conversation_generation import (
    ConversationAnswerPipeline,
    ConversationTaskInput,
)
from story_rag.pipelines.evidence_mapping import (
    EvidenceMappingPipeline,
    EvidencePoolEntry,
)
from story_rag.pipelines.history_resolution import HistoryScope
from story_rag.pipelines.task_planning import SourceBackedText, TaskPlan
from story_rag.pipelines.token_budget import BudgetExceededError, TokenBudget
from tests.test_conversation_generation import BatchLLM, _tasks
from tests.test_conversation_ledger import postgres_dsn
from tests.test_conversation_workflow import _workflow
from tests.test_evidence_mapping import FixedKnowledgeStore, NoisyRetriever

__all__ = ["postgres_dsn"]
pytestmark = pytest.mark.contract
QUESTION = "Calculate KV cache for batch=2, seq=4, FP32 in GiB"
PAIR = f"Q: {QUESTION}\nA: 这组配置的已有答案为 0.5 GiB。"


class LookupLLM(BatchLLM):
    selection_mode: str = "match"

    def stream_chat(self, messages: list[ChatMessage], **kwargs):
        content = "\n".join(message.content or "" for message in messages)
        if '"stored_qa_candidates"' not in content:
            yield from super().stream_chat(messages, **kwargs)
            return
        self.prompts.append(content)
        self.calls.append("qa_match")
        payload, _ = json.JSONDecoder().raw_decode(content[content.index('{"task_key"'):])
        result = {
            key: payload[key] for key in ("task_key", "question_id", "question_version")
        }
        match = self.selection_mode not in {"reject", "wrong_fields"}
        result.update(
            candidate_id="qa1" if match else None,
            applicable=match, complete_answer=match, conditions_match=match,
        )
        if self.selection_mode == "unknown":
            result["candidate_id"] = "another-task-qa"
        if self.selection_mode == "identity":
            result["question_id"] = str(uuid4())
        if self.selection_mode == "invent":
            result["answer"] = "LLM 自行计算的答案"
        body = "invalid JSON" if self.selection_mode == "invalid" else json.dumps(result)
        yield ChatResponse(
            message=ChatMessage(role="assistant", content=body), delta=body,
            raw={"choices": [{"finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 50}},
        )


def _task(body: str | None = PAIR, **overrides) -> ConversationTaskInput:
    data = {
        "task_key": "t1", "question_id": uuid4(), "question_version": 1,
        "question_text": QUESTION, "requirements": [QUESTION],
        "answer_mode": "calculation_lookup",
        "evidence": [] if body is None else [EvidencePoolEntry(
            evidence_id="verified-qa", ref=EvidenceRef(document_id="doc", node_id="node"),
            text=body, source="fixture.txt",
        )],
    }
    data.update(overrides)
    return ConversationTaskInput(**data)


def test_existing_answer_is_copied_and_never_generated() -> None:
    llm = LookupLLM()
    pipeline = ConversationAnswerPipeline(llm)
    task = _task()
    preview = pipeline.preview([task], include_messages=True)
    assert llm.calls == []
    assert preview["batches"] == []
    assert preview["calculation_lookups"][0]["candidate_ids"] == ["qa1"]
    result = pipeline.answer([task])
    assert result.items[0].answer == extract_qa_pairs(PAIR)[0].answer
    assert result.items[0].status == "answered"
    assert result.items[0].evidence_refs == [task.evidence[0].ref]
    assert llm.calls == ["qa_match"]
    assert result.metrics[0].stage == "qa_match"
    assert result.budget["settled_tokens"] == 150
    assert result.budget["calculation_lookup_count"] == 1


@pytest.mark.parametrize("body", [
    None,
    "Q: Generic KV cache formula\nA: bytes = 2 * L * B * S * H * D",
    f"Q: {QUESTION}\nA: 公式为 bytes = 2 * L * B * S * H * D",
    PAIR.replace("batch=2", "batch=3"),
    PAIR.replace("FP32", "FP16"),
    PAIR.replace("GiB", "MiB"),
    f"Q: {QUESTION}\nA: 无",
    "历史助手算过是 0.5 GiB。",
])
def test_no_applicable_pair_is_not_computed(body: str | None) -> None:
    llm = LookupLLM()
    task = _task(body, user_facts=[{"field": "formula", "source_excerpt": "2 * 4 = 8"}],
                 direct_fact_reasoning=True)
    result = ConversationAnswerPipeline(llm).answer([task])
    assert result.items[0].status == "no_evidence"
    assert result.items[0].answer == "无"
    assert llm.calls == []


@pytest.mark.parametrize("mode,code", [
    ("unknown", "QA_SELECTION_SCOPE_VIOLATION"),
    ("identity", "QA_SELECTION_SCOPE_VIOLATION"),
    ("invent", "INVALID_QA_SELECTION"),
    ("invalid", "INVALID_QA_SELECTION"),
])
def test_invalid_selection_is_failure_not_missing_knowledge(mode: str, code: str) -> None:
    llm = LookupLLM(selection_mode=mode)
    item = ConversationAnswerPipeline(llm).answer([_task()]).items[0]
    assert item.status == "technical_failure"
    assert item.error_code == code
    assert item.answer != "无"
    assert llm.calls == ["qa_match"]


def test_semantic_match_rejects_swapped_field_values() -> None:
    llm = LookupLLM(selection_mode="wrong_fields")
    item = ConversationAnswerPipeline(llm).answer([
        _task(PAIR.replace("batch=2, seq=4", "batch=4, seq=2"))
    ]).items[0]
    assert item.status == "no_evidence"
    assert item.answer == "无"
    assert llm.calls == ["qa_match"]


def test_mixed_tasks_and_retrieval_failure_stay_separate() -> None:
    llm = LookupLLM()
    normal = _tasks(1)[0].model_copy(update={"task_key": "t2"})
    failure = _task(None, task_key="t3", evidence_decision="technical_failure",
                    error_code="STORAGE_UNAVAILABLE")
    result = ConversationAnswerPipeline(llm).answer([_task(None), normal, failure])
    assert [item.status for item in result.items] == [
        "no_evidence", "answered", "technical_failure"
    ]
    assert llm.calls == ["generation", "gate"]


def test_matching_pair_does_not_accept_a_different_parameter_or_unit() -> None:
    assert not literals_compatible("batch=11", "batch=1")
    assert not literals_compatible("FP32", "FP16")
    assert not literals_compatible("1024 MiB", "1 GiB")
    assert not literals_compatible("1 GiB", "1 MiB")
    assert resolve_answer_mode(None, "batch 改成 11", "calculation_lookup") == (
        "calculation_lookup"
    )
    assert resolve_answer_mode("standard", "解释原理", "calculation_lookup") == "standard"
    assert resolve_answer_mode(None, "解释计算图") == "standard"


def test_paraphrase_and_completed_pair_are_available_without_answer_rewriting() -> None:
    llm = LookupLLM()
    task = _task("Q: truncated fragment\nA: incomplete")
    task.evidence[0].complete_qa_pairs = [PAIR]
    task.question_text = "How much memory for KV cache, batch=2, seq=4, FP32 in GiB?"
    item = ConversationAnswerPipeline(llm).answer([task]).items[0]
    assert item.status == "answered"
    assert item.answer == extract_qa_pairs(PAIR)[0].answer
    assert '"answer":"incomplete"' not in llm.prompts[0]


def test_oversized_stored_answer_is_not_silently_cut() -> None:
    llm = LookupLLM()
    item = ConversationAnswerPipeline(llm).answer([
        _task(PAIR + "长" * 2000)
    ]).items[0]
    assert item.status == "technical_failure"
    assert item.error_code == "STORED_ANSWER_TOO_LONG"
    assert item.answer == ""


def test_qa_selection_budget_is_checked_before_calls() -> None:
    llm = LookupLLM()
    budget = TokenBudget(context_limit=6000, safety_margin=100, minimum_context_headroom=100)
    with pytest.raises(BudgetExceededError):
        ConversationAnswerPipeline(llm, budget).answer([_task()])
    assert llm.calls == []


def test_ambiguous_history_and_theory_modes_are_not_overridden() -> None:
    llm = LookupLLM()
    pipeline = ConversationAnswerPipeline(llm)
    ambiguous = _task(evidence_decision="ambiguous", missing_inputs=["指哪个对象？"])
    item = pipeline.answer([ambiguous]).items[0]
    assert item.status == "ambiguous"
    assert llm.calls == []
    theory = _tasks(1)[0].model_copy(update={
        "answer_mode": "standard", "question_text": "KV Cache 计算；任务：仅解释其原理"
    })
    assert pipeline.answer([theory]).items[0].status == "answered"
    assert llm.calls == ["generation", "gate"]


def _plan(message: str, *, followup: bool = False, mode=None) -> TaskPlan:
    return TaskPlan.model_validate({
        "schema_version": 2,
        "goals": [{"goal_key": "g1", "goal_text": message,
                   "source_excerpt": message, "task_keys": ["t1"]}],
        "tasks": [{
            "task_key": "t1", "intent": "revise" if followup else "answer",
            "answer_mode": mode,
            "objective": {"text": message, "source_excerpt": message},
            "object_scope": {"text": "KV cache", "source_excerpt": "KV cache"}
            if not followup else None,
            "object_alias": "kv",
            "history_need": "required" if followup else "none",
            "history_question_no": 1 if followup else None,
            "conditions": [{
                "field": "batch_size", "scope": "object",
                "op": "replace" if followup else "supplement",
                "value": "11" if followup else "2", "source_excerpt": message,
            }],
            "information_gaps": [{
                "description": "缺少 layers 参数", "state": "not_provided",
                "basis_excerpt": message, "origin": "user_input", "necessary": True,
            }],
        }],
    })


def test_multiturn_changed_parameter_looks_up_again_without_waiting(postgres_dsn: str) -> None:
    store, binding, _llm, workflow = _workflow(postgres_dsn)
    knowledge = FixedKnowledgeStore(binding)
    knowledge.add("kv.txt", PAIR)
    retriever = NoisyRetriever(knowledge)
    workflow._evidence = EvidenceMappingPipeline(store, knowledge, retriever)
    llm = LookupLLM()
    workflow._generator = ConversationAnswerPipeline(llm)
    session = store.create_session("计算题仅查库", binding)
    first = workflow.answer_turn(
        session.session_id, binding, request_id="lookup-1", expected_revision=0,
        payload=RequestPayload(message=QUESTION), plan=_plan(QUESTION), diagnostic=True,
    )
    assert first.status == "succeeded", first.error_code
    assert first.items[0].status == "answered"
    assert llm.calls == ["qa_match"]
    second = workflow.answer_turn(
        session.session_id, binding, request_id="lookup-2", expected_revision=1,
        payload=RequestPayload(message="batch 改成 11"),
        plan=_plan("batch 改成 11", followup=True), diagnostic=True,
    )
    assert second.status == "succeeded", second.error_code
    assert second.items[0].status == "no_evidence"
    assert second.items[0].answer == "无"
    assert second.items[0].question_id == first.items[0].question_id
    assert len(retriever.calls) == 2
    assert "batch_size=11" in retriever.calls[-1]
    assert llm.calls == ["qa_match"]
    knowledge.add("kv-batch11.txt", PAIR.replace("batch=2", "batch=11"))
    third = workflow.answer_turn(
        session.session_id, binding, request_id="lookup-3", expected_revision=2,
        payload=RequestPayload(message="batch 改成 11"),
        plan=_plan("batch 改成 11", followup=True), diagnostic=True,
    )
    assert third.status == "succeeded", third.error_code
    assert third.items[0].status == "answered"
    assert third.items[0].answer == extract_qa_pairs(PAIR)[0].answer
    assert len(retriever.calls) == 3
    assert llm.calls == ["qa_match", "qa_match"]


def test_theory_and_calculation_followups_are_not_coalesced(postgres_dsn: str) -> None:
    store, binding, _llm, workflow = _workflow(postgres_dsn)
    session = store.create_session("理论与计算独立", binding)
    # No Q&A in the fixed evidence: first turn records the question with no_evidence.
    first = workflow.answer_turn(
        session.session_id, binding, request_id="mixed-first", expected_revision=0,
        payload=RequestPayload(message=QUESTION), plan=_plan(QUESTION), diagnostic=True,
    )
    assert first.status == "succeeded", first.error_code
    message = "解释KV cache原理，再计算其具体大小"
    theory = _plan(message, followup=True, mode="standard").tasks[0]
    theory.intent = "answer"
    theory.objective.text = "解释KV cache原理"
    theory.conditions = []
    theory.information_gaps = []
    theory.history_reference = SourceBackedText(text="KV cache", source_excerpt=message)
    calculation = theory.model_copy(deep=True)
    calculation.task_key = "t2"
    calculation.intent = "revise"
    calculation.answer_mode = "calculation_lookup"
    calculation.objective.text = "计算其具体大小"
    candidate = TaskPlan.model_validate({
        "schema_version": 2,
        "goals": [{"goal_key": "g1", "goal_text": message,
                   "source_excerpt": message, "task_keys": ["t1", "t2"]}],
        "tasks": [theory, calculation],
    })
    scope = HistoryScope(session_id=session.session_id, binding=binding, as_of_revision=1)
    coalesced = workflow._coalesce_explanation_revisions(scope, candidate)
    assert len(coalesced.tasks) == 2
    assert [task.answer_mode for task in coalesced.tasks] == ["standard", "calculation_lookup"]
