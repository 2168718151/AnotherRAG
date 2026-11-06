from __future__ import annotations

import json

import httpx
import pytest
from llama_index.core.base.llms.types import ChatMessage

from story_rag.adapters.model_factory import BudgetedOpenAILike
from story_rag.pipelines.conversation_generation import ConversationAnswerPipeline
from tests.test_conversation_generation import _tasks
from tests.test_task_planning import _case_messages


@pytest.mark.unit
def test_phase0_final_http_body_keeps_stage_limits_and_disables_thinking() -> None:
    sent: list[dict] = []
    timeouts: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        timeouts.append(request.extensions.get("timeout", {}))
        if body.get("stream"):
            first = {
                "id": "mock-stream", "object": "chat.completion.chunk", "created": 1,
                "model": "deepseek-flash", "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "{}"},
                     "finish_reason": None}
                ],
            }
            last = {
                "id": "mock-stream", "object": "chat.completion.chunk", "created": 1,
                "model": "deepseek-flash", "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
            sse = "".join(f"data: {json.dumps(part)}\n\n" for part in (first, last))
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"},
                text=sse + "data: [DONE]\n\n",
            )
        return httpx.Response(200, json={
            "id": "mock-chat", "object": "chat.completion", "created": 1,
            "model": "deepseek-flash", "choices": [
                {"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "{}"}}
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21},
        })

    client = httpx.Client(transport=httpx.MockTransport(respond))
    llm = BudgetedOpenAILike(
        model="deepseek-flash", api_base="https://unit.invalid/v1",
        api_key="sk-fake-contract-key", is_chat_model=True,
        context_window=32768, max_tokens=8192, timeout=30, max_retries=0,
        http_client=client,
    )
    messages = [ChatMessage(role="user", content=_case_messages()[0][:100])]
    llm.chat(messages, max_tokens=8192, thinking={"type": "disabled"})
    list(llm.stream_chat(messages, max_tokens=8192, thinking={"type": "disabled"}))
    list(llm.stream_chat(messages, max_tokens=4096, thinking={"type": "disabled"}))
    list(llm.stream_chat(messages, max_tokens=2048, thinking={"type": "disabled"}))

    assert [body["max_tokens"] for body in sent] == [8192, 8192, 4096, 2048]
    assert all(body["thinking"] == {"type": "disabled"} for body in sent)
    assert all(body["model"] == "deepseek-flash" for body in sent)
    assert all(timeout.get("read") == 30 for timeout in timeouts)
    client.close()


@pytest.mark.unit
def test_phase0_pipeline_sends_generation_and_gate_budgets_to_http() -> None:
    sent: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        call = json.loads(request.content)
        sent.append(call)
        content = call["messages"][0]["content"]
        payload, _ = json.JSONDecoder().raw_decode(
            content[content.index('{"evidence_pool"'):]
        )
        rows = payload["tasks"]
        if "candidate" in rows[0]:
            items = [{
                "task_key": row["task_key"],
                "question_id": row["question_id"],
                "question_version": row["question_version"],
                "scope_ok": True,
                "covered_requirement_indexes": list(range(len(row["requirements"]))),
                "missing_requirement_indexes": [],
                "evidence_supported": True,
                "answer_relevant": True,
                "unsupported_claims": [],
            } for row in rows]
        else:
            items = [{
                "task_key": row["task_key"],
                "question_id": row["question_id"],
                "question_version": row["question_version"],
                "answer": "依据服务端权限和用户确认分别校验。",
                "evidence_ids": row["allowed_evidence_ids"],
            } for row in rows]
        body = json.dumps({"items": items}, ensure_ascii=False)
        chunks = (
            {"id": "phase0", "object": "chat.completion.chunk",
             "created": 1, "model": "deepseek-flash",
             "choices": [{"index": 0, "delta": {"role": "assistant",
                            "content": body}, "finish_reason": None}]},
            {"id": "phase0", "object": "chat.completion.chunk",
             "created": 1, "model": "deepseek-flash",
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                         for chunk in chunks) + "data: [DONE]\n\n",
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    llm = BudgetedOpenAILike(
        model="deepseek-flash", api_base="https://unit.invalid/v1",
        api_key="sk-fake-contract-key", is_chat_model=True,
        context_window=32768, max_tokens=8192, timeout=30, max_retries=0,
        http_client=client,
    )
    try:
        result = ConversationAnswerPipeline(llm).answer(_tasks(2))
        assert [item.status for item in result.items] == ["answered", "answered"]
        assert len(sent) == 2
        assert [call["max_tokens"] for call in sent] == [20480, 4096]
        assert all(call["thinking"] == {"type": "disabled"} for call in sent)
        assert all(call["model"] == "deepseek-flash" for call in sent)
        assert result.budget["expected_item_count"] == 2
        assert result.budget["produced_item_count"] == 2
    finally:
        client.close()
