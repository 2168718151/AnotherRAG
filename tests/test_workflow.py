from __future__ import annotations

from story_rag.domain import QueryRequest
from tests.conftest import CHINESE_STORY, ScriptedLLM

CHINESE_ANSWER = "白龙穿过云层，把雨水带给干涸的村庄。"


async def test_workflow_returns_answer_result(container, scripted_llm: ScriptedLLM) -> None:
    container.ingestion.ingest_bytes(CHINESE_STORY.encode("utf-8"), source_name="dragon.txt")
    scripted_llm.responses = [CHINESE_ANSWER, "Yes"]

    result = await container.workflow.answer(QueryRequest(query="白龙把雨水带给了谁？"))

    assert result.status == "answered"
    assert result.answer == CHINESE_ANSWER
    assert result.request_id


async def test_workflow_stream_trace_has_two_events(container, scripted_llm: ScriptedLLM) -> None:
    container.ingestion.ingest_bytes(CHINESE_STORY.encode("utf-8"), source_name="dragon.txt")
    scripted_llm.responses = [CHINESE_ANSWER, "Yes"]

    events = await container.workflow.stream_trace(QueryRequest(query="白龙把雨水带给了谁？"))

    assert [event["event"] for event in events] == ["retrieval", "generation"]
    assert events[0]["data"]["nodes"]
    assert events[1]["data"]["result"]["answer"] == CHINESE_ANSWER


async def test_workflow_without_documents_returns_no(container) -> None:
    result = await container.workflow.answer(QueryRequest(query="白龙把雨水带给了谁？"))

    assert result.answer == "无"
    assert result.status == "no_answer"
