from __future__ import annotations

from story_rag.adapters.model_factory import build_llm
from story_rag.config import Settings
from story_rag.domain import NodeTrace, QueryRequest, RetrievalResult
from story_rag.pipelines.generation import ANSWER_PROMPT
from tests.conftest import ScriptedLLM

CHINESE_ANSWER = "白龙穿过云层，把雨水带给干涸的村庄。"


def _retrieval_result(*, text: str | None = None) -> RetrievalResult:
    nodes = []
    if text is not None:
        nodes.append(
            NodeTrace(
                node_id="node-1",
                text=text,
                retriever="rerank",
                rank=1,
                score=1.0,
                rerank_score=1.0,
                metadata={"source": "dragon.txt", "chunk_index": 0},
            )
        )
    return RetrievalResult(query="白龙把雨水带给了谁？", nodes=nodes)


def test_no_context_returns_no_answer_without_calling_llm(
    container,
    scripted_llm: ScriptedLLM,
) -> None:
    result = container.generation.answer(
        QueryRequest(query="白龙把雨水带给了谁？"),
        _retrieval_result(),
    )

    assert result.answer == "无"
    assert result.status == "no_answer"
    assert result.relevance_yes is False
    assert scripted_llm.prompts == []


def test_total_gate_no_returns_no_answer(container, scripted_llm: ScriptedLLM) -> None:
    scripted_llm.responses = [CHINESE_ANSWER, "No"]

    result = container.generation.answer(
        QueryRequest(query="今天天气如何？"),
        _retrieval_result(text="白龙穿过云层，把雨水带给干涸的村庄。"),
    )

    assert result.status == "no_answer"
    assert result.answer == "无"
    assert result.relevance_yes is False
    assert result.citations == []
    assert len(scripted_llm.prompts) == 2


def test_total_gate_yes_returns_answer_with_citations(
    container, scripted_llm: ScriptedLLM
) -> None:
    scripted_llm.responses = [CHINESE_ANSWER, "Yes"]

    result = container.generation.answer(
        QueryRequest(query="白龙把雨水带给了谁？"),
        _retrieval_result(text="白龙穿过云层，把雨水带给干涸的村庄。"),
    )

    assert result.status == "answered"
    assert result.answer == CHINESE_ANSWER
    assert result.relevance_yes is True
    assert result.citations == ["dragon.txt#chunk-0"]
    assert result.contexts


def test_llm_no_answer_skips_gate(container, scripted_llm: ScriptedLLM) -> None:
    scripted_llm.responses = ["无"]

    result = container.generation.answer(
        QueryRequest(query="白龙把雨水带给了谁？"),
        _retrieval_result(text="白龙穿过云层。"),
    )

    assert result.answer == "无"
    assert result.status == "no_answer"
    assert len(scripted_llm.prompts) == 1


def test_debug_port_exposes_prompt_and_gate(container, scripted_llm: ScriptedLLM) -> None:
    scripted_llm.responses = [CHINESE_ANSWER, "yes, the answer matches"]

    trace = container.generation.answer_trace(
        QueryRequest(query="白龙把雨水带给了谁？"),
        _retrieval_result(text="白龙穿过云层，把雨水带给干涸的村庄。"),
    )

    assert "白龙穿过云层" in trace["prompt"]
    assert trace["raw_answer"] == CHINESE_ANSWER
    assert "yes" in trace["gate_raw"].casefold()
    assert trace["result"]["status"] == "answered"


def test_prepare_llm_input_matches_generation_prompt_without_calling_llm(
    container, scripted_llm: ScriptedLLM
) -> None:
    request = QueryRequest(query="白龙把雨水带给了谁？")
    retrieval = _retrieval_result(text="白龙把雨水带给干涸的村庄。")

    llm_input = container.generation.prepare_llm_input(request, retrieval)

    assert scripted_llm.prompts == []
    assert llm_input["would_call_llm"] is True
    assert llm_input["messages"] == [{"role": "user", "content": llm_input["prompt"]}]
    assert llm_input["contexts"] == ["白龙把雨水带给干涸的村庄。"]

    scripted_llm.responses = [CHINESE_ANSWER, "Yes"]
    container.generation.answer(request, retrieval)
    assert llm_input["prompt"] in scripted_llm.prompts[0]


def test_prepare_llm_input_has_no_message_without_context(container) -> None:
    llm_input = container.generation.prepare_llm_input(
        QueryRequest(query="没有资料？"), _retrieval_result()
    )

    assert llm_input["would_call_llm"] is False
    assert llm_input["messages"] == []
    assert llm_input["prompt"] is None


def test_prepared_messages_match_openai_like_request(
    container, settings: Settings
) -> None:
    request = QueryRequest(query="白龙把雨水带给了谁？")
    llm_input = container.generation.prepare_llm_input(
        request, _retrieval_result(text="白龙把雨水带给干涸的村庄。")
    )
    production_llm = build_llm(settings)

    actual_messages = production_llm._get_messages(
        ANSWER_PROMPT,
        context_str=llm_input["context_str"],
        query_str=request.query,
    )

    assert llm_input["messages"] == [
        {"role": message.role.value, "content": message.content}
        for message in actual_messages
    ]
