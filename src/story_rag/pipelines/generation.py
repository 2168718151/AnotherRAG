from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from llama_index.core.prompts import PromptTemplate

from story_rag.config import Settings
from story_rag.domain import AnswerResult, NodeTrace, QueryRequest, RetrievalResult
from story_rag.logging_config import TraceWriter

logger = logging.getLogger(__name__)

ANSWER_PROMPT = PromptTemplate(
    """You answer questions using stored Q&A text files.

Rules:
1. Use only the supplied context.
2. Do not invent facts.
3. If the context does not contain the answer, reply exactly: 无
4. Answer in the same language as the user's question unless the user asks otherwise.

Context:
{context_str}

Question:
{query_str}

Answer:"""
)

RELEVANCE_PROMPT = PromptTemplate(
    """Judge whether the answer directly addresses the question.
Return exactly Yes or No. Do not explain.

Question:
{query_str}

Answer:
{answer_str}

Judgement:"""
)


class StoryGenerationPipeline:
    def __init__(
        self,
        settings: Settings,
        llm: Any,
        retrieval_pipeline: Any,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._retrieval_pipeline = retrieval_pipeline
        self._trace_writer = trace_writer

    def answer(
        self,
        request: QueryRequest,
        retrieval: RetrievalResult | None = None,
    ) -> AnswerResult:
        result, _ = self._answer_with_trace(request, retrieval)
        return result

    def answer_trace(
        self,
        request: QueryRequest,
        retrieval: RetrievalResult | None = None,
    ) -> dict[str, Any]:
        result, trace = self._answer_with_trace(request, retrieval)
        return {"result": result.model_dump(), **trace}

    def prepare_llm_input(
        self,
        request: QueryRequest,
        retrieval: RetrievalResult | None = None,
    ) -> dict[str, Any]:
        resolved = (
            retrieval if retrieval is not None else self._retrieval_pipeline.retrieve(request)
        )
        contexts = [node.text for node in resolved.nodes]
        if not contexts:
            return {
                "query": request.query,
                "model": self._settings.llm_model,
                "would_call_llm": False,
                "context_str": "",
                "contexts": [],
                "nodes": [],
                "prompt": None,
                "messages": [],
            }

        context_str = _format_context(resolved.nodes)
        prompt_args = {"context_str": context_str, "query_str": request.query}
        messages = ANSWER_PROMPT.format_messages(llm=self._llm, **prompt_args)
        return {
            "query": request.query,
            "model": self._settings.llm_model,
            "would_call_llm": True,
            "context_str": context_str,
            "contexts": contexts,
            "nodes": [node.model_dump() for node in resolved.nodes],
            "prompt": ANSWER_PROMPT.format(**prompt_args),
            "messages": [
                {"role": message.role.value, "content": message.content} for message in messages
            ],
        }

    def _answer_with_trace(
        self,
        request: QueryRequest,
        retrieval: RetrievalResult | None,
    ) -> tuple[AnswerResult, dict[str, Any]]:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        resolved_retrieval = (
            retrieval if retrieval is not None else self._retrieval_pipeline.retrieve(request)
        )
        llm_input = self.prepare_llm_input(request, resolved_retrieval)
        contexts = llm_input["contexts"]

        if not contexts:
            result = AnswerResult(
                request_id=request_id,
                status="no_answer",
                answer="无",
                relevance_yes=False,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            return result, {"reason": "no_context"}

        context_str = llm_input["context_str"]
        prompt = llm_input["prompt"]
        raw_answer = (
            str(
                self._llm.predict(ANSWER_PROMPT, context_str=context_str, query_str=request.query)
            ).strip()
            or "无"
        )

        if raw_answer == "无":
            result = AnswerResult(
                request_id=request_id,
                status="no_answer",
                answer="无",
                relevance_yes=False,
                contexts=contexts,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            return result, {"prompt": prompt, "raw_answer": raw_answer, "reason": "llm_no_answer"}

        gate_prompt = RELEVANCE_PROMPT.format(
            query_str=request.query,
            answer_str=raw_answer,
        )
        gate_raw = str(
            self._llm.predict(RELEVANCE_PROMPT, query_str=request.query, answer_str=raw_answer)
        ).strip()
        relevance_yes = _parse_yes_no(gate_raw)
        status = "answered" if relevance_yes else "no_answer"
        answer = raw_answer if relevance_yes else "无"
        citations = [_citation(node) for node in resolved_retrieval.nodes]
        result = AnswerResult(
            request_id=request_id,
            status=status,
            answer=answer,
            relevance_yes=relevance_yes,
            citations=citations if relevance_yes else [],
            contexts=contexts,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        trace = {
            "prompt": prompt,
            "raw_answer": raw_answer,
            "gate_prompt": gate_prompt,
            "gate_raw": gate_raw,
        }
        logger.info(
            "Generated request=%s status=%s relevance=%s latency_ms=%.2f",
            request_id,
            status,
            relevance_yes,
            result.latency_ms,
        )
        if self._trace_writer:
            self._trace_writer.write(
                request_id,
                "generation",
                {"result": result.model_dump(), **trace},
            )
        return result, trace


def _format_context(nodes: list[NodeTrace]) -> str:
    parts = []
    for index, node in enumerate(nodes, start=1):
        source = str(node.metadata.get("source", "unknown"))
        filename = node.metadata.get("display_name") or source.replace("\\", "/").rsplit("/", 1)[-1]
        parts.append(
            f"[{index}] source={source} filename={filename} "
            f"chunk={node.metadata.get('chunk_index', '?')}\n{node.text}"
        )
    return "\n\n".join(parts)


def _parse_yes_no(value: str) -> bool:
    match = re.search(r"\b(yes|no)\b", value.strip(), flags=re.IGNORECASE)
    return bool(match and match.group(1).casefold() == "yes")


def _citation(node: NodeTrace) -> str:
    source = node.metadata.get("source", "unknown")
    chunk = node.metadata.get("chunk_index", "?")
    return f"{source}#chunk-{chunk}"
