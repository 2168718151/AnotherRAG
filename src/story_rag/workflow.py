from __future__ import annotations

import asyncio
from typing import Any

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

from story_rag.domain import AnswerResult, QueryRequest, RetrievalResult
from story_rag.logging_config import TraceWriter


class QueryStarted(StartEvent):
    request: QueryRequest


class RetrievalFinished(Event):
    request: QueryRequest
    result: RetrievalResult


class StoryRagWorkflow(Workflow):
    def __init__(
        self,
        retrieval_pipeline: Any,
        generation_pipeline: Any,
        trace_writer: TraceWriter | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._retrieval_pipeline = retrieval_pipeline
        self._generation_pipeline = generation_pipeline
        self._trace_writer = trace_writer

    @step
    async def retrieve(self, ev: QueryStarted) -> RetrievalFinished:
        result = await asyncio.to_thread(self._retrieval_pipeline.retrieve, ev.request)
        return RetrievalFinished(request=ev.request, result=result)

    @step
    async def generate(self, ev: RetrievalFinished) -> StopEvent:
        result = await asyncio.to_thread(
            self._generation_pipeline.answer,
            ev.request,
            ev.result,
        )
        return StopEvent(result=result)

    async def answer(self, request: QueryRequest) -> AnswerResult:
        result = await self.run(request=request)
        if not isinstance(result, AnswerResult):
            raise TypeError(f"Unexpected workflow result: {type(result)!r}")
        return result

    async def stream_trace(self, request: QueryRequest) -> list[dict[str, Any]]:
        retrieval = await asyncio.to_thread(self._retrieval_pipeline.retrieve_trace, request)
        generation = await asyncio.to_thread(
            self._generation_pipeline.answer_trace,
            request,
            retrieval,
        )
        events = [
            {"event": "retrieval", "data": retrieval.model_dump()},
            {"event": "generation", "data": generation},
        ]
        if self._trace_writer:
            self._trace_writer.write(
                f"workflow-{abs(hash(request.query))}",
                "workflow",
                {"events": events},
            )
        return events
