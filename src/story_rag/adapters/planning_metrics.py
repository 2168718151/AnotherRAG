from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

from story_rag.pipelines.conversation_generation import _read_finish_reason, _read_usage

planning_responses: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "planning_responses", default=None
)


class PlanningMetricsHandler(BaseCallbackHandler):
    """Collect provider usage from native LlamaIndex events, scoped to one call."""

    def __init__(self) -> None:
        super().__init__([], [])

    def on_event_start(self, event_type, payload=None, event_id="", parent_id="", **kwargs):
        return event_id

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        target = planning_responses.get()
        if target is None or event_type != CBEventType.LLM or not payload:
            return
        response = payload.get(EventPayload.RESPONSE) or payload.get(EventPayload.COMPLETION)
        raw = getattr(response, "raw", None)
        usage = _read_usage(raw)
        target.append(
            {
                "prompt_tokens": usage[0] if usage else None,
                "completion_tokens": usage[1] if usage else None,
                "finish_reason": _read_finish_reason(raw),
            }
        )

    def start_trace(self, trace_id=None):
        pass

    def end_trace(self, trace_id=None, trace_map=None):
        pass
