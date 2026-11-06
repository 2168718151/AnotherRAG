from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from llama_index.core.base.llms.types import ChatMessage


class BudgetExceededError(ValueError):
    def __init__(self, stage: str, required: int, allowed: int) -> None:
        super().__init__(f"{stage} input budget exceeded: {required} > {allowed}")
        self.stage = stage
        self.required = required
        self.allowed = allowed


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """Conservative request guardrails; provider limits must be verified separately."""

    context_limit: int = 65_536
    safety_margin: int = 4_096
    planning_output: int = 8_192
    history_output: int = 2_048
    generation_output: int = 20_480
    gate_output: int = 4_096
    generation_batch_size: int = 2
    max_tasks_per_turn: int = 12
    per_task_generation_reserve: int = 4_096
    per_task_gate_reserve: int = 768
    minimum_generation_output_headroom: int = 1_536
    minimum_gate_output_headroom: int = 2_048
    minimum_context_headroom: int = 4_096
    message_overhead: int = 128
    response_buffer_bytes: int = 131_072

    def input_limit(self, output_reserve: int) -> int:
        return self.context_limit - output_reserve - self.safety_margin

    def estimate_messages(self, messages: Sequence[ChatMessage]) -> int:
        # UTF-8 bytes upper-bound token count for byte-fallback BPE tokenizers.
        # Framing allowance covers roles and provider message serialization.
        return sum(
            len(message.role.value.encode("utf-8"))
            + len((message.content or "").encode("utf-8"))
            + self.message_overhead
            for message in messages
        )

    def estimate_text(self, text: str, *, schema_bytes: int = 0) -> int:
        return len(text.encode("utf-8")) + schema_bytes + self.message_overhead * 2

    def check(self, stage: str, input_upper_bound: int, output_reserve: int) -> None:
        allowed = self.input_limit(output_reserve) - self.minimum_context_headroom
        if input_upper_bound > allowed:
            raise BudgetExceededError(stage, input_upper_bound, allowed)
        if output_reserve + self.safety_margin > self.context_limit:
            raise BudgetExceededError(
                stage, output_reserve + self.safety_margin, self.context_limit
            )

    def snapshot(self) -> dict[str, int | str]:
        return {
            "estimator": "utf8_bytes_upper_bound_plus_message_framing",
            "context_guardrail_tokens": self.context_limit,
            "safety_margin_tokens": self.safety_margin,
            "planning_output_tokens": self.planning_output,
            "history_output_tokens": self.history_output,
            "generation_output_tokens": self.generation_output,
            "gate_output_tokens": self.gate_output,
            "generation_batch_size": self.generation_batch_size,
            "max_tasks_per_turn": self.max_tasks_per_turn,
            "per_task_generation_reserve_tokens_and_utf8_bytes": self.per_task_generation_reserve,
            "per_task_gate_reserve_tokens": self.per_task_gate_reserve,
            "minimum_generation_output_headroom_tokens": self.minimum_generation_output_headroom,
            "minimum_gate_output_headroom_tokens": self.minimum_gate_output_headroom,
            "minimum_context_headroom_tokens": self.minimum_context_headroom,
            "response_buffer_bytes": self.response_buffer_bytes,
        }
