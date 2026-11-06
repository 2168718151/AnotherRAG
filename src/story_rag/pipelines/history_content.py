from __future__ import annotations

import hashlib
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from story_rag.conversation_domain import (
    HistoryContentQuery,
    HistoryContentRef,
    LedgerError,
    QuestionVersionRecord,
    RequestRecord,
)
from story_rag.pipelines.history_resolution import HistoryScope

logger = logging.getLogger(__name__)


class HistoryContentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["matched", "waiting_input", "technical_failure"]
    text: str | None = None
    user_message: str | None = None
    ref: HistoryContentRef | None = None
    question: QuestionVersionRecord | None = None
    error_code: str | None = None
    explanation: str | None = None


class HistoryContentPipeline:
    """Scoped display recall. Stored assistant text is context, never knowledge evidence."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def read(
        self, scope: HistoryScope, request: RequestRecord, query: HistoryContentQuery
    ) -> HistoryContentResult:
        if not scope.use_history:
            raise LedgerError("HISTORY_DISABLED", "Historical content cannot be read")
        try:
            snapshot = self._store.read_history_content(
                scope.session_id, scope.binding, as_of_revision=scope.as_of_revision,
                before_turn_id=request.turn_id, query=query,
            )
        except LedgerError:
            raise
        except Exception as exc:
            logger.warning(
                "History read failed request_id=%s error_type=%s",
                request.request_id, type(exc).__name__,
            )
            return HistoryContentResult(
                status="technical_failure", error_code="HISTORY_READ_FAILED",
                explanation="历史记录读取失败，请稍后重试。",
            )
        if snapshot is None:
            return self._missing("HISTORY_NOT_FOUND", "未找到指定历史，请明确轮次或题号。")
        turn = snapshot.turn
        question = snapshot.selected_question
        if query.operation == "transform" and question is None and len(snapshot.questions) == 1:
            question = snapshot.questions[0]
        if query.operation == "followup" and question is None:
            if len(snapshot.questions) != 1:
                return self._missing(
                    "HISTORY_AMBIGUOUS", "该轮没有唯一可继续的问题，请明确要继续的题号。"
                )
            question = snapshot.questions[0]
        role = "user" if query.operation == "read_user" else "assistant"
        text = turn.user_message if role == "user" else turn.assistant_answer
        if role == "assistant" and turn.status != "succeeded":
            return self._missing(
                "HISTORY_ANSWER_UNAVAILABLE", "该次请求没有成功返回答案，无法复述未提交的候选。"
            )
        if text is None:
            if query.operation == "followup" and question is not None:
                # Continuing a known question does not require an unavailable old answer.
                return HistoryContentResult(
                    status="matched", question=question, user_message=turn.user_message
                )
            return self._missing("HISTORY_NOT_RETAINED", "该轮原文未保留，无法逐字还原。")
        if question is not None and role == "assistant":
            span = question.effective_state.get("display_answer_span")
            if span:
                start, end = span.get("start"), span.get("end")
                if (
                    not isinstance(start, int) or not isinstance(end, int)
                    or not 0 <= start <= end <= len(text)
                    or span.get("turn_id") != str(turn.turn_id)
                    or hashlib.sha256(text[start:end].encode("utf-8")).hexdigest()
                    != span.get("sha256")
                ):
                    return HistoryContentResult(
                        status="technical_failure",
                        error_code="HISTORY_MAPPING_INVALID",
                        explanation="历史分题映射校验失败，请检查保存记录。",
                    )
                text = text[start:end]
            elif len(snapshot.questions) > 1:
                return self._missing(
                    "HISTORY_ITEM_MAPPING_MISSING",
                    "该轮只保留整轮原文，缺少可靠分题映射；请改为读取该轮完整回答。",
                )
        elif question is not None and role == "user" and len(snapshot.questions) > 1:
            return self._missing(
                "HISTORY_ITEM_MAPPING_MISSING",
                "该轮用户原话包含多题，无法保证分题原文完整；请读取该轮完整提问。",
            )
        return HistoryContentResult(
            status="matched", text=text, user_message=turn.user_message, question=question,
            ref=HistoryContentRef(
                turn_id=turn.turn_id, turn_sequence=turn.turn_sequence,
                question_id=question.question_id if question else None,
                question_version=question.version if question else None,
                role=role, content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            ),
        )

    @staticmethod
    def _missing(code: str, explanation: str) -> HistoryContentResult:
        return HistoryContentResult(
            status="waiting_input", error_code=code, explanation=explanation
        )
