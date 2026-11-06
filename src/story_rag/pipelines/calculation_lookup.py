from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AnswerMode = Literal["standard", "calculation_lookup"]


def requests_calculation(text: str) -> bool:
    return bool(re.search(
        r"计算(?!方法|原理|过程|复杂度|图|机)|算出|算一下|求出|重新算|换算|"
        r"(?:多少|多大).{0,12}(?:字节|显存|内存|[GMKT]i?B)|"
        r"(?:显存|内存|[GMKT]i?B).{0,12}(?:多少|多大)|"
        r"\b(?:calculate|compute|recalculate|convert)\b|"
        r"\bhow (?:many bytes|much (?:memory|VRAM|RAM))\b",
        text, re.I,
    ))


def resolve_answer_mode(
    mode: AnswerMode | None, text: str, previous: str | None = None
) -> AnswerMode:
    if mode == "calculation_lookup" or requests_calculation(text):
        return "calculation_lookup"
    if mode == "standard":
        return "standard"
    return "calculation_lookup" if previous == "calculation_lookup" else "standard"


class StoredQaSelection(BaseModel):
    """Selection only: the model cannot return or rewrite an answer."""

    model_config = ConfigDict(extra="forbid", strict=True)
    task_key: str
    question_id: str
    question_version: int
    candidate_id: str | None = Field(max_length=96)
    applicable: bool
    complete_answer: bool
    conditions_match: bool


def literal_numbers(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return set(re.findall(r"(?<![\d.])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\d.])", normalized))


def literals_compatible(query: str, stored_question: str) -> bool:
    # No arithmetic normalization: 1024 MiB must not be treated as 1 GiB.
    if not literal_numbers(query) <= literal_numbers(stored_question):
        return False
    return units_compatible(query, stored_question)


def units_compatible(query: str, stored_question: str) -> bool:
    def units(text: str) -> set[str]:
        values = re.findall(
            r"(?i)(?<![a-z])(?:[kmgt]i?b|bytes?|bits?|字节|比特)(?![a-z])", text
        )
        aliases = {"bytes": "byte", "字节": "byte", "bits": "bit", "比特": "bit"}
        return {aliases.get(value.casefold(), value.casefold()) for value in values}

    return units(query) <= units(stored_question)
