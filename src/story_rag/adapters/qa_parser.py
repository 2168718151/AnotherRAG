from __future__ import annotations

import re
from dataclasses import dataclass

_QUESTION_LINE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:Q|问|问题)\s*(?:\d+)?\s*[:：]\s*(.*)$",
    re.IGNORECASE,
)
_ANSWER_LINE = re.compile(r"^\s*(?:A|答|答案)\s*[:：]", re.IGNORECASE)
_QA_SECTION = "技术问题 -> 答案"


@dataclass(frozen=True, slots=True)
class QaPair:
    question: str
    text: str

    @property
    def answer(self) -> str:
        lines = self.text.splitlines()
        for index, line in enumerate(lines[1:], start=1):
            marker = _ANSWER_LINE.match(line)
            if marker:
                return "\n".join([line[marker.end():], *lines[index + 1:]]).strip()
        return ""


def extract_qa_pairs(text: str) -> list[QaPair]:
    lines = text.splitlines()
    section_start = next(
        (index + 1 for index, line in enumerate(lines) if line.strip() == _QA_SECTION),
        0,
    )
    lines = lines[section_start:]
    starts = [index for index, line in enumerate(lines) if _QUESTION_LINE.match(line)]
    pairs: list[QaPair] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        block = lines[start:end]
        if not any(_ANSWER_LINE.match(line) for line in block[1:]):
            continue
        question = _QUESTION_LINE.match(block[0])
        if question is None or not question.group(1).strip():
            continue
        pairs.append(QaPair(question=question.group(1).strip(), text="\n".join(block).strip()))
    return pairs
