from __future__ import annotations

import json

from scripts.run_qa_benchmark import export_llm_inputs, reference_rank
from story_rag.domain import NodeTrace


def test_reference_rank_ignores_whitespace_and_unicode_width() -> None:
    nodes = [
        NodeTrace(node_id="n1", text="无关内容", retriever="rerank", rank=1),
        NodeTrace(node_id="n2", text="答案：使用 RRF 融合。", retriever="rerank", rank=2),
    ]

    assert reference_rank("使用ＲＲＦ融合。", nodes) == 2


def test_llm_input_export_omits_reference_field(tmp_path) -> None:
    results = tmp_path / "results.jsonl"
    exported = tmp_path / "llm_inputs.jsonl"
    results.write_text(
        json.dumps(
            {
                "case": {"case_id": "case-001", "expected_answer": "private reference"},
                "llm_input": {
                    "model": "deepseek-flash",
                    "would_call_llm": True,
                    "messages": [{"role": "user", "content": "Question only"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert export_llm_inputs(results, exported) == 1
    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["case_id"] == "case-001"
    assert payload["messages"] == [{"role": "user", "content": "Question only"}]
    assert "expected_answer" not in payload
    assert "private reference" not in exported.read_text(encoding="utf-8")
