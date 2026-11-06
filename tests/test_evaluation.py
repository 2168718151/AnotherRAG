from __future__ import annotations

import json

import pytest
from llama_index.core.evaluation import EvaluationResult

from story_rag.config import Settings
from story_rag.evaluation import StoryEvaluationPipeline, build_report, detect_language
from tests.conftest import CHINESE_STORY, ENGLISH_STORY, ScriptedLLM


def test_detect_language_handles_bilingual_text() -> None:
    assert detect_language("白龙回到了山谷") == "zh"
    assert detect_language("the dragon returned") == "en"
    assert detect_language("白龙 called the dragon") == "mixed"
    assert detect_language("123") == "unknown"


def test_build_report_aggregates_scores_by_language() -> None:
    results = {
        "correctness": [
            EvaluationResult(query="q1", response="a1", score=4.0, passing=True),
            EvaluationResult(query="q2", response="a2", score=2.0, passing=False),
        ],
        "faithfulness": [
            EvaluationResult(query="q1", response="a1", score=1.0, passing=True),
            EvaluationResult(query="q2", response="a2", score=0.0, passing=False),
        ],
    }

    report = build_report(results, ["zh", "en"])

    assert report["evaluators"]["correctness"]["mean_score"] == 3.0
    assert report["evaluators"]["correctness"]["passing_count"] == 1
    assert report["by_language"]["faithfulness"]["zh"]["mean_score"] == 1.0
    assert report["by_language"]["faithfulness"]["en"]["mean_score"] == 0.0
    assert report["items"][1]["language"] == "en"
    assert report["items"][0]["correctness"]["passing"] is True


@pytest.mark.evaluation
def test_generate_dataset_from_txt(container, settings: Settings, tmp_path) -> None:
    story_dir = tmp_path / "stories"
    story_dir.mkdir()
    (story_dir / "zh.txt").write_text(CHINESE_STORY, encoding="utf-8")
    (story_dir / "en.txt").write_text(ENGLISH_STORY, encoding="utf-8")

    llm = ScriptedLLM(responses=["白龙把雨水带给了谁？\n山谷里住着谁？"] * 2)
    pipeline = StoryEvaluationPipeline(settings, llm, container, report_dir=tmp_path / "reports")
    output_path = tmp_path / "reports" / "dataset.json"

    written = pipeline.generate_dataset([story_dir], output_path, max_questions=4)

    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["generator"] == "llama_index.DatasetGenerator"
    assert payload["items"]
    assert all(item["query"] for item in payload["items"])
    assert {item["language"] for item in payload["items"]} <= {"zh", "en", "mixed", "unknown"}

    preview = pipeline.preview_case(output_path, index=0)
    assert preview["id"] == "0000"
