from __future__ import annotations

import json

import pytest
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import NodeWithScore, TextNode

from scripts.benchmark_resources import MIB, summarize_resources
from scripts.run_nowcoder_benchmark import (
    StageCapture,
    attach_captures,
    prepare_cases,
    select_files,
    stage_metrics,
    summarize_group,
    write_comparison,
)
from story_rag.domain import NodeTrace, QueryRequest


def test_select_files_excludes_categories_recursively(tmp_path):
    for filename in ("01_a.txt", "other/02_b.txt", "产品/03_c.txt", "机械/sub/04_d.txt"):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sample", encoding="utf-8")
    included, excluded = select_files(tmp_path)
    assert len(included) == 2
    assert len(excluded) == 2
    assert all("产品" not in path.parts and "机械" not in path.parts for path in included)


def test_native_capture_snapshots_scores_and_metadata():
    capture = StageCapture("fused", CBEventType.RETRIEVE)
    node = NodeWithScore(node=TextNode(id_="n1", text="answer", metadata={"a": 1}), score=0.2)
    capture.on_event_start(CBEventType.RETRIEVE, event_id="one")
    capture.on_event_end(CBEventType.RETRIEVE, {EventPayload.NODES: [node]}, "one")
    node.score = 9
    node.node.metadata["a"] = 5
    assert capture.nodes[0].score == 0.2
    assert capture.nodes[0].metadata["a"] == 1
    assert capture.count == 1
    capture.reset()
    assert not capture.nodes
    assert capture.count == 0


def test_native_metrics_accept_any_identical_reference_node():
    case = {
        "expected_node_ids": ["n1", "n2"],
        "expected_answer": "reference",
        "reference_in_corpus": True,
    }
    nodes = [
        NodeTrace(node_id="n3", text="other", retriever="vector", rank=1),
        NodeTrace(node_id="n2", text="reference", retriever="vector", rank=2),
    ]
    result = stage_metrics(case, nodes)
    assert result["hit_at_1"] == 0
    assert result["hit_at_3"] == 1
    assert result["mrr"] == 0.5


def test_missing_reference_is_not_a_retrieval_miss_or_success():
    case = {"expected_node_ids": [], "expected_answer": "unavailable", "reference_in_corpus": False}
    result = stage_metrics(case, [])
    assert result["hit_at_1"] is None
    assert result["mrr"] is None


def test_answer_present_but_not_in_any_node_is_a_miss():
    case = {
        "expected_node_ids": [],
        "expected_answer": "fragmented answer",
        "reference_in_corpus": True,
    }
    assert stage_metrics(case, [])["hit_at_5"] == 0


def test_empty_group_has_no_fabricated_hit_rate():
    summary = summarize_group([])
    assert summary["cases"] == 0
    assert summary["stages"]["reranked"]["hit_at_1"]["answerable_rate"] is None


def test_memory_aggregate_peak_uses_simultaneous_samples():
    samples = []
    for index, (python, qdrant) in enumerate(((10, 2), (3, 9))):
        values = {
            name: {"rss_bytes": amount * MIB, "private_bytes": amount * MIB}
            for name, amount in (
                ("python", python),
                ("qdrant", qdrant),
                ("lm_studio", 0),
                ("aggregate", python + qdrant),
            )
        }
        samples.append(
            {
                "elapsed_seconds": index * 4.3,
                "processes": values,
                "gpu": None,
                "system_used_bytes": 20 * MIB,
            }
        )
    summary = summarize_resources(samples)
    assert summary["processes"]["python"]["rss_bytes_peak_mib"] == 10
    assert summary["processes"]["qdrant"]["rss_bytes_peak_mib"] == 9
    assert summary["processes"]["aggregate"]["rss_bytes_peak_mib"] == 12
    assert summary["gpu"]["device_used_peak_mib"] is None
    assert summary["gpu"]["missing_sample_count"] == 2
    assert summary["observed_interval_seconds"]["p50"] == 4.3


def test_empty_resource_samples_have_no_fabricated_interval_or_peak():
    summary = summarize_resources([])
    assert summary["observed_interval_seconds"]["count"] == 0
    assert summary["observed_interval_seconds"]["p95"] is None
    assert summary["gpu"]["missing_sample_count"] == 0
    assert summary["system_used_peak_mib"] is None


def test_comparison_reports_actual_sampling_interval_and_missing_gpu(tmp_path):
    summary = {
        "groups": {},
        "run_wall_seconds": 1,
        "phases_ms": {},
        "resources": {
            "sample_count": 3,
            "processes": {},
            "gpu": {"missing_sample_count": 1},
            "observed_interval_seconds": {"count": 2, "p50": 4.3, "p95": 5, "max": 5},
        },
    }
    write_comparison(tmp_path, {}, summary)
    report = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "P50 4.300s / P95 5.000s" in report
    assert "显存缺测 1 次" in report


def test_native_capture_does_not_repeat_production_retrieval(container):
    from story_rag.adapters.model_factory import ModelBundle

    container.ingestion.ingest_bytes("问题：test?\n答案：reference".encode(), source_name="qa.txt")
    models = ModelBundle(
        llm=container.llm,
        embedding=container.storage.index._embed_model,
        reranker=container.retrieval._reranker,
    )
    captures = attach_captures(container, models)
    result = container.retrieval.retrieve(QueryRequest(query="test?"))
    assert result.nodes
    assert all(captures[name].count == 1 for name in ("vector", "bm25", "fused"))
    assert all(captures[name].nodes for name in ("vector", "bm25", "fused"))


def test_paraphrase_fixture_count_and_frozen_no_llm_boundary():
    from scripts.run_qa_benchmark import PROJECT_ROOT, GenerationDisabledLLM

    rows = json.loads(
        (PROJECT_ROOT / "scripts/nowcoder_paraphrases.json").read_text(encoding="utf-8")
    )["cases"]
    assert len(rows) == 128
    assert len({(row[0], row[1]) for row in rows}) == 128
    assert len({row[2] for row in rows}) == 128
    with pytest.raises(RuntimeError, match="disabled"):
        GenerationDisabledLLM().complete("never send this")


def test_prepare_rejects_verbatim_paraphrase(tmp_path, monkeypatch):
    corpus = tmp_path / "75_test.txt"
    corpus.write_text("问题：same question\n答案：reference", encoding="utf-8")
    annotation = tmp_path / "annotations.json"
    annotation.write_text(json.dumps({"cases": [[75, 0, "same question"]] * 128}), encoding="utf-8")
    monkeypatch.setattr("scripts.run_nowcoder_benchmark.load_cases", lambda _: [])
    with pytest.raises(ValueError, match="verbatim"):
        prepare_cases([corpus], annotation)


def test_prepare_labels_are_separate_from_user_question(tmp_path, monkeypatch):
    corpus = tmp_path / "75_test.txt"
    corpus.write_text("问题：original question\n答案：private reference", encoding="utf-8")
    annotation = tmp_path / "annotations.json"
    annotation.write_text(
        json.dumps({"cases": [[75, 0, f"changed wording {number}?"] for number in range(128)]}),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.run_nowcoder_benchmark.load_cases", lambda _: [])
    cases, manifest = prepare_cases([corpus], annotation)
    assert len(cases) == 128
    assert all("private reference" not in case["question"] for case in cases)
    assert cases[0]["expected_answer"] == "private reference"
    assert manifest[0]["qa_pairs"] == 1
