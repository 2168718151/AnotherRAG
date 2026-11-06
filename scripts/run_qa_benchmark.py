"""Run the V1.4 frozen QA cases through ingestion, retrieval, and LLM input assembly."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import statistics
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from llama_index.core.llms.custom import CustomLLM

from story_rag.adapters.model_factory import ModelBundle, build_embedding_model, build_reranker
from story_rag.config import Settings
from story_rag.container import ApplicationContainer, build_container
from story_rag.domain import NodeTrace, QueryRequest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = Path(r"D:\text_Project\找面经八股\raw")
DEFAULT_CASES = Path(
    r"D:\text_Project\Agentic RAG\data\benchmark_v14_20260918\cases.jsonl"
)


class GenerationDisabledLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="generation-disabled", is_chat_model=True)

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        raise RuntimeError("LLM generation is disabled for this benchmark")

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        raise RuntimeError("LLM generation is disabled for this benchmark")


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def reference_rank(reference: str, nodes: list[NodeTrace]) -> int | None:
    target = normalized(reference)
    if not target:
        return None
    return next(
        (index for index, node in enumerate(nodes, start=1) if target in normalized(node.text)),
        None,
    )


def source_rank(source_title: str, nodes: list[NodeTrace]) -> int | None:
    target = source_title.casefold()
    return next(
        (
            index
            for index, node in enumerate(nodes, start=1)
            if str(node.metadata.get("title", "")).casefold() == target
        ),
        None,
    )


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(cases) != 128:
        raise ValueError(f"Expected 128 V1.4 cases, found {len(cases)}")
    required = {"case_id", "question", "source_title", "expected_answer"}
    for case in cases:
        missing = required - case.keys()
        if missing:
            raise ValueError(f"Case {case.get('case_id', '?')} is missing {sorted(missing)}")
    return cases


def build_settings(output_dir: Path) -> Settings:
    data_dir = output_dir / "index"
    return Settings(
        project_root=PROJECT_ROOT,
        api_file=PROJECT_ROOT / "api.txt",
        deepseek_base_url="https://api.deepseek.com",
        llm_model="deepseek-flash",
        llm_api_key="generation-disabled",
        data_dir=data_dir,
        qdrant_url="http://127.0.0.1:6333",
        docstore_path=data_dir / "docstore" / "docstore.json",
        log_dir=output_dir / "logs",
        collection_name=f"qa_benchmark_{hashlib.sha256(str(output_dir.resolve()).encode()).hexdigest()[:20]}",
        query_fusion_num_queries=1,
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def export_llm_inputs(results_path: Path, output_path: Path) -> int:
    count = 0
    with results_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            row = json.loads(line)
            llm_input = row["llm_input"]
            output.write(
                json.dumps(
                    {
                        "case_id": row["case"]["case_id"],
                        "model": llm_input["model"],
                        "would_call_llm": llm_input["would_call_llm"],
                        "messages": llm_input["messages"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def summarize(rows: list[dict[str, Any]], documents: int, nodes: int) -> dict[str, Any]:
    latencies = [row["retrieval"]["latency_ms"] for row in rows]
    metrics: dict[str, Any] = {
        "cases": len(rows),
        "documents": documents,
        "nodes": nodes,
        "llm_generation_calls": 0,
        "prompt_count": sum(bool(row["llm_input"]["would_call_llm"]) for row in rows),
        "retrieval_latency_ms": {
            "p50": statistics.median(latencies),
            "p95": percentile(latencies, 0.95),
        },
        "stages": {},
    }
    for stage in ("vector", "bm25", "fused", "reranked"):
        ranks = [row["metrics"][stage]["reference_rank"] for row in rows]
        source_ranks = [row["metrics"][stage]["source_rank"] for row in rows]
        metrics["stages"][stage] = {
            "reference_hit_at_1": sum(rank == 1 for rank in ranks) / len(rows),
            "reference_hit_at_3": sum(rank is not None and rank <= 3 for rank in ranks)
            / len(rows),
            "reference_hit_at_5": sum(rank is not None and rank <= 5 for rank in ranks)
            / len(rows),
            "source_hit_at_5": sum(rank is not None and rank <= 5 for rank in source_ranks)
            / len(rows),
        }
    return metrics


def run(corpus_dir: Path, cases_path: Path, output_dir: Path) -> dict[str, Any]:
    cases = load_cases(cases_path)
    files = sorted(corpus_dir.glob("*.txt"))
    if len(files) != 133:
        raise ValueError(f"Expected 133 V1.4 TXT files, found {len(files)}")
    missing = sorted(
        {case["source_title"] for case in cases}
        - {path.stem for path in files}
    )
    if missing:
        raise ValueError(f"Missing source TXT files for {len(missing)} cases")
    output_dir.mkdir(parents=True, exist_ok=False)

    settings = build_settings(output_dir)
    container = build_container(
        settings,
        models=ModelBundle(
            llm=GenerationDisabledLLM(),
            embedding=build_embedding_model(settings),
            reranker=build_reranker(settings),
        ),
    )
    try:
        return _run_pipeline(container, cases, files, cases_path, corpus_dir, output_dir)
    finally:
        asyncio.run(container.storage.aclose())


def _run_pipeline(
    container: ApplicationContainer,
    cases: list[dict[str, Any]],
    files: list[Path],
    cases_path: Path,
    corpus_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    settings = container.settings
    started_at = datetime.now(UTC).isoformat()
    print(f"Indexing {len(files)} TXT files...", flush=True)
    for index, path in enumerate(files, start=1):
        container.ingestion.ingest_path(path)
        if index % 25 == 0 or index == len(files):
            print(f"Indexed {index}/{len(files)} files", flush=True)

    rows: list[dict[str, Any]] = []
    result_path = output_dir / "results.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, start=1):
            request = QueryRequest(query=case["question"], top_k=5)
            retrieval = container.retrieval.retrieve_trace(request)
            llm_input = container.generation.prepare_llm_input(request, retrieval)
            candidates = {
                "vector": retrieval.vector_candidates,
                "bm25": retrieval.bm25_candidates,
                "fused": retrieval.fused_candidates,
                "reranked": retrieval.nodes,
            }
            row = {
                "case": case,
                "metrics": {
                    stage: {
                        "reference_rank": reference_rank(case["expected_answer"], stage_nodes),
                        "source_rank": source_rank(case["source_title"], stage_nodes),
                    }
                    for stage, stage_nodes in candidates.items()
                },
                "retrieval": retrieval.model_dump(),
                "llm_input": llm_input,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 16 == 0 or index == len(cases):
                print(f"Processed {index}/{len(cases)} cases", flush=True)

    input_path = output_dir / "llm_inputs.jsonl"
    if export_llm_inputs(result_path, input_path) != len(cases):
        raise RuntimeError("LLM input export is incomplete")
    summary = summarize(rows, container.storage.document_count(), container.storage.node_count())
    summary["llm_input_file"] = str(input_path.resolve())
    summary["dataset"] = {
        "name": "Agentic RAG V1.4 20260918 frozen 128 cases",
        "cases_path": str(cases_path.resolve()),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "corpus_dir": str(corpus_dir.resolve()),
        "source_files": len(files),
        "embedding_model": settings.embedding_model,
        "reranker_model": str(settings.reranker_model_path),
        "vector_backend": "qdrant",
        "qdrant_url": settings.qdrant_url,
        "collection_name": settings.collection_name,
        "vector_distance": "Euclid",
        "embedding_dimension": container.storage.embedding_dimension,
        "query_fusion_num_queries": settings.query_fusion_num_queries,
        "vector_top_k": settings.vector_top_k,
        "bm25_top_k": settings.bm25_top_k,
        "fusion_top_k": settings.fusion_top_k,
        "rerank_top_n": settings.rerank_top_n,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.corpus_dir, args.cases, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
