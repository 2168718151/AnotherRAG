"""Benchmark real retrieval on non-product/non-mechanical TXT, without LLM generation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil
import torch
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.evaluation.retrieval.metrics import MRR, HitRate

from scripts.benchmark_resources import MIB, ResourceSampler
from scripts.run_qa_benchmark import (
    DEFAULT_CASES,
    PROJECT_ROOT,
    GenerationDisabledLLM,
    build_settings,
    export_llm_inputs,
    load_cases,
    normalized,
    percentile,
    reference_rank,
)
from story_rag.adapters.model_factory import ModelBundle, build_embedding_model, build_reranker
from story_rag.adapters.qa_parser import extract_qa_pairs
from story_rag.adapters.text_loader import read_text_file
from story_rag.container import ApplicationContainer, build_container
from story_rag.domain import NodeTrace, QueryRequest
from story_rag.logging_config import configure_logging

EXCLUDED_NAMES = {"\u4ea7\u54c1", "\u673a\u68b0"}
STAGES = ("vector", "bm25", "fused", "reranked")
HISTORY = {
    "cpu_chroma": PROJECT_ROOT / "reports" / "qa_v14d1_20261009_01",
    "cuda_qdrant": PROJECT_ROOT / "reports" / "qa_v14d1_qdrant_20261011_01",
}


def select_files(root: Path) -> tuple[list[Path], list[Path]]:
    root = root.resolve(strict=True)
    included, excluded = [], []
    for path in sorted(root.rglob("*.txt")):
        relative = path.relative_to(root)
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"TXT resolves outside the corpus: {relative}")
        excluded_path = any(part in EXCLUDED_NAMES for part in relative.parts[:-1])
        excluded_file = any(path.stem.startswith(name) for name in EXCLUDED_NAMES)
        (excluded if excluded_path or excluded_file else included).append(path)
    if not included:
        raise ValueError("No eligible TXT files")
    return included, excluded


def prepare_cases(files: list[Path], annotations: Path) -> tuple[list[dict], list[dict]]:
    documents = {path: read_text_file(path) for path in files}
    corpus = [normalized(document.text) for document in documents.values()]
    numbers: dict[int, Path] = {}
    for path in files:
        match = re.match(r"^(\d+)_", path.name)
        if match:
            number = int(match[1])
            if number in numbers:
                raise ValueError(f"Ambiguous source number {number}")
            numbers[number] = path
    cases = []
    for original in load_cases(DEFAULT_CASES):
        cases.append(
            {
                **original,
                "group": "frozen",
                "reference_in_corpus": any(
                    normalized(original["expected_answer"]) in text for text in corpus
                ),
                "language": "zh" if re.search(r"[\u3400-\u9fff]", original["question"]) else "en",
            }
        )
    rows = json.loads(annotations.read_text(encoding="utf-8"))["cases"]
    if len(rows) != 128:
        raise ValueError("Exactly 128 paraphrase annotations are required")
    queries: set[str] = set()
    for index, (number, pair_index, question) in enumerate(rows, start=1):
        path = numbers[number]
        document = documents[path]
        pair = extract_qa_pairs(document.text)[pair_index]
        target = normalized(question)
        if not target or target in queries or any(target in text for text in corpus):
            raise ValueError(f"Paraphrase {index} is empty, duplicated, or verbatim in corpus")
        queries.add(target)
        answer = re.split(
            r"(?m)^\s*(?:A|\u7b54|\u7b54\u6848)\s*[:\uff1a]\s*",
            pair.text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[1].strip()
        if not answer:
            raise ValueError(f"Paraphrase {index} has no reference answer")
        cases.append(
            {
                "case_id": f"paraphrase-{index:03d}",
                "group": "paraphrase",
                "question": question,
                "source_title": path.stem,
                "source_file": str(path.resolve()),
                "source_number": number,
                "source_question": pair.question,
                "source_pair_index": pair_index,
                "source_sha256": document.content_hash,
                "expected_answer": answer,
                "reference_in_corpus": True,
                "language": "zh" if re.search(r"[\u3400-\u9fff]", question) else "en",
            }
        )
    manifest = [
        {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "qa_pairs": len(extract_qa_pairs(documents[path].text)),
        }
        for path in files
    ]
    return cases, manifest


class StageCapture(BaseCallbackHandler):
    """Snapshot native callback results before fusion/reranking mutates node scores."""

    def __init__(self, stage: str, event_type: CBEventType) -> None:
        super().__init__([], [])
        self.stage, self.event_type = stage, event_type
        self.reset()

    def reset(self) -> None:
        self.starts: dict[str, float] = {}
        self.elapsed_ms = 0.0
        self.count = 0
        self.nodes: list[NodeTrace] = []

    def on_event_start(self, event_type, payload=None, event_id="", parent_id="", **kwargs):
        if event_type == self.event_type:
            self.starts[event_id] = time.perf_counter()
        return event_id

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        if event_type != self.event_type or event_id not in self.starts:
            return
        self.elapsed_ms += (time.perf_counter() - self.starts.pop(event_id)) * 1000
        self.count += 1
        if payload and EventPayload.NODES in payload:
            self.nodes = [
                NodeTrace(
                    node_id=item.node.node_id,
                    text=item.node.get_content(),
                    retriever=self.stage,
                    rank=index,
                    score=item.score,
                    metadata=dict(item.node.metadata),
                )
                for index, item in enumerate(payload[EventPayload.NODES], start=1)
            ]

    def start_trace(self, trace_id=None):
        pass

    def end_trace(self, trace_id=None, trace_map=None):
        pass


def attach_captures(
    container: ApplicationContainer, models: ModelBundle
) -> dict[str, StageCapture]:
    retrievers = container.retrieval._ensure_retrievers()
    captures = {}
    for stage, component, event_type in (
        *(
            (name, retriever, CBEventType.RETRIEVE)
            for name, retriever in zip(STAGES[:3], retrievers, strict=True)
        ),
        ("rerank", models.reranker, CBEventType.RERANKING),
        ("embedding", models.embedding, CBEventType.EMBEDDING),
    ):
        capture = StageCapture(stage, event_type)
        component.callback_manager = CallbackManager([capture])
        captures[stage] = capture
    return captures


def stage_metrics(case: dict, nodes: list[NodeTrace]) -> dict[str, Any]:
    ids = [node.node_id for node in nodes]
    expected = case["expected_node_ids"]
    answerable = case["reference_in_corpus"]
    metrics: dict[str, Any] = {"reference_rank": reference_rank(case["expected_answer"], nodes)}
    for k in (1, 3, 5):
        metrics[f"hit_at_{k}"] = (
            (
                HitRate().compute(expected_ids=expected, retrieved_ids=ids[:k]).score
                if expected and ids
                else 0.0
            )
            if answerable
            else None
        )
    metrics["mrr"] = (
        (MRR().compute(expected_ids=expected, retrieved_ids=ids).score if expected and ids else 0.0)
        if answerable
        else None
    )
    return metrics


def latency_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "sum": 0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "sum": sum(values),
    }


def summarize_group(rows: list[dict]) -> dict[str, Any]:
    answerable = [row for row in rows if row["case"]["reference_in_corpus"]]
    result: dict[str, Any] = {
        "cases": len(rows),
        "reference_in_corpus": len(answerable),
        "reference_missing": len(rows) - len(answerable),
        "full_reference_in_nodes": sum(bool(row["case"]["expected_node_ids"]) for row in rows),
        "pipeline_latency_ms": latency_summary([row["timing_ms"]["pipeline"] for row in rows]),
        "retrieval_latency_ms": latency_summary([row["retrieval"]["latency_ms"] for row in rows]),
        "stages": {},
        "timings_ms": {},
    }
    for stage in STAGES:
        result["stages"][stage] = {
            f"hit_at_{k}": {
                "hits": sum(row["metrics"][stage][f"hit_at_{k}"] == 1 for row in answerable),
                "answerable_rate": statistics.mean(
                    row["metrics"][stage][f"hit_at_{k}"] for row in answerable
                )
                if answerable
                else None,
                "all_cases_rate": sum(row["metrics"][stage][f"hit_at_{k}"] == 1 for row in rows)
                / len(rows)
                if rows
                else None,
            }
            for k in (1, 3, 5)
        }
        result["stages"][stage]["mrr_answerable"] = (
            statistics.mean(row["metrics"][stage]["mrr"] for row in answerable)
            if answerable
            else None
        )
    for key in ("embedding", "vector", "bm25", "fused", "rerank", "prompt", "pipeline"):
        result["timings_ms"][key] = latency_summary([row["timing_ms"][key] for row in rows])
    return result


@contextmanager
def isolated_qdrant(storage: Path):
    executable = PROJECT_ROOT / "tools" / "qdrant" / "qdrant.exe"
    if not executable.is_file():
        raise FileNotFoundError("Install the project Qdrant binary first")
    storage.mkdir(parents=True, exist_ok=False)
    for _ in range(50):
        with socket.socket() as first, socket.socket() as second:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            if port >= 65534:
                continue
            try:
                second.bind(("127.0.0.1", port + 1))
                break
            except OSError:
                continue
    else:
        raise RuntimeError("No free Qdrant port pair")
    env = {
        **os.environ,
        "QDRANT__SERVICE__HOST": "127.0.0.1",
        "QDRANT__SERVICE__HTTP_PORT": str(port),
        "QDRANT__SERVICE__GRPC_PORT": str(port + 1),
        "QDRANT__STORAGE__STORAGE_PATH": str(storage.resolve()),
        "QDRANT__TELEMETRY_DISABLED": "true",
    }
    with (storage / "server.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(executable)],
            cwd=storage,
            env=env,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError(f"Qdrant exited; inspect {storage / 'server.log'}")
                try:
                    if httpx.get(f"{url}/readyz", timeout=0.5).is_success:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError("Qdrant startup timed out")
            yield url, process.pid
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_history(root: Path) -> dict:
    copied = {}
    for name, source in HISTORY.items():
        target = root / name
        target.mkdir()
        for filename in ("summary.json", "results.jsonl", "llm_inputs.jsonl"):
            if (source / filename).is_file():
                shutil.copy2(source / filename, target / filename)
        copied[name] = {
            "original": str(source),
            "copy": str(target),
            "summary": json.loads((target / "summary.json").read_text(encoding="utf-8")),
            "memory_measured": False,
        }
    return copied


def write_comparison(root: Path, history: dict, summary: dict) -> None:
    lines = [
        "# Q&A 检索测试汇总",
        "",
        "旧 CPU+Chroma、CUDA+Qdrant 的原始 JSON 报告已复制到同级目录，原目录保持不变。",
        "本次采用真实本地模型，生产 retrieve() 单次调用及原生回调，不调用付费 LLM。",
        "",
        "| 测试组 | TXT / 节点 | 参考答案覆盖 | Rerank Hit@1 | Hit@5 | "
        "检索 P50 / P95 | 到 Prompt P50 / P95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in history.items():
        old = entry["summary"]
        latency = old["retrieval_latency_ms"]
        stage = old["stages"]["reranked"]
        count = old["cases"]
        lines.append(
            f"| {name}，旧语料原题 | {old['documents']} / {old['nodes']} | {count}/{count} "
            f"| {round(stage['reference_hit_at_1'] * count)}/{count} "
            f"| {round(stage['reference_hit_at_5'] * count)}/{count} "
            f"| {latency['p50'] / 1000:.3f}s / {latency['p95'] / 1000:.3f}s | 未测 |"
        )
    for name, group in summary["groups"].items():
        count = group["reference_in_corpus"]
        stage = group["stages"]["reranked"]
        latency, pipeline = group["retrieval_latency_ms"], group["pipeline_latency_ms"]
        lines.append(
            f"| CUDA+Qdrant，{name}，新语料 | {summary['documents']} / {summary['nodes']} "
            f"| {count}/{group['cases']} | {stage['hit_at_1']['hits']}/{count} "
            f"| {stage['hit_at_5']['hits']}/{count} "
            f"| {latency['p50'] / 1000:.3f}s / {latency['p95'] / 1000:.3f}s "
            f"| {pipeline['p50'] / 1000:.3f}s / {pipeline['p95'] / 1000:.3f}s |"
        )
    lines += [
        "",
        "参考答案覆盖为规范化后逐字完整匹配，不代表其他题必定无法作答。",
        "命中以包含完整参考答案的节点为依据，语义等价但措辞不同的证据可能未计入，指标偏保守。",
        "旧测试有额外诊断检索，新测试仅一次生产检索；语料和计时口径不同，不能直接算加速比。",
        "到 Prompt 不含 LLM 生成、Yes/No 门控、API HTTP 开销和并发压测。",
        "原题参考答案缺失的案例保留结果但不计入可判分分母，也不当作无答案检测成功。",
        "",
        "## 运行时间与资源",
        "",
        f"本次整体运行到汇总输出：{summary['run_wall_seconds']:.3f} 秒。",
        "LM Studio 服务和 Embedding 装载属于运行前准备；Qdrant 与应用模型初始化计入本次计时。",
        "",
        "| 阶段 | 秒 |",
        "|---|---:|",
        *[f"| {name} | {value / 1000:.3f} |" for name, value in summary["phases_ms"].items()],
        "",
        "| 进程组 | 工作集采样峰值 MiB | Private Bytes 采样峰值 MiB |",
        "|---|---:|---:|",
        *[
            f"| {name} | {value['rss_bytes_peak_mib']:.2f} "
            f"| {value['private_bytes_peak_mib']:.2f} |"
            for name, value in summary["resources"]["processes"].items()
        ],
        "",
        "Private Bytes 是提交内存，不是实际物理内存；合计工作集可能重复计算共享页。",
        "旧测试未测内存，不能补造历史峰值或宣称内存下降。",
        "整卡 VRAM 还包含其他进程；PyTorch 分配/保留峰值仅属于本次 Python 进程。",
        "",
        "```json",
        json.dumps(summary["resources"]["gpu"], ensure_ascii=False, indent=2),
        "```",
        "",
        "逐题结果、256 个最终输入、语料哈希、缺失标签和未命中案例见 nowcoder/。",
        "语料 SHA256 在运行后复核；原始 TXT 未改写。",
    ]
    observed = summary["resources"].get("observed_interval_seconds")
    if observed and observed["count"]:
        lines += [
            "",
            f"资源采样共 {summary['resources']['sample_count']} 条，实际间隔 "
            f"P50 {observed['p50']:.3f}s / P95 {observed['p95']:.3f}s，"
            f"最长 {observed['max']:.3f}s；等待设置不包含测量耗时。",
            f"显存缺测 {summary['resources']['gpu']['missing_sample_count']} 次；"
            "采样峰值可能漏掉两次采样之间的短暂尖峰。",
        ]
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    files, excluded = select_files(args.corpus_dir)
    cases, manifest = prepare_cases(files, args.annotations)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    history = copy_history(root)
    output = root / "nowcoder"
    output.mkdir()
    write_json(
        output / "corpus_manifest.json",
        {
            "selected": manifest,
            "excluded": [str(path.resolve()) for path in excluded],
            "annotation_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
            "frozen_cases_sha256": hashlib.sha256(DEFAULT_CASES.read_bytes()).hexdigest(),
        },
    )
    write_json(output / "cases.json", cases)
    if args.prepare_only:
        return {
            "selected_files": len(files),
            "excluded_files": len(excluded),
            "qa_pairs": sum(row["qa_pairs"] for row in manifest),
            "frozen_reference_coverage": sum(case["reference_in_corpus"] for case in cases[:128]),
            "paraphrase_cases": len(cases[128:]),
            "output_dir": str(root),
        }
    if not torch.cuda.is_available():
        raise RuntimeError("The requested real CUDA benchmark needs an available CUDA device")
    sampler = ResourceSampler(output / "resources.jsonl")
    sampler.start()
    phases = {"preparation_ms": (time.perf_counter() - started) * 1000}
    storage_name = "n" + hashlib.sha256(str(root).encode()).hexdigest()[:6]
    short_storage = PROJECT_ROOT / "reports" / storage_name
    container = None
    try:
        phase = time.perf_counter()
        with isolated_qdrant(short_storage) as (qdrant_url, qdrant_pid):
            sampler.qdrant_pid = qdrant_pid
            phases["qdrant_startup_ms"] = (time.perf_counter() - phase) * 1000
            settings = replace(
                build_settings(output),
                qdrant_url=qdrant_url,
                collection_name=f"n_{short_storage.name}",
            )
            configure_logging(settings)
            phase = time.perf_counter()
            models = ModelBundle(
                llm=GenerationDisabledLLM(),
                embedding=build_embedding_model(settings),
                reranker=build_reranker(settings, device="cuda:0"),
            )
            container = build_container(settings, models=models)
            phases["models_container_startup_ms"] = (time.perf_counter() - phase) * 1000
            environment = {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "reranker_device": str(models.reranker._model.device),
                "reranker_dtype": str(models.reranker._model.model.dtype),
                "embedding_model": settings.embedding_model,
                "reranker_model": str(settings.reranker_model_path),
                "vector_backend": "qdrant",
                "qdrant_url": qdrant_url,
                "qdrant_pid": qdrant_pid,
                "qdrant_storage": str(short_storage),
                "collection": settings.collection_name,
                "top_k": {
                    "vector": settings.vector_top_k,
                    "bm25": settings.bm25_top_k,
                    "fusion": settings.fusion_top_k,
                    "rerank": settings.rerank_top_n,
                },
            }
            try:
                summary, rows = run_pipeline(
                    container, models, cases, files, output, sampler, phases
                )
                sampler.phase = "export"
                resource_summary = sampler.stop()
                resource_summary["gpu"]["python_allocated_peak_mib"] = (
                    torch.cuda.max_memory_allocated(0) / MIB
                )
                resource_summary["gpu"]["python_reserved_peak_mib"] = (
                    torch.cuda.max_memory_reserved(0) / MIB
                )
                process_memory = psutil.Process().memory_info()
                resource_summary["python_os_peak_working_set_mib"] = (
                    getattr(process_memory, "peak_wset", 0) / MIB or None
                )
                summary["resources"] = resource_summary
                summary["environment"] = environment
                summary["phases_ms"] = phases
                summary["run_wall_seconds"] = time.perf_counter() - started
                summary["python_process_seconds_to_summary"] = (
                    time.time() - psutil.Process().create_time()
                )
                summary["finished_at"] = datetime.now(UTC).isoformat()
                summary["source_hashes_unchanged"] = all(
                    hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]
                    for row in manifest
                )
                summary["cases_sha256"] = hashlib.sha256(
                    (output / "cases.json").read_bytes()
                ).hexdigest()
                summary["corpus_dir"] = str(args.corpus_dir.resolve())
                summary["excluded_files"] = len(excluded)
                summary["qa_pairs"] = sum(row["qa_pairs"] for row in manifest)
                write_json(output / "summary.json", summary)
                write_json(root / "comparison.json", {"history": history, "nowcoder": summary})
                write_comparison(root, history, summary)
                write_json(
                    output / "failures.json",
                    [
                        {
                            "case": row["case"],
                            "metrics": row["metrics"]["reranked"],
                            "retrieved": row["retrieval"]["nodes"],
                        }
                        for row in rows
                        if row["case"]["reference_in_corpus"]
                        and row["metrics"]["reranked"]["hit_at_1"] != 1
                    ],
                )
                return summary
            finally:
                asyncio.run(container.storage.aclose())
    finally:
        if not sampler._stop.is_set():
            sampler.stop()


def run_pipeline(container, models, cases, files, output, sampler, phases) -> tuple[dict, list]:
    started_at = datetime.now(UTC).isoformat()
    sampler.phase = "ingestion"
    phase = time.perf_counter()
    with (output / "ingestion.jsonl").open("x", encoding="utf-8") as handle:
        for index, path in enumerate(files, start=1):
            tick = time.perf_counter()
            result = container.ingestion.ingest_path(path)
            handle.write(
                json.dumps(
                    {
                        "path": str(path),
                        "latency_ms": (time.perf_counter() - tick) * 1000,
                        "result": result.model_dump(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            if index % 25 == 0 or index == len(files):
                print(f"Indexed {index}/{len(files)} files", flush=True)
    phases["ingestion_ms"] = (time.perf_counter() - phase) * 1000
    sampler.phase = "retriever_build"
    phase = time.perf_counter()
    captures = attach_captures(container, models)
    phases["retriever_build_ms"] = (time.perf_counter() - phase) * 1000
    node_texts = [
        (node.node_id, normalized(node.get_content())) for node in container.storage.text_nodes()
    ]
    for case in cases:
        reference = normalized(case["expected_answer"])
        case["expected_node_ids"] = [node_id for node_id, text in node_texts if reference in text]
    write_json(output / "cases.json", cases)
    covered = sum(case["reference_in_corpus"] for case in cases[:128])
    print(f"Frozen reference coverage: {covered}/128", flush=True)
    rows = []
    result_path = output / "results.jsonl"
    phase = time.perf_counter()
    with result_path.open("x", encoding="utf-8") as handle:
        for index, case in enumerate(cases, start=1):
            sampler.phase = f"query:{case['group']}"
            for capture in captures.values():
                capture.reset()
            request = QueryRequest(query=case["question"], top_k=5)
            tick = time.perf_counter()
            retrieval = container.retrieval.retrieve(request)
            prompt_tick = time.perf_counter()
            llm_input = container.generation.prepare_llm_input(request, retrieval)
            prompt_ms = (time.perf_counter() - prompt_tick) * 1000
            pipeline_ms = (time.perf_counter() - tick) * 1000
            if any(captures[stage].count != 1 for stage in (*STAGES[:3], "rerank")):
                raise RuntimeError("Native callback capture did not observe one pass per stage")
            candidates = {stage: captures[stage].nodes for stage in STAGES[:3]}
            candidates["reranked"] = retrieval.nodes
            retrieval = retrieval.model_copy(
                update={
                    "vector_candidates": candidates["vector"],
                    "bm25_candidates": candidates["bm25"],
                    "fused_candidates": candidates["fused"],
                }
            )
            row = {
                "case": case,
                "metrics": {
                    stage: stage_metrics(case, nodes) for stage, nodes in candidates.items()
                },
                "timing_ms": {
                    **{name: capture.elapsed_ms for name, capture in captures.items()},
                    "prompt": prompt_ms,
                    "pipeline": pipeline_ms,
                },
                "native_event_counts": {name: capture.count for name, capture in captures.items()},
                "retrieval": retrieval.model_dump(),
                "llm_input": llm_input,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 8 == 0 or index == len(cases):
                print(
                    f"Processed {index}/{len(cases)}; last pipeline {pipeline_ms / 1000:.2f}s",
                    flush=True,
                )
    phases["query_loop_with_reporting_ms"] = (time.perf_counter() - phase) * 1000
    phase = time.perf_counter()
    if export_llm_inputs(result_path, output / "llm_inputs.jsonl") != len(cases):
        raise RuntimeError("Incomplete final LLM input export")
    phases["input_export_ms"] = (time.perf_counter() - phase) * 1000
    return {
        "started_at": started_at,
        "cases": len(cases),
        "llm_generation_calls": 0,
        "documents": container.storage.document_count(),
        "nodes": container.storage.node_count(),
        "embedding_dimension": container.storage.embedding_dimension,
        "prompt_count": sum(bool(row["llm_input"]["would_call_llm"]) for row in rows),
        "groups": {
            name: summarize_group([row for row in rows if row["case"]["group"] == name])
            for name in ("frozen", "paraphrase")
        },
        "paraphrase_languages": {
            name: summarize_group(
                [
                    row
                    for row in rows
                    if row["case"]["group"] == "paraphrase" and row["case"]["language"] == name
                ]
            )
            for name in ("zh", "en")
        },
        "timing_scope": "Production retrieve() plus final prompt assembly; native callbacks only.",
        "timing_notes": [
            "Vector includes embedding; fusion includes parallel vector/BM25. Do not sum stages.",
            "No paid generation, relevance gate, API HTTP transport, or concurrent-load test.",
            "Historical runs used extra diagnostic retrieval passes and did not measure memory.",
            "LM Studio model/service startup is external preparation, not part of this run timer.",
        ],
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path(r"D:\data\nowcoder_raw"))
    parser.add_argument(
        "--annotations", type=Path, default=PROJECT_ROOT / "scripts" / "nowcoder_paraphrases.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    summary = run(parser.parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
