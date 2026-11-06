"""对已保存候选复验门控，并用官方 Evaluator 评分，不重新生成对话。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx
from llama_index.core.evaluation import (
    BatchEvalRunner,
    ContextRelevancyEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import system_acceptance as runtime

from story_rag.adapters.model_factory import build_conversation_llm
from story_rag.config import Settings
from story_rag.conversation_domain import EvidenceRef
from story_rag.logging_config import configure_logging
from story_rag.pipelines.conversation_generation import (
    ConversationAnswerPipeline,
    ConversationTaskInput,
    GeneratedItem,
)
from story_rag.pipelines.evidence_mapping import EvidencePoolEntry


class AsyncStream(httpx.AsyncByteStream):
    def __init__(self, inner, audit, record):
        self.inner, self.audit, self.record = inner, audit, record

    async def __aiter__(self):
        chunks = []
        try:
            async for chunk in self.inner:
                chunks.append(chunk)
                yield chunk
        finally:
            self.audit.finish(self.record, b"".join(chunks))

    async def aclose(self):
        await self.inner.aclose()


class AsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, audit):
        self.audit, self.inner = audit, httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request):
        record = self.audit.begin(request)
        try:
            response = await self.inner.handle_async_request(request)
        except Exception as exc:
            record["transport_error"] = type(exc).__name__
            self.audit.finish(record, b"")
            raise
        record["status_code"] = response.status_code
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            stream=AsyncStream(response.stream, self.audit, record),
            extensions=response.extensions,
        )

    async def aclose(self):
        await self.inner.aclose()


def frozen_legacy(out):
    inputs = []
    for name in ["MT15", "MT31", "MT36"]:
        candidate_path = (
            runtime.ROOT
            / "reports/pipeline_baseline_20261018_210132_d1e17c/candidates"
            / (name + "_U1.json")
        )
        trace_path = (
            runtime.ROOT
            / "reports/pipeline_baseline_20261018_204424_ebbbda/turns"
            / (name + "_U1.json")
        )
        if not candidate_path.exists() or not trace_path.exists():
            inputs.append({"id": name, "status": "阻塞", "reason": "原候选或原证据缺失"})
            continue
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        inputs.append(
            {
                "id": name,
                "query": candidate["query"],
                "answer": candidate["answer"],
                "nodes": trace["preview"]["nodes"],
                "gold_accept": True,
                "provenance": {
                    "candidate_path": str(candidate_path),
                    "candidate_sha": runtime.sha(candidate_path),
                    "trace_path": str(trace_path),
                    "trace_sha": runtime.sha(trace_path),
                },
                "gold_note": (
                    "核对原来源：AOF缺点/刷盘权衡，或StreamDemo原版本7天，候选有来源支持。"
                    "只复验U1门控，不重跑旧四轮或声称整个旧Bug关闭。"
                ),
            }
        )
    path = out / "legacy_frozen.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == inputs
    else:
        runtime.dump(path, inputs)
    return inputs


def legacy(out):
    inputs = frozen_legacy(out)
    directory = out / "legacy_gate"
    directory.mkdir(exist_ok=False)
    settings = Settings.from_env(runtime.ROOT)
    runtime.SECRET = settings.llm_api_key
    from dataclasses import replace

    configure_logging(replace(settings, log_dir=runtime.ROOT / "logs" / out.name / "legacy_gate"))
    audit = runtime.Audit(directory)
    client = httpx.Client(
        transport=runtime.AuditTransport(audit), timeout=settings.request_timeout_seconds
    )
    llm = build_conversation_llm(settings)
    llm._http_client = client
    pipeline = ConversationAnswerPipeline(llm)
    outputs = []
    for row in inputs:
        if row.get("status") == "阻塞":
            outputs.append(row)
            continue
        audit.position = row["id"]
        evidence = [
            EvidencePoolEntry(
                evidence_id=n["node_id"],
                ref=EvidenceRef(
                    document_id=str(n["metadata"].get("ref_doc_id", n["node_id"])),
                    node_id=n["node_id"],
                ),
                text=n["text"],
                source=str(n["metadata"].get("source", "original_saved_context")),
            )
            for n in row["nodes"]
        ]
        task = ConversationTaskInput(
            task_key="t1",
            question_id=uuid5(NAMESPACE_URL, row["id"]),
            question_version=1,
            question_text=row["query"],
            requirements=[row["query"]],
            evidence=evidence,
        )
        answers = {
            "t1": GeneratedItem(
                task_key="t1",
                question_id=task.question_id,
                question_version=1,
                answer=row["answer"],
                evidence_ids=[e.evidence_id for e in evidence],
            )
        }
        try:
            body, metric = pipeline._call(
                pipeline._messages("gate", [task], answers=answers),
                stage="gate",
                input_upper=pipeline._estimate("gate", [task], answers=answers, output=4096),
                max_output=4096,
            )
            parsed = pipeline._gate_parser.parse(body)
            outputs.append(
                {
                    "id": row["id"],
                    "gate": parsed.model_dump(mode="json"),
                    "metric": metric.model_dump(mode="json"),
                    "gold_accept": True,
                }
            )
        except Exception as exc:
            outputs.append({"id": row["id"], "status": "失败", "error": str(exc)})
        runtime.dump(directory / "results.json", outputs)
    client.close()


def prompt_tasks(record):
    result = {}
    for batch in record.get("response", {}).get("trace", {}).get("messages", []):
        for message in batch:
            text = message.get("content") or "".join(
                b.get("text", "") for b in message.get("blocks", [])
            )
            start = text.find('{"evidence_pool"')
            if start < 0:
                continue
            payload = json.JSONDecoder().raw_decode(text[start:])[0]
            pool = payload["evidence_pool"]
            for task in payload["tasks"]:
                result[task["task_key"]] = (task, pool)
    return result


def evaluation_inputs(out, *, include_requirements=False):
    inputs = []
    for stage in ["B", "C1b", "C2"]:
        path = out / stage / "results.json"
        if not path.exists():
            continue
        for record in json.loads(path.read_text(encoding="utf-8")):
            tasks = prompt_tasks(record)
            for item in record.get("response", {}).get("items", []):
                if item["status"] != "answered" or item["task_key"] not in tasks:
                    continue
                task, pool = tasks[item["task_key"]]
                contexts = []
                for eid in task.get("allowed_evidence_ids", []):
                    node = pool[eid]
                    contexts.append(node["text"] if isinstance(node, dict) else str(node))
                facts = [x.get("source_excerpt", "") for x in task.get("user_provided_facts", [])]
                contexts += [
                    "用户提供材料（含显式来源历史，非独立验证知识）：" + f
                    for f in facts if f
                ]
                if task.get("current_conditions"):
                    contexts.append(
                        "本轮范围校验后的有效用户条件（不是知识库事实）："
                        + json.dumps(task["current_conditions"], ensure_ascii=False)
                    )
                # 参考只来自执行前冻结的证据和数字；不把gold场景标签当事实答案。
                reference = "\n".join(contexts)
                if record["group"] == "L02":
                    reference += "\n" + json.dumps(
                        json.loads((out / "gold_provenance.json").read_text(encoding="utf-8"))[
                            "numeric_gold"
                        ]
                    )
                query = task["question"]
                if include_requirements:
                    query += "\n只评估此子题。完成要求：\n" + json.dumps(
                        task.get("requirements", []), ensure_ascii=False
                    )
                    query += "\n此子题禁止事项：\n" + json.dumps(
                        task.get("prohibited_actions", []), ensure_ascii=False
                    )
                inputs.append(
                    {
                        "id": (
                            f"{stage}-{record['group']}-{record['position']:02d}-{item['task_key']}"
                        ),
                        "query": query,
                        "includes_task_requirements": include_requirements,
                        "answer": item["answer"],
                        "contexts": contexts,
                        "reference": reference,
                        "reference_limit": (
                            "来源事实集合而非逐题独立人工标准答案；Correctness仅作辅助，"
                            "不据此发布人工95%通过率"
                        ),
                        "source": str(path),
                        "source_sha": runtime.sha(path),
                    }
                )
    return inputs


async def evaluate(out, *, name="official_evaluators", include_requirements=False):
    directory = out / name
    directory.mkdir(exist_ok=False)
    inputs = evaluation_inputs(out, include_requirements=include_requirements)
    runtime.dump(directory / "frozen_inputs.json", inputs)
    settings = Settings.from_env(runtime.ROOT)
    runtime.SECRET = settings.llm_api_key
    from dataclasses import replace

    configure_logging(replace(settings, log_dir=runtime.ROOT / "logs" / out.name / name))
    audit = runtime.Audit(directory)
    client = httpx.AsyncClient(
        transport=AsyncTransport(audit), timeout=settings.request_timeout_seconds
    )
    llm = build_conversation_llm(settings)
    llm._async_http_client = client
    llm.additional_kwargs = {"extra_body": {"thinking": {"type": "disabled"}}}
    runner = BatchEvalRunner(
        {
            "context_relevancy": ContextRelevancyEvaluator(llm=llm),
            "correctness": CorrectnessEvaluator(llm=llm),
            "relevancy": RelevancyEvaluator(llm=llm),
            "faithfulness": FaithfulnessEvaluator(llm=llm),
        },
        workers=2,
        show_progress=False,
    )
    results = []
    for row in inputs:
        provider_rejection = audit.terminal_provider_rejection()
        if provider_rejection:
            results.append({
                "id": row["id"], "status": "阻塞",
                "execution_state": "not_executed",
                "blocking_reason": provider_rejection,
            })
            runtime.dump(directory / "results.json", results)
            continue
        audit.position = row["id"]
        try:
            evaluations = await runner.aevaluate_response_strs(
                queries=[row["query"]],
                response_strs=[row["answer"]],
                contexts_list=[row["contexts"]],
                correctness={"reference": [row["reference"]]},
            )
            record = {
                "id": row["id"],
                "results": {
                    k: [x.model_dump(mode="json") for x in v] for k, v in evaluations.items()
                },
            }
        except Exception as exc:
            record = {"id": row["id"], "error": str(exc), "status": "失败"}
        results.append(record)
        runtime.dump(directory / "results.json", results)
        print(row["id"], "evaluated", len(results), "/", len(inputs), flush=True)
    await client.aclose()
    runtime.dump(
        directory / "usage.json",
        {
            "model": settings.llm_model,
            "calls": len(audit.calls),
            "known_usage_calls": sum(x.get("usage") is not None for x in audit.calls),
            "known_total_tokens": sum(
                (x.get("usage") or {}).get("total_tokens", 0) for x in audit.calls
            ),
            "dialogue_regenerations": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["freeze_legacy", "legacy", "evaluate"], required=True)
    parser.add_argument("--evaluation-name", default="official_evaluators")
    parser.add_argument("--include-requirements", action="store_true")
    args = parser.parse_args()
    if args.mode == "freeze_legacy":
        frozen_legacy(args.output)
    elif args.mode == "legacy":
        legacy(args.output)
    else:
        asyncio.run(
            evaluate(
                args.output,
                name=args.evaluation_name,
                include_requirements=args.include_requirements,
            )
        )
