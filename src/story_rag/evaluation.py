from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.evaluation import (
    BatchEvalRunner,
    ContextRelevancyEvaluator,
    CorrectnessEvaluator,
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from story_rag.config import Settings
from story_rag.container import ApplicationContainer, build_container
from story_rag.domain import QueryRequest


class StoryQueryEngine(BaseQueryEngine):
    def __init__(self, workflow: Any) -> None:
        super().__init__(callback_manager=None)
        self._workflow = workflow

    def _query(self, query_bundle: QueryBundle) -> Response:
        return asyncio.run(self._aquery(query_bundle))

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        result = await self._workflow.answer(QueryRequest(query=query_bundle.query_str))
        source_nodes = [
            NodeWithScore(
                node=TextNode(
                    id_=str(index),
                    text=context,
                    metadata={"citation": citation},
                ),
                score=1.0,
            )
            for index, (context, citation) in enumerate(
                zip(result.contexts, result.citations, strict=False), start=1
            )
        ]
        return Response(response=result.answer, source_nodes=source_nodes)

    def _get_prompt_modules(self) -> dict[str, Any]:
        return {}


class StoryEvaluationPipeline:
    def __init__(
        self,
        settings: Settings,
        llm: Any,
        container: ApplicationContainer,
        report_dir: Path | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._container = container
        self._report_dir = report_dir or settings.project_root / "reports"
        self._report_dir.mkdir(parents=True, exist_ok=True)

    def generate_dataset(
        self,
        source_paths: list[Path],
        output_path: Path,
        *,
        max_questions: int = 12,
        questions_per_chunk: int = 2,
    ) -> Path:
        nodes = self._load_nodes(source_paths)
        if not nodes:
            raise ValueError("No TXT nodes found for dataset generation")

        generators = [
            DatasetGenerator(
                nodes=nodes,
                llm=self._llm,
                num_questions_per_chunk=questions_per_chunk,
                question_gen_query=(
                    "基于给定文本生成 {num_questions_per_chunk} 个可独立回答的中文问题。"
                    "每行一个问题，不要编号，不要给出答案。"
                ),
            ),
            DatasetGenerator(
                nodes=nodes,
                llm=self._llm,
                num_questions_per_chunk=questions_per_chunk,
                question_gen_query=(
                    "Generate {num_questions_per_chunk} standalone questions in English "
                    "from the supplied text. Return one question per line, without numbering "
                    "or answers."
                ),
            ),
        ]
        items: list[dict[str, Any]] = []
        for generator in generators:
            dataset = generator.generate_dataset_from_nodes(num=max_questions)
            for query_id, query in dataset.queries.items():
                reference = dataset.responses.get(query_id, "")
                items.append(
                    {
                        "id": f"{len(items):04d}",
                        "query": query,
                        "reference": reference,
                        "language": detect_language(query),
                    }
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "schema_version": 1,
            "generator": "llama_index.DatasetGenerator",
            "items": items,
        }
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def run(self, dataset_path: Path) -> dict[str, Any]:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        items = dataset.get("items", [])
        if not items:
            raise ValueError("Evaluation dataset contains no items")

        queries = [item["query"] for item in items]
        references = [item.get("reference", "") for item in items]
        languages = [item.get("language", "unknown") for item in items]

        evaluators = {
            "context_relevancy": ContextRelevancyEvaluator(llm=self._llm),
            "correctness": CorrectnessEvaluator(llm=self._llm),
            "relevancy": RelevancyEvaluator(llm=self._llm),
            "faithfulness": FaithfulnessEvaluator(llm=self._llm),
        }
        runner = BatchEvalRunner(evaluators, workers=2, show_progress=False)
        query_engine = StoryQueryEngine(workflow=self._container.workflow)
        results = runner.evaluate_queries(
            query_engine,
            queries=queries,
            correctness={"reference": references},
        )
        report = build_report(results, languages)
        report["dataset_path"] = str(dataset_path)
        report_path = self._report_dir / "evaluation_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    def preview_case(self, dataset_path: Path, index: int = 0) -> dict[str, Any]:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        return dataset["items"][index]

    def _load_nodes(self, source_paths: list[Path]) -> list[Any]:
        files: list[Path] = []
        for path in source_paths:
            if path.is_dir():
                files.extend(sorted(path.rglob("*.txt")))
            elif path.is_file() and path.suffix.casefold() == ".txt":
                files.append(path)
        nodes: list[Any] = []
        for path in files:
            nodes.extend(
                self._container.ingestion.parse_nodes(
                    path.read_bytes(),
                    source_name=str(path.resolve()),
                    title=path.stem,
                )
            )
        return nodes


def detect_language(text: str) -> str:
    has_cjk = bool(re.search(r"[\u3400-\u9fff]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return "unknown"


def build_report(
    results: dict[str, list[Any]],
    languages: list[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "evaluators": {},
        "by_language": {},
        "items": [],
    }
    for evaluator_name, evaluations in results.items():
        scores = [float(item.score) for item in evaluations if item.score is not None]
        report["evaluators"][evaluator_name] = {
            "count": len(evaluations),
            "scored_count": len(scores),
            "mean_score": sum(scores) / len(scores) if scores else None,
            "passing_count": sum(bool(item.passing) for item in evaluations),
        }
        grouped: dict[str, list[float]] = {}
        for language, item in zip(languages, evaluations, strict=False):
            if item.score is None:
                continue
            grouped.setdefault(language, []).append(float(item.score))
        report["by_language"][evaluator_name] = {
            language: {
                "count": len(values),
                "mean_score": sum(values) / len(values),
            }
            for language, values in grouped.items()
        }

    for index in range(len(languages)):
        item_result: dict[str, Any] = {"language": languages[index]}
        for evaluator_name, evaluations in results.items():
            evaluation = evaluations[index]
            item_result[evaluator_name] = {
                "score": evaluation.score,
                "passing": evaluation.passing,
                "feedback": evaluation.feedback,
                "invalid_result": evaluation.invalid_result,
            }
        report["items"].append(item_result)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="story-rag-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--path", type=Path, action="append", required=True)
    generate.add_argument("--output", type=Path, default=Path("reports/dataset_v0.json"))
    run = subparsers.add_parser("run")
    run.add_argument("--dataset", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    container = build_container(settings)
    try:
        _evaluate(container, args)
    finally:
        asyncio.run(container.storage.aclose())


def _evaluate(container: ApplicationContainer, args: argparse.Namespace) -> None:
    settings = container.settings
    pipeline = StoryEvaluationPipeline(settings, container.llm, container)
    if args.command == "generate":
        output = pipeline.generate_dataset(args.path, args.output)
        print(output)
    elif args.command == "run":
        report = pipeline.run(args.dataset)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
