from __future__ import annotations

import logging
import sys

from story_rag.config import Settings
from story_rag.domain import QueryRequest
from tests.conftest import CHINESE_STORY, FAKE_API_KEY, ScriptedLLM


def _file_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]


def test_logging_uses_files_instead_of_console(container, settings: Settings) -> None:
    root = logging.getLogger()
    console_handlers = [
        handler
        for handler in root.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        and getattr(handler, "stream", None) in {sys.stdout, sys.stderr}
    ]

    assert console_handlers == []
    assert _file_handlers(root)
    assert (settings.log_dir / "app.log").is_file()
    assert (settings.log_dir / "error.log").is_file()
    assert (settings.log_dir / "llama_index.log").is_file()


def test_application_log_records_pipeline_events(container, settings: Settings) -> None:
    container.ingestion.ingest_bytes(CHINESE_STORY.encode("utf-8"), source_name="dragon.txt")
    container.retrieval.retrieve(QueryRequest(query="白龙"))

    content = (settings.log_dir / "app.log").read_text(encoding="utf-8")

    assert "Ingested document=" in content
    assert "Rebuilt retrievers" in content
    assert "Retrieved query=" in content
    assert FAKE_API_KEY not in content


def test_llama_index_logger_writes_to_its_own_file(container, settings: Settings) -> None:
    llama_logger = logging.getLogger("llama_index")
    assert llama_logger.propagate is False
    assert _file_handlers(llama_logger)

    logging.getLogger("llama_index.test").debug("llama_index-routing-marker")

    content = (settings.log_dir / "llama_index.log").read_text(encoding="utf-8")
    app_content = (settings.log_dir / "app.log").read_text(encoding="utf-8")

    assert "llama_index-routing-marker" in content
    assert "llama_index-routing-marker" not in app_content


def test_written_workflow_trace_is_jsonl(
    container,
    settings: Settings,
    scripted_llm: ScriptedLLM,
) -> None:
    container.ingestion.ingest_bytes(CHINESE_STORY.encode("utf-8"), source_name="dragon.txt")
    scripted_llm.responses = ["白龙把雨水带给了村子。", "Yes"]

    container.generation.answer(QueryRequest(query="白龙把雨水带给了谁？"))

    traces = list((settings.log_dir / "traces").glob("*.jsonl"))
    assert traces
    assert any("generation" in trace.read_text(encoding="utf-8") for trace in traces)
