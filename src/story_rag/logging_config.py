from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from story_rag.config import Settings

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 5


def _file_handler(path: Path, level: int) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    return handler


def configure_logging(settings: Settings) -> None:
    _set_console_env()
    logs_dir = settings.log_dir
    traces_dir = logs_dir / "traces"
    logs_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(_file_handler(logs_dir / "app.log", logging.DEBUG))
    root_logger.addHandler(_file_handler(logs_dir / "error.log", logging.ERROR))

    llama_logger = logging.getLogger("llama_index")
    llama_logger.handlers.clear()
    llama_logger.setLevel(getattr(logging, settings.llama_index_log_level, logging.DEBUG))
    llama_logger.propagate = False
    llama_logger.addHandler(
        _file_handler(
            logs_dir / "llama_index.log",
            getattr(logging, settings.llama_index_log_level, logging.DEBUG),
        )
    )

    for noisy_logger in ("httpx", "httpcore", "openai", "qdrant_client"):
        logging.getLogger(noisy_logger).setLevel(logging.INFO)

    for loud_logger in ("transformers", "sentence_transformers", "huggingface_hub"):
        # Warnings and errors still land in logs/app.log instead of the console.
        logging.getLogger(loud_logger).setLevel(logging.WARNING)

    logging.captureWarnings(True)
    logging.getLogger(__name__).info("Logging configured at %s", logs_dir)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def silence_third_party_console() -> None:
    """Keep model-loading banners and progress bars out of the console."""
    _set_console_env()
    for name in ("transformers", "sentence_transformers", "huggingface_hub"):
        # A null handler stops logging's last-resort stderr output; records still
        # propagate to the file handlers once configure_logging runs.
        library_logger = logging.getLogger(name)
        if not any(isinstance(handler, logging.NullHandler) for handler in library_logger.handlers):
            library_logger.addHandler(logging.NullHandler())
    try:
        from transformers.utils import logging as hf_logging
    except ImportError:
        return
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()


def _set_console_env() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class TraceWriter:
    def __init__(self, log_dir: Path) -> None:
        self._trace_dir = log_dir / "traces"
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, trace_id: str, event: str, payload: dict[str, Any]) -> None:
        safe_trace_id = "".join(ch for ch in trace_id if ch.isalnum() or ch in "-_")
        if not safe_trace_id:
            safe_trace_id = uuid.uuid4().hex
        path = self._trace_dir / f"{safe_trace_id}.jsonl"
        record = {"event": event, **payload}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
