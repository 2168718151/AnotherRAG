from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class DiagnosticArchive:
    """Request-bound snapshots, published atomically without overwriting files."""

    def __init__(self, root: Path, session_id: str, request_id: str, attempt: str):
        self.identity = {"session_id": session_id, "request_id": request_id, "attempt": attempt}
        parts = [
            hashlib.sha256(value.encode()).hexdigest()
            for value in (session_id, request_id, attempt)
        ]
        self.directory = root.joinpath(*parts)

    def write(self, stage: str, payload: Any, *, snapshot_id: str | None = None) -> Path:
        name = hashlib.sha256((snapshot_id or uuid4().hex).encode()).hexdigest() + ".json"
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / name
        fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {**self.identity, "stage": stage, "payload": payload},
                    handle,
                    ensure_ascii=False,
                    default=str,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return target


@dataclass
class ModelCallContext:
    before_call: Callable[[str], None]
    archive: DiagnosticArchive | None = None


model_call_context: ContextVar[ModelCallContext | None] = ContextVar(
    "model_call_context", default=None
)


def before_model_call(stage: str, payload: Any) -> None:
    context = model_call_context.get()
    if context is not None:
        context.before_call(stage)
        if context.archive is not None:
            context.archive.write(stage + "_input", payload)


def after_model_call(stage: str, payload: Any) -> None:
    context = model_call_context.get()
    if context is not None and context.archive is not None:
        try:
            context.archive.write(stage + "_output", payload)
        except OSError:
            # A failed optional export must not discard a received paid response.
            logger.exception("Conversation diagnostic output could not be archived")
