"""Isolated runtime and redacted LlamaIndex callbacks for the phase-zero probe."""

from __future__ import annotations

import json
import socket
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "docs/多轮询问模块重建子方案/00-唯一连续回归用例.md"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def serializable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serializable(value), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


class Capture(BaseCallbackHandler):
    def __init__(self, directory, secret):
        super().__init__([], [])
        self.directory, self.secret = directory, secret
        self.stage = "initializing"
        self.records = []

    def append(self, record):
        record.update(timestamp=now(), stage=self.stage)
        text = json.dumps(serializable(record), ensure_ascii=False, default=str)
        if self.secret:
            text = text.replace(self.secret, "[REDACTED]")
        self.records.append(json.loads(text))
        with (self.directory / "llm_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    def on_event_start(self, event_type, payload=None, event_id="", parent_id="", **kwargs):
        if event_type == CBEventType.LLM:
            inputs = {
                getattr(k, "value", str(k)): serializable(v)
                for k, v in (payload or {}).items()
                if getattr(k, "value", str(k)) in {"prompt", "messages"}
            }
            self.append(dict(event="start", event_id=event_id, inputs=inputs))
        return event_id

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        if event_type != CBEventType.LLM:
            return
        record = dict(event="end", event_id=event_id)
        for key, value in (payload or {}).items():
            name = getattr(key, "value", str(key))
            if name in {"response", "completion"}:
                raw = serializable(getattr(value, "raw", None))
                if isinstance(raw, dict):
                    record["provider"] = {k: raw.get(k) for k in ("id", "model", "usage")}
                    record["choices"] = [
                        {
                            "finish_reason": c.get("finish_reason"),
                            "content": c.get("message", {}).get("content", c.get("text")),
                        }
                        for c in raw.get("choices", [])
                    ]
            elif name == "exception":
                record["error"] = str(value)
        self.append(record)

    def start_trace(self, trace_id=None):
        pass

    def end_trace(self, trace_id=None, trace_map=None):
        pass

    def usage(self):
        responses = {
            r["provider"]["id"]: r["provider"]
            for r in self.records
            if r.get("provider", {}).get("id")
        }
        return {
            "llm_calls": len(responses),
            **{
                key: sum((v.get("usage") or {}).get(key, 0) for v in responses.values())
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
        }


@contextmanager
def isolated_postgres(output, log_dir):
    binaries = ROOT / "tools/postgresql/17.11-5/pgsql/bin"
    cluster = output / "postgres_cluster"
    if cluster.exists():
        raise RuntimeError("Refusing to reuse an existing PostgreSQL cluster")
    init = subprocess.run(
        [str(binaries / "initdb.exe"), "-D", str(cluster), "-A", "trust", "-U", "postgres"],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    (log_dir / "postgres_init.log").write_text(init.stdout + init.stderr, encoding="utf-8")
    if init.returncode:
        raise RuntimeError("Isolated PostgreSQL initialization failed")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    dsn = f"host=127.0.0.1 port={port} user=postgres dbname=postgres"
    with (log_dir / "postgres.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                str(binaries / "postgres.exe"),
                "-D",
                str(cluster),
                "-h",
                "127.0.0.1",
                "-p",
                str(port),
            ],
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("Isolated PostgreSQL exited during startup")
                try:
                    with psycopg.connect(dsn, connect_timeout=1):
                        break
                except psycopg.OperationalError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("Isolated PostgreSQL startup timeout")
            yield dsn
        finally:
            result = subprocess.run(
                [str(binaries / "pg_ctl.exe"), "-D", str(cluster), "stop", "-m", "fast", "-w"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            (log_dir / "postgres_stop.log").write_text(
                result.stdout + result.stderr, encoding="utf-8"
            )
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=15)
