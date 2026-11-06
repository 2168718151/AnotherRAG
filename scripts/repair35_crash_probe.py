"""隔离验证真实供应商已返回分片、应用提交前崩溃后的保守恢复。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4

import httpx
from phase0_probe_support import isolated_postgres

from story_rag.adapters.session_store import PostgresSessionStore
from story_rag.config import Settings
from story_rag.conversation_domain import DatabaseBinding, RequestPayload

ROOT = Path(__file__).resolve().parents[1]


def child(args):
    config = json.loads(args.child.read_text(encoding="utf-8"))
    store = PostgresSessionStore(config["dsn"], schema=config["schema"])
    binding = DatabaseBinding.model_validate(config["binding"])
    session_id = UUID(config["session_id"])
    payload = RequestPayload(message="真实分片后崩溃验收")
    store.claim_request(
        session_id, binding, request_id="crash", payload=payload, expected_revision=0
    )
    store.start_request(session_id, binding, "crash")
    store.mark_model_attempt(session_id, binding, "crash", "generation")
    settings = Settings.from_env(ROOT)
    with (
        httpx.Client(timeout=60) as client,
        client.stream(
            "POST",
            settings.deepseek_base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": "Bearer " + settings.llm_api_key},
            json={
                "model": settings.llm_model,
                "stream": True,
                "max_tokens": 2048,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": "请详细解释Redis的AOF持久化。"}],
            },
        ) as response,
    ):
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                event = json.loads(line[5:])
                if any(
                    choice.get("delta", {}).get("content") for choice in event.get("choices", [])
                ):
                    marker = {
                        "response_id": event.get("id"),
                        "model": event.get("model"),
                        "fragment_sha256": hashlib.sha256(line.encode()).hexdigest(),
                    }
                    Path(config["marker"]).write_text(json.dumps(marker), encoding="utf-8")
                    # Parent terminates this owned process after observing the real fragment.
                    time.sleep(60)
                    raise RuntimeError("Parent failed to terminate the crash probe")
        raise RuntimeError("No content fragment received")


def parent(out):
    out.mkdir(parents=True, exist_ok=False)
    logs = ROOT / "logs" / "repair35_crash" / out.name
    logs.mkdir(parents=True, exist_ok=False)
    with isolated_postgres(out, logs) as dsn:
        store = PostgresSessionStore(dsn, schema="crash_probe")
        store.migrate()
        binding = DatabaseBinding(database_id=uuid4(), storage_fingerprint="d" * 64)
        session = store.create_session("真实分片中断", binding)
        marker = out / "received_fragment.json"
        config = out / "child_config.json"
        config.write_text(
            json.dumps(
                {
                    "dsn": dsn,
                    "schema": store.schema,
                    "binding": binding.model_dump(mode="json"),
                    "session_id": str(session.session_id),
                    "marker": str(marker.resolve()),
                }
            ),
            encoding="utf-8",
        )
        with (logs / "child.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--child", str(config.resolve())],
                cwd=ROOT,
                stdout=log,
                stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            try:
                deadline = time.monotonic() + 60
                while not marker.exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("Child exited before receiving a model fragment")
                    time.sleep(0.1)
                if not marker.exists():
                    raise RuntimeError("No model fragment within the bounded probe")
                process.terminate()
                process.wait(timeout=10)
                recovered = store.recover_orphaned_request(session.session_id, binding, "crash")
                assert recovered.status == "unknown"
                assert store.get_session(session.session_id, binding).revision == 0
                assert store.list_question_versions(session.session_id, binding) == []
                replay = store.claim_request(
                    session.session_id,
                    binding,
                    request_id="crash",
                    expected_revision=0,
                    payload=RequestPayload(message="真实分片后崩溃验收"),
                )
                assert replay.status == "unknown"
                report = {
                    "scenario": "T05",
                    "layer": "live_provider_owned_child",
                    "passed": True,
                    "request_attempts": 1,
                    "automatic_retries": 0,
                    "revision": 0,
                    "provider_usage": None,
                    "usage_reason": "terminated before usage tail",
                    "fragment": json.loads(marker.read_text(encoding="utf-8")),
                    "recovered_request": recovered.model_dump(mode="json"),
                    "replayed_status": replay.status,
                }
                (out / "result.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print("T05 real provider fragment -> owned child terminated -> unknown; no retry")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    if args.child:
        child(args)
    elif args.output:
        parent(args.output)
    else:
        parser.error("--output is required")
