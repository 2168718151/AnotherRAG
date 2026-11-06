"""Sample process working sets and device-wide VRAM without reading credentials."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil

MIB = 1024**2
GROUPS = ("python", "qdrant", "lm_studio")


def summarize_resources(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"sample_count": len(samples), "processes": {}}
    timestamps = [row["elapsed_seconds"] for row in samples]
    gaps = sorted(
        later - earlier
        for earlier, later in zip(timestamps[:-1], timestamps[1:], strict=True)
    )
    result["observed_interval_seconds"] = {
        "count": len(gaps),
        "mean": statistics.mean(gaps) if gaps else None,
        "p50": statistics.median(gaps) if gaps else None,
        "p95": gaps[math.ceil(len(gaps) * 0.95) - 1] if gaps else None,
        "max": max(gaps) if gaps else None,
    }
    for group in (*GROUPS, "aggregate"):
        values = [row["processes"][group] for row in samples]
        result["processes"][group] = (
            {
                f"{field}_{stat}_mib": function([item[field] / MIB for item in values])
                for field in ("rss_bytes", "private_bytes")
                for stat, function in (("peak", max), ("mean", statistics.mean))
            }
            if values
            else {}
        )
    gpu = [row["gpu"] for row in samples if row["gpu"] is not None]
    result["gpu"] = {
        "sample_count": len(gpu),
        "missing_sample_count": len(samples) - len(gpu),
        "device_used_peak_mib": max(row["used_mib"] for row in gpu) if gpu else None,
        "device_used_baseline_mib": gpu[0]["used_mib"] if gpu else None,
        "device_total_mib": gpu[0]["total_mib"] if gpu else None,
    }
    result["system_used_peak_mib"] = max(
        (row["system_used_bytes"] / MIB for row in samples), default=None
    )
    result["notes"] = [
        "Aggregate RSS sums simultaneous working sets and may double-count shared pages.",
        "Windows private bytes are committed allocations, not physical RAM usage.",
        "GPU figures cover the entire device, including unrelated apps; they are sampled peaks.",
        "Configured wait excludes measurement time; use observed intervals for sampling cadence.",
    ]
    return result


class ResourceSampler:
    def __init__(self, output_path: Path, interval: float = 1.0) -> None:
        if interval <= 0:
            raise ValueError("Resource interval must be positive")
        self.output_path = output_path
        self.interval = interval
        self.phase = "startup"
        self.qdrant_pid: int | None = None
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = time.perf_counter()
        self._gpu_command = shutil.which("nvidia-smi")
        self._errors: list[str] = []

    def start(self) -> None:
        self._handle = self.output_path.open("x", encoding="utf-8")
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("Resource sampler did not stop")
        self._sample()
        self._handle.close()
        summary = summarize_resources(self.samples)
        summary["interval_seconds"] = self.interval
        summary["errors"] = self._errors
        return summary

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception as exc:
                self._errors.append(type(exc).__name__)

    def _sample(self) -> None:
        processes = {group: {"rss_bytes": 0, "private_bytes": 0, "pids": []} for group in GROUPS}
        inventory = list(psutil.process_iter(["pid", "name", "ppid"]))
        lm_pids = {
            item.pid
            for item in inventory
            if any(
                part in (item.info["name"] or "").casefold()
                for part in ("lm studio", "lmstudio", "llmster", "llama")
            )
        }
        for _ in range(5):
            children = {item.pid for item in inventory if item.info["ppid"] in lm_pids}
            if children <= lm_pids:
                break
            lm_pids.update(children)
        for process in inventory:
            group = (
                "python"
                if process.pid == os.getpid()
                else "qdrant"
                if process.pid == self.qdrant_pid
                else "lm_studio"
                if process.pid in lm_pids
                else None
            )
            if group is None:
                continue
            try:
                memory = process.memory_info()
                processes[group]["rss_bytes"] += memory.rss
                processes[group]["private_bytes"] += getattr(memory, "private", memory.vms)
                processes[group]["pids"].append(process.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        processes["aggregate"] = {
            field: sum(processes[group][field] for group in GROUPS)
            for field in ("rss_bytes", "private_bytes")
        }
        memory = psutil.virtual_memory()
        sample = {
            "elapsed_seconds": time.perf_counter() - self._started,
            "phase": self.phase,
            "processes": processes,
            "system_used_bytes": memory.total - memory.available,
            "gpu": self._gpu(),
        }
        self.samples.append(sample)
        self._handle.write(json.dumps(sample) + "\n")
        self._handle.flush()

    def _gpu(self) -> dict[str, Any] | None:
        if not self._gpu_command:
            return None
        try:
            result = subprocess.run(
                [
                    self._gpu_command,
                    "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            row = next(csv.reader(io.StringIO(result.stdout)))
            return {
                "index": int(row[0]),
                "used_mib": float(row[1]),
                "total_mib": float(row[2]),
                "utilization_percent": float(row[3]),
            }
        except (subprocess.SubprocessError, ValueError, StopIteration):
            return None
