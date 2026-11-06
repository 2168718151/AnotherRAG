"""Diagnostic entry point for the real local models (no DeepSeek calls).

Usage:
    uv run python scripts/verify_local_models.py

It checks the LM Studio embedding endpoint and loads the local Qwen3 reranker
through the same LlamaIndex components the service uses.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode  # noqa: E402

from story_rag.adapters.model_factory import build_embedding_model, build_reranker  # noqa: E402
from story_rag.config import Settings  # noqa: E402

PAIRS = [
    ("What did the white dragon bring to the village?", "The white dragon brought rain."),
    ("What did the white dragon bring to the village?", "A boy bought a bicycle."),
]


def check_embedding(settings: Settings) -> dict[str, object]:
    try:
        model = build_embedding_model(settings)
        vector = model.get_text_embedding("白龙带来了雨水")
    except Exception as exc:  # noqa: BLE001 - diagnostic summary
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "dimension": len(vector)}


def check_reranker(settings: Settings) -> dict[str, object]:
    started = time.perf_counter()
    try:
        reranker = build_reranker(settings)
        nodes = [
            NodeWithScore(node=TextNode(text=text, id_=f"n{index}"), score=0.0)
            for index, (_, text) in enumerate(PAIRS)
        ]
        ranked = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(PAIRS[0][0]))
    except Exception as exc:  # noqa: BLE001 - diagnostic summary
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "load_seconds": round(time.perf_counter() - started, 1),
        "top_text": ranked[0].node.get_content(),
        "scores": [round(float(node.score or 0.0), 4) for node in ranked],
    }


def main() -> int:
    settings = Settings.from_env()
    report = {
        "embedding_model": settings.embedding_model,
        "lmstudio_base_url": settings.lmstudio_base_url,
        "embedding": check_embedding(settings),
        "reranker_model": str(settings.reranker_model_path),
        "reranker": check_reranker(settings),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["reranker"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
