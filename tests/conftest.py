from __future__ import annotations

import asyncio
import os
import re
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from llama_index.core.bridge.pydantic import Field
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from qdrant_client import QdrantClient

from story_rag.adapters.model_factory import ModelBundle
from story_rag.config import Settings
from story_rag.container import ApplicationContainer, build_container

FAKE_API_KEY = "sk-test-key-not-a-real-credential"
CHINESE_STORY = (
    "很久以前，山谷里住着一位会讲故事的老人。"
    "他每天黄昏都会坐在青石桥上，给放学的孩子们讲一个关于白龙的故事。"
    "白龙穿过厚厚的云层，把雨水带给干涸的村庄，也带来了新的春天。"
    "老人说，只要还有人愿意听，白龙就会一直住在山谷的云雾里。"
    "孩子们把这些故事记在心里，一代一代地讲了下去。"
)
ENGLISH_STORY = (
    "Once upon a time an old storyteller lived in a quiet valley. "
    "Every evening he sat on the stone bridge and told the children a story about a dragon. "
    "The children loved the part where the white dragon crossed the clouds "
    "and carried rain to the dry village, bringing a new spring with it. "
    "The old man said the dragon would stay in the mist as long as someone listened. "
    "The children kept the words in their hearts and told them again and again."
    "Years later, travellers still stopped by the stone bridge to hear that story. "
    "They said the valley had never been so green as in the year of the white dragon."
)
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
WORD_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class ScriptedLLM(CustomLLM):
    """Deterministic LLM double: pops queued answers and records prompts."""

    responses: list[str] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)
    fallback: str = "无"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name="scripted-llm",
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=4096,
            num_output=512,
        )

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else self.fallback
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        yield self.complete(prompt, formatted=formatted, **kwargs)


class KeywordReranker(BaseNodePostprocessor):
    """Deterministic reranker double: ranks nodes by query term overlap."""

    top_n: int = 5
    calls: int = 0

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        self.calls += 1
        query = query_bundle.query_str if query_bundle else ""
        terms = set(WORD_PATTERN.findall(query.casefold()))
        terms.update(char for char in query if CJK_PATTERN.match(char))
        for node in nodes:
            content = node.node.get_content().casefold()
            node.score = float(sum(content.count(term) for term in terms))
        return sorted(nodes, key=lambda item: item.score or 0.0, reverse=True)[: self.top_n]


@pytest.fixture
def settings(tmp_path, qdrant_url: str) -> Iterator[Settings]:
    api_file = tmp_path / "api.txt"
    api_file.write_text(
        f"{FAKE_API_KEY}\n模型: deepseek-flash\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    configured = Settings(
        project_root=tmp_path,
        api_file=api_file,
        deepseek_base_url="https://api.deepseek.com",
        llm_model="deepseek-flash",
        llm_api_key=FAKE_API_KEY,
        reranker_model_path=tmp_path / "missing-reranker",
        data_dir=data_dir,
        qdrant_url=qdrant_url,
        collection_name=f"t_{uuid.uuid4().hex[:12]}",
        docstore_path=data_dir / "docstore" / "docstore.json",
        log_dir=tmp_path / "logs",
        chunk_size=200,
        chunk_overlap=32,
        vector_top_k=10,
        bm25_top_k=10,
        fusion_top_k=10,
        rerank_top_n=3,
        query_fusion_num_queries=1,
        request_timeout_seconds=30.0,
    )
    yield configured
    with closing(QdrantClient(url=qdrant_url)) as client:
        if client.collection_exists(configured.collection_name):
            client.delete_collection(configured.collection_name)


@pytest.fixture(scope="session")
def qdrant_url(tmp_path_factory) -> Iterator[str]:
    external_url = os.getenv("QDRANT_TEST_URL")
    if external_url:
        yield external_url
        return
    executable = Path(__file__).resolve().parents[1] / "tools" / "qdrant" / "qdrant.exe"
    if not executable.is_file():
        pytest.fail("Run scripts/install_qdrant.ps1 or set QDRANT_TEST_URL before storage tests")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    root = tmp_path_factory.mktemp("qd")
    env = {
        **os.environ,
        "QDRANT__SERVICE__HOST": "127.0.0.1",
        "QDRANT__SERVICE__HTTP_PORT": str(port),
        "QDRANT__SERVICE__GRPC_PORT": str(port + 1),
        "QDRANT__STORAGE__STORAGE_PATH": str(root / "storage"),
        "QDRANT__TELEMETRY_DISABLED": "true",
    }
    with (root / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(executable)],
            cwd=root,
            env=env,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            for _ in range(100):
                if process.poll() is not None:
                    pytest.fail(f"Test Qdrant exited; see {root / 'server.log'}")
                try:
                    if httpx.get(f"{url}/readyz", timeout=0.5).is_success:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail(f"Test Qdrant startup timed out; see {root / 'server.log'}")
            yield url
        finally:
            process.terminate()
            process.wait(timeout=15)


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM(responses=[])


@pytest.fixture
def reranker(settings: Settings) -> KeywordReranker:
    return KeywordReranker(top_n=settings.rerank_top_n)


@pytest.fixture
def container(
    settings: Settings,
    scripted_llm: ScriptedLLM,
    reranker: KeywordReranker,
) -> Iterator[ApplicationContainer]:
    models = ModelBundle(
        llm=scripted_llm,
        embedding=MockEmbedding(embed_dim=8),
        reranker=reranker,
    )
    built = build_container(settings, models=models)
    try:
        yield built
    finally:
        asyncio.run(built.storage.aclose())
