from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever

from story_rag.adapters.storage import StorageBundle
from story_rag.config import Settings
from story_rag.domain import NodeTrace, QueryRequest, RetrievalResult
from story_rag.logging_config import TraceWriter

logger = logging.getLogger(__name__)
BILINGUAL_TOKEN_PATTERN = r"(?u)[\u3400-\u9fff]|\w+"


class EmptyRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return []


class StoryRetrievalPipeline:
    def __init__(
        self,
        settings: Settings,
        storage: StorageBundle,
        reranker: Any,
        retriever_builder: Callable[[], tuple[BaseRetriever, BaseRetriever, BaseRetriever]],
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._reranker = reranker
        self._retriever_builder = retriever_builder
        self._trace_writer = trace_writer
        self._retrievers: tuple[BaseRetriever, BaseRetriever, BaseRetriever] | None = None
        self._built_revision = -1
        self._lock = threading.Lock()

    def _ensure_retrievers(self) -> tuple[BaseRetriever, BaseRetriever, BaseRetriever]:
        """Rebuild BM25 and fusion retrievers whenever the corpus changes."""
        with self._lock:
            if self._retrievers is None or self._built_revision != self._storage.revision:
                self._retrievers = self._retriever_builder()
                self._built_revision = self._storage.revision
                logger.info(
                    "Rebuilt retrievers: revision=%s nodes=%s",
                    self._built_revision,
                    self._storage.node_count(),
                )
            return self._retrievers

    def retrieve(self, request: QueryRequest) -> RetrievalResult:
        return self._retrieve(request, include_trace=False)

    def retrieve_trace(self, request: QueryRequest) -> RetrievalResult:
        return self._retrieve(request, include_trace=True)

    def _retrieve(self, request: QueryRequest, *, include_trace: bool) -> RetrievalResult:
        with self._storage.lock:
            self._storage.ensure_available()
            return self._retrieve_locked(request, include_trace=include_trace)

    def _retrieve_locked(self, request: QueryRequest, *, include_trace: bool) -> RetrievalResult:
        start = time.perf_counter()
        query_bundle = QueryBundle(query_str=request.query)

        if not self._storage.docstore.docs:
            return RetrievalResult(
                query=request.query,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        vector_retriever, bm25_retriever, fusion_retriever = self._ensure_retrievers()

        vector_candidates: list[NodeWithScore] = []
        bm25_candidates: list[NodeWithScore] = []
        if include_trace:
            vector_candidates = vector_retriever.retrieve(query_bundle)
            bm25_candidates = bm25_retriever.retrieve(query_bundle)

        fused_candidates = fusion_retriever.retrieve(query_bundle)
        fused_snapshot = self._to_traces(fused_candidates, "rrf")
        rerank_start = time.perf_counter()
        reranked = self._reranker.postprocess_nodes(
            fused_candidates,
            query_bundle=query_bundle,
        )
        rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000
        top_k = min(request.top_k or self._settings.rerank_top_n, self._settings.rerank_top_n)
        final_nodes = reranked[:top_k]

        result = RetrievalResult(
            query=request.query,
            rewritten_queries=[request.query],
            nodes=self._to_traces(final_nodes, "rerank", reranked=True),
            vector_candidates=self._to_traces(vector_candidates, "vector"),
            bm25_candidates=self._to_traces(bm25_candidates, "bm25"),
            fused_candidates=fused_snapshot,
            latency_ms=(time.perf_counter() - start) * 1000,
            rerank_latency_ms=rerank_latency_ms,
        )
        logger.info(
            "Retrieved query=<redacted> query_length=%s fused=%s reranked=%s latency_ms=%.2f",
            len(request.query),
            len(fused_candidates),
            len(final_nodes),
            result.latency_ms,
        )
        if self._trace_writer and include_trace:
            self._trace_writer.write(
                f"retrieval-{abs(hash(request.query))}",
                "retrieval",
                {"result": result.model_dump()},
            )
        return result

    @staticmethod
    def _to_traces(
        nodes: list[NodeWithScore],
        retriever: str,
        *,
        reranked: bool = False,
    ) -> list[NodeTrace]:
        traces: list[NodeTrace] = []
        for rank, item in enumerate(nodes, start=1):
            metadata = dict(item.node.metadata)
            traces.append(
                NodeTrace(
                    node_id=item.node.node_id,
                    text=item.node.get_content(),
                    retriever=retriever,
                    rank=rank,
                    score=item.score,
                    rerank_score=item.score if reranked else None,
                    metadata=metadata,
                )
            )
        return traces


def build_retrieval_pipeline(
    settings: Settings,
    storage: StorageBundle,
    llm: Any,
    reranker: SentenceTransformerRerank,
    trace_writer: TraceWriter | None = None,
) -> StoryRetrievalPipeline:
    def build() -> tuple[BaseRetriever, BaseRetriever, BaseRetriever]:
        vector_retriever = storage.index.as_retriever(similarity_top_k=settings.vector_top_k)
        nodes = storage.text_nodes()
        bm25_retriever: BaseRetriever = (
            BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=settings.bm25_top_k,
                token_pattern=BILINGUAL_TOKEN_PATTERN,
            )
            if nodes
            else EmptyRetriever()
        )
        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=settings.fusion_top_k,
            num_queries=settings.query_fusion_num_queries,
            mode="reciprocal_rerank",
            use_async=True,
            llm=llm,
            verbose=False,
        )
        return vector_retriever, bm25_retriever, fusion_retriever

    return StoryRetrievalPipeline(
        settings=settings,
        storage=storage,
        reranker=reranker,
        retriever_builder=build,
        trace_writer=trace_writer,
    )
