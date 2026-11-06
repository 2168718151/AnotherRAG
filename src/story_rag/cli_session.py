from __future__ import annotations

import asyncio
import json
import shlex
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.llms import MockLLM

from story_rag.adapters.database_registry import DatabaseRecord
from story_rag.config import load_api_config, preview_model_name
from story_rag.domain import DatabaseError, NodeTrace, QueryRequest
from story_rag.pipelines.databases import DatabasesPipeline
from story_rag.pipelines.documents import DocumentManagementPipeline
from story_rag.pipelines.generation import StoryGenerationPipeline
from story_rag.pipelines.retrieval import build_retrieval_pipeline


class PreviewLLM(MockLLM):
    def complete(self, *args: Any, **kwargs: Any):
        raise RuntimeError("LLM calls are disabled in the query session")

    def stream_complete(self, *args: Any, **kwargs: Any):
        raise RuntimeError("LLM calls are disabled in the query session")


class StageSnapshot(BaseCallbackHandler):
    def __init__(self, stage: str) -> None:
        super().__init__([], [])
        self.stage = stage
        self.nodes: list[dict[str, Any]] = []
        self.count = 0
        self.elapsed_ms = 0.0
        self.starts: dict[str, float] = {}

    def on_event_start(self, event_type, payload=None, event_id="", parent_id="", **kwargs):
        if event_type == CBEventType.RETRIEVE:
            self.starts[event_id] = time.perf_counter()
        return event_id

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        if event_type == CBEventType.RETRIEVE and payload and EventPayload.NODES in payload:
            self.count += 1
            if event_id in self.starts:
                self.elapsed_ms += (time.perf_counter() - self.starts.pop(event_id)) * 1000
            self.nodes = [
                NodeTrace(
                    node_id=item.node.node_id,
                    text=item.node.get_content(),
                    retriever=self.stage,
                    rank=i,
                    score=item.score,
                    metadata=dict(item.node.metadata),
                ).model_dump()
                for i, item in enumerate(payload[EventPayload.NODES], 1)
            ]

    def start_trace(self, trace_id=None):
        pass

    def end_trace(self, trace_id=None, trace_map=None):
        pass


def split_command(line: str) -> list[str]:
    lexer = shlex.shlex(line, posix=False)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return [
        token[1:-1] if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'" else token
        for token in lexer
    ]


class QuerySession:
    def __init__(
        self, databases: DatabasesPipeline, *, models_factory: Callable | None = None
    ) -> None:
        self.databases = databases
        self.record: DatabaseRecord | None = None
        self.storage: Any = None
        self.models: Any = None
        self.retrieval: Any = None
        self.conversation_store: Any = None
        self.conversation_workflow: Any = None
        self.selected_conversation: Any = None
        self.models_factory = models_factory
        self.model_name = preview_model_name(
            databases.settings.api_file, databases.settings.llm_model
        )

    def select(self, name: str | None = None, db_id: str | None = None) -> dict[str, Any]:
        record = self.databases.resolve(name, db_id)
        if self.record and self.record.db_id == record.db_id:
            return self.current()
        storage = self.databases.open(record)
        old_storage = self.storage
        self.record, self.storage, self.retrieval = record, storage, None
        self.conversation_workflow = None
        self.selected_conversation = None
        if old_storage:
            asyncio.run(old_storage.aclose())
        return self.current()

    def current(self) -> dict[str, Any]:
        if not self.record:
            raise DatabaseError("DB_NOT_SELECTED", "Select a database with db use first", 2)
        return {
            "db_id": self.record.db_id,
            "name": self.record.name,
            "status": self.record.status,
            "storage_fingerprint": self.storage.database_id,
            "document_count": self.storage.document_count(),
            "node_count": self.storage.node_count(),
        }

    def release(self) -> dict[str, str]:
        if self.storage:
            asyncio.run(self.storage.aclose())
        self.record, self.storage, self.retrieval = None, None, None
        self.conversation_workflow = None
        self.selected_conversation = None
        return {"status": "released"}

    def close(self) -> None:
        self.release()
        self.models = None

    def documents(self) -> DocumentManagementPipeline:
        self.current()
        return DocumentManagementPipeline(self.storage)

    def _conversation_scope(self):
        from uuid import UUID

        from story_rag.adapters.session_store import build_session_store
        from story_rag.conversation_domain import DatabaseBinding

        self.current()
        if self.conversation_store is None:
            self.conversation_store = build_session_store(self.databases.settings)
        return self.conversation_store, DatabaseBinding(
            database_id=UUID(self.record.db_id),
            storage_fingerprint=self.storage.database_id,
        )

    def create_conversation(self, name: str) -> dict[str, Any]:
        store, binding = self._conversation_scope()
        self.selected_conversation = store.create_session(name, binding)
        return self.selected_conversation.model_dump(mode="json")

    def list_conversations(self) -> list[dict[str, Any]]:
        store, binding = self._conversation_scope()
        return [item.model_dump(mode="json") for item in store.list_sessions(binding)]

    def use_conversation(self, session_id: str) -> dict[str, Any]:
        from uuid import UUID

        store, binding = self._conversation_scope()
        selected = store.get_session(UUID(session_id), binding)
        self.selected_conversation = selected
        return selected.model_dump(mode="json")

    def archive_conversation(self, session_id: str) -> dict[str, Any]:
        from uuid import UUID

        store, binding = self._conversation_scope()
        target = store.get_session(UUID(session_id), binding)
        archived = store.archive_session(
            target.session_id, binding, expected_revision=target.revision
        )
        if (
            self.selected_conversation
            and self.selected_conversation.session_id == target.session_id
        ):
            self.selected_conversation = None
        return archived.model_dump(mode="json")

    def _selected_scope(self):
        store, binding = self._conversation_scope()
        if self.selected_conversation is None:
            raise DatabaseError("CONVERSATION_NOT_SELECTED", "Select a conversation first", 2)
        session = store.get_session(self.selected_conversation.session_id, binding)
        self.selected_conversation = session
        return store, binding, session

    def list_questions(self) -> list[dict[str, Any]]:
        store, binding, session = self._selected_scope()
        return [
            item.model_dump(mode="json")
            for item in store.current_questions_at(
                session.session_id, binding, as_of_revision=session.revision
            )
        ]

    def question_versions(self, question_no: int) -> list[dict[str, Any]]:
        store, binding, session = self._selected_scope()
        current = store.read_questions_at(
            session.session_id,
            binding,
            as_of_revision=session.revision,
            question_no=question_no,
        )
        if len(current) != 1:
            raise DatabaseError("QUESTION_NOT_FOUND", "Question number is not unique", 2)
        return [
            item.model_dump(mode="json")
            for item in store.list_question_versions(
                session.session_id, binding, current[0].question_id
            )
        ]

    def recover_conversation_index(self) -> dict[str, int]:
        store, binding, session = self._selected_scope()
        processed = store.process_history_index(session.session_id, binding)
        return {
            "processed": processed,
            "pending": store.pending_history_index(session.session_id, binding),
        }

    def acknowledge_unknown_request(self, request_id: str) -> dict[str, Any]:
        store, binding, session = self._selected_scope()
        return store.acknowledge_unknown_request(
            session.session_id,
            binding,
            request_id,
            acknowledge_result_unknown=True,
        ).model_dump(mode="json")

    def recover_orphaned_request(self, request_id: str) -> dict[str, Any]:
        store, binding, session = self._selected_scope()
        return store.recover_orphaned_request(
            session.session_id,
            binding,
            request_id,
        ).model_dump(mode="json")

    def ask(
        self,
        text: str,
        *,
        request_id: str | None = None,
        expected_revision: int | None = None,
        question_no: int | None = None,
        use_history: bool = True,
    ) -> dict[str, Any]:
        from story_rag.adapters.model_factory import build_conversation_llm
        from story_rag.conversation_domain import RequestPayload
        from story_rag.conversation_workflow import ConversationWorkflow
        from story_rag.pipelines.conversation_generation import ConversationAnswerPipeline
        from story_rag.pipelines.evidence_mapping import EvidenceMappingPipeline
        from story_rag.pipelines.history_resolution import HistoryResolutionPipeline
        from story_rag.pipelines.task_planning import TaskPlanningPipeline

        if question_no is not None and not use_history:
            raise DatabaseError(
                "HISTORY_DISABLED", "--question cannot be used with --no-history", 2
            )
        store, binding, session = self._selected_scope()
        reply_to = None
        if question_no is not None:
            current = store.read_questions_at(
                session.session_id,
                binding,
                as_of_revision=session.revision,
                question_no=question_no,
            )
            if len(current) != 1:
                raise DatabaseError("QUESTION_NOT_FOUND", "Question number is not unique", 2)
            reply_to = current[0].question_id
        if self.conversation_workflow is None:
            api_config = load_api_config(self.storage.settings.api_file)
            llm_settings = replace(
                self.storage.settings,
                llm_model=api_config.model,
                llm_api_key=api_config.api_key,
            )
            llm = build_conversation_llm(llm_settings)
            if self.retrieval is None:
                self._ensure_retrieval()
            self.conversation_workflow = ConversationWorkflow(
                store=store,
                planner=TaskPlanningPipeline(llm),
                history=HistoryResolutionPipeline(store),
                evidence=EvidenceMappingPipeline(store, self.storage, self.retrieval),
                generator=ConversationAnswerPipeline(llm),
            )
        result = asyncio.run(
            self.conversation_workflow.answer(
                session.session_id,
                binding,
                request_id=request_id or str(uuid.uuid4()),
                expected_revision=(
                    session.revision if expected_revision is None else expected_revision
                ),
                payload=RequestPayload(
                    message=text,
                    reply_to_question_id=reply_to,
                    use_history=use_history,
                ),
            )
        )
        if result.status == "succeeded":
            self.selected_conversation = store.get_session(session.session_id, binding)
            if result.answer_not_retained:
                answer = store.get_display_answer(
                    session.session_id, binding, result.request.turn_id
                )
                if answer is not None:
                    result = result.model_copy(
                        update={"answer": answer, "answer_not_retained": False}
                    )
        return result.model_dump(mode="json")

    def _ensure_retrieval(self) -> None:
        settings = replace(
            self.storage.settings,
            query_fusion_num_queries=1,
            llm_model=self.model_name,
        )
        if self.models is None:
            if self.models_factory:
                self.models = self.models_factory(settings)
            else:
                from story_rag.adapters.model_factory import (
                    ModelBundle,
                    build_embedding_model,
                    build_reranker,
                )

                self.models = ModelBundle(
                    llm=PreviewLLM(),
                    embedding=build_embedding_model(settings),
                    reranker=build_reranker(settings),
                )
        if self.retrieval is None:
            dimension = len(self.models.embedding.get_text_embedding("embedding dimension probe"))
            if dimension != self.record.embedding_dimension:
                raise DatabaseError("STORAGE_INCONSISTENT", "Embedding dimension changed", 7)
            self.storage.index._embed_model = self.models.embedding
            self.retrieval = build_retrieval_pipeline(
                settings, self.storage, self.models.llm, self.models.reranker
            )

    def query(
        self,
        text: str,
        *,
        top_k: int | None = None,
        trace: bool = False,
        export: Path | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        identity = self.current()
        request = QueryRequest(query=text, top_k=top_k)
        if export and (not export.is_absolute() or not export.parent.is_dir()):
            raise DatabaseError(
                "EXPORT_PATH_INVALID", "Use an absolute path with an existing parent", 8
            )
        if export and export.exists():
            raise DatabaseError("EXPORT_EXISTS", "Export file already exists", 8)
        settings = replace(
            self.storage.settings,
            query_fusion_num_queries=1,
            llm_model=self.model_name,
        )
        self._ensure_retrieval()
        captures: list[tuple[Any, Any, StageSnapshot]] = []
        if trace:
            for stage, component in zip(
                ("vector", "bm25", "rrf"), self.retrieval._ensure_retrievers(), strict=True
            ):
                snapshot = StageSnapshot(stage)
                previous = component.callback_manager
                component.callback_manager = CallbackManager([*previous.handlers, snapshot])
                captures.append((component, previous, snapshot))
        try:
            retrieval = self.retrieval.retrieve(request)
            generation = StoryGenerationPipeline(settings, self.models.llm, self.retrieval)
            llm_input = generation.prepare_llm_input(request, retrieval)
        finally:
            for component, previous, _ in captures:
                component.callback_manager = previous
        result = {
            **identity,
            "request_id": str(uuid.uuid4()),
            "query": text,
            "status": "context_found" if retrieval.nodes else "no_context",
            "requested_top_k": top_k,
            "effective_top_k": min(top_k or settings.rerank_top_n, settings.rerank_top_n),
            "latency_to_prompt_ms": (time.perf_counter() - start) * 1000,
            "retrieval": retrieval.model_dump(),
            "llm_input": llm_input,
            "llm_calls": 0,
        }
        if trace:
            result["stages"] = {snapshot.stage: snapshot.nodes for _, _, snapshot in captures}
            result["stages"]["rerank"] = [n.model_dump() for n in retrieval.nodes]
            result["stage_metrics"] = {
                snapshot.stage: {"calls": snapshot.count, "elapsed_ms": snapshot.elapsed_ms}
                for _, _, snapshot in captures
            }
            result["stage_metrics"]["rerank"] = {
                "calls": 1 if retrieval.nodes or retrieval.fused_candidates else 0,
                "elapsed_ms": retrieval.rerank_latency_ms,
            }
        if export:
            created = False
            try:
                with export.open("x", encoding="utf-8") as stream:
                    created = True
                    json.dump(result, stream, ensure_ascii=False, indent=2)
            except FileExistsError as exc:
                raise DatabaseError("EXPORT_EXISTS", "Export file already exists", 8) from exc
            except OSError as exc:
                if created:
                    export.unlink(missing_ok=True)
                raise DatabaseError("EXPORT_FAILED", "Could not write query export", 8) from exc
        return result
