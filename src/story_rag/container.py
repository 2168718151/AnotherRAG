from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from story_rag.adapters.model_factory import ModelBundle, model_bundle
from story_rag.adapters.session_store import build_session_store
from story_rag.adapters.storage import StorageBundle, build_storage
from story_rag.config import Settings, ensure_directories
from story_rag.conversation_workflow import ConversationWorkflow
from story_rag.logging_config import TraceWriter, configure_logging
from story_rag.pipelines.conversation_generation import ConversationAnswerPipeline
from story_rag.pipelines.documents import DocumentManagementPipeline
from story_rag.pipelines.evidence_mapping import EvidenceMappingPipeline
from story_rag.pipelines.generation import StoryGenerationPipeline
from story_rag.pipelines.history_resolution import HistoryResolutionPipeline
from story_rag.pipelines.ingestion import StoryIngestionPipeline
from story_rag.pipelines.retrieval import build_retrieval_pipeline
from story_rag.pipelines.task_planning import TaskPlanningPipeline
from story_rag.pipelines.token_budget import TokenBudget
from story_rag.workflow import StoryRagWorkflow


@dataclass(slots=True)
class ApplicationContainer:
    settings: Settings
    llm: Any
    storage: StorageBundle
    ingestion: Any
    retrieval: Any
    generation: Any
    workflow: Any
    trace_writer: TraceWriter
    documents: DocumentManagementPipeline
    task_planner: TaskPlanningPipeline
    history_resolver: HistoryResolutionPipeline | None
    evidence_mapper: EvidenceMappingPipeline | None
    session_store: Any | None
    conversation_generation: ConversationAnswerPipeline | None
    conversation_workflow: ConversationWorkflow | None

    @property
    def ready(self) -> bool:
        return self.storage.is_ready()


def build_container(
    settings: Settings,
    *,
    models: ModelBundle | None = None,
    storage: StorageBundle | None = None,
) -> ApplicationContainer:
    ensure_directories(settings)
    configure_logging(settings)
    trace_writer = TraceWriter(settings.log_dir)

    bundle = models or model_bundle(settings)
    llm = bundle.llm
    conversation_llm = bundle.conversation_llm or llm
    embedding = bundle.embedding
    reranker = bundle.reranker
    storage = storage or build_storage(settings, embedding)

    ingestion = StoryIngestionPipeline(settings, storage, trace_writer)
    retrieval = build_retrieval_pipeline(settings, storage, llm, reranker, trace_writer)
    generation = StoryGenerationPipeline(settings, llm, retrieval, trace_writer)
    workflow = StoryRagWorkflow(
        retrieval_pipeline=retrieval,
        generation_pipeline=generation,
        trace_writer=trace_writer,
        timeout=settings.request_timeout_seconds,
    )
    task_planner = TaskPlanningPipeline(conversation_llm, trace_writer)
    session_store = build_session_store(settings) if settings.postgres_dsn else None
    history_resolver = HistoryResolutionPipeline(session_store) if session_store else None
    evidence_mapper = (
        EvidenceMappingPipeline(session_store, storage, retrieval)
        if session_store and settings.registered_db_id else None
    )
    conversation_generation = ConversationAnswerPipeline(conversation_llm, TokenBudget())
    conversation_workflow = (
        ConversationWorkflow(
            store=session_store,
            planner=task_planner,
            history=history_resolver,
            evidence=evidence_mapper,
            generator=conversation_generation,
            diagnostic_directory=settings.project_root / "reports" / "conversation_diagnostics",
        )
        if session_store and history_resolver and evidence_mapper
        else None
    )
    return ApplicationContainer(
        settings=settings,
        llm=llm,
        storage=storage,
        ingestion=ingestion,
        retrieval=retrieval,
        generation=generation,
        workflow=workflow,
        trace_writer=trace_writer,
        documents=DocumentManagementPipeline(storage),
        task_planner=task_planner,
        history_resolver=history_resolver,
        evidence_mapper=evidence_mapper,
        session_store=session_store,
        conversation_generation=conversation_generation,
        conversation_workflow=conversation_workflow,
    )
