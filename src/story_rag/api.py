from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from ipaddress import ip_address
from typing import Annotated, Any
from uuid import UUID

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from story_rag.adapters.storage import StorageUnavailableError
from story_rag.adapters.text_loader import TextDecodeError
from story_rag.container import ApplicationContainer
from story_rag.conversation_domain import (
    ConversationDisplayPage,
    ConversationTurnInput,
    DatabaseBinding,
    LedgerError,
    RequestPayload,
)
from story_rag.conversation_workflow import ConversationWorkflowError
from story_rag.domain import (
    AnswerResult,
    DebugConfigSnapshot,
    DocumentContentPreview,
    DocumentDeletePreview,
    DocumentDeletionResult,
    DocumentDetail,
    DocumentPage,
    IngestionResult,
    QueryRequest,
    ReadinessResult,
    RetrievalResult,
)
from story_rag.pipelines.conversation_generation import ConversationTaskInput
from story_rag.pipelines.documents import DocumentManagementError
from story_rag.pipelines.task_planning import TaskPlan

logger = logging.getLogger(__name__)


class ConversationCreateInput(BaseModel):
    name: str = Field(min_length=1, max_length=64)


class ConversationArchiveInput(BaseModel):
    expected_revision: int = Field(ge=0)


class ConversationUnknownAckInput(BaseModel):
    acknowledge_result_unknown: bool


class ConversationPreviewInput(BaseModel):
    message: str = Field(min_length=1)
    plan: TaskPlan
    task_inputs: list[ConversationTaskInput]


def create_app(container: ApplicationContainer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await container.storage.aclose()

    app = FastAPI(
        title="Another RAG",
        version="0.1.0",
        docs_url="/docs" if container.settings.debug_enabled else None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.container = container

    @app.exception_handler(DocumentManagementError)
    async def document_error(request: Request, exc: DocumentManagementError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.exception_handler(StorageUnavailableError)
    async def storage_error(request: Request, exc: StorageUnavailableError) -> JSONResponse:
        logger.error("Storage unavailable on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Storage unavailable; check logs and restart to recover pending deletion"
            },
        )

    @app.middleware("http")
    async def request_logging(request: Request, call_next: Any) -> JSONResponse:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        management = request.url.path.startswith(
            (
                "/v1/qa/ingest",
                "/v1/stories/ingest",
                "/v1/documents",
                "/debug/v1/documents",
                "/v1/conversations",
                "/debug/conversations",
            )
        )
        if management and not _local_management_request(request):
            return JSONResponse(status_code=403, content={"detail": "Local management only"})
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_id=%s method=%s path=%s failed",
                request_id,
                request.method,
                request.url.path,
            )
            raise
        response.headers["x-request-id"] = request_id
        if management:
            response.headers["cache-control"] = "no-store"
            response.headers["x-content-type-options"] = "nosniff"
            response.headers["content-security-policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'"
            )
        logger.info(
            "request_id=%s method=%s path=%s status=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
        )
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadinessResult)
    async def ready() -> ReadinessResult | JSONResponse:
        database_ready = await run_in_threadpool(container.storage.is_ready)
        result = ReadinessResult(
            status="ready" if database_ready else "not_ready",
            llm_model=container.settings.llm_model,
            embedding_model=container.settings.embedding_model,
            reranker_model=container.settings.reranker_model_path.name,
            collection_name=container.settings.collection_name,
            debug_enabled=container.settings.debug_enabled,
            vector_backend="qdrant",
            database_ready=database_ready,
        )
        if not database_ready:
            return JSONResponse(status_code=503, content=result.model_dump())
        return result

    @app.post("/v1/stories/ingest", response_model=IngestionResult, include_in_schema=False)
    @app.post("/v1/qa/ingest", response_model=IngestionResult)
    async def ingest_story(
        file: Annotated[UploadFile, File()],
        title: Annotated[str | None, Form()] = None,
        user_instruction: Annotated[str | None, Form()] = None,
    ) -> IngestionResult:
        source_name = file.filename or "untitled.txt"
        if not source_name.casefold().endswith(".txt"):
            raise HTTPException(status_code=415, detail="Only .txt files are accepted")
        data = await file.read()
        if len(data) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="TXT file exceeds 50 MB")
        try:
            return await run_in_threadpool(
                container.ingestion.ingest_bytes,
                data,
                source_name=source_name,
                title=title,
                user_instruction=user_instruction,
            )
        except TextDecodeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Ingestion failed for %s", source_name)
            raise HTTPException(status_code=503, detail="Ingestion service unavailable") from exc

    @app.post("/v1/chat/query", response_model=AnswerResult)
    async def query_story(request: QueryRequest) -> AnswerResult:
        try:
            return await container.workflow.answer(request)
        except Exception as exc:
            logger.exception("Query failed: %r", request.query)
            raise HTTPException(status_code=503, detail="RAG service unavailable") from exc

    @app.post("/v1/conversations")
    async def create_conversation(payload: ConversationCreateInput) -> dict[str, Any]:
        store, binding = _conversation_scope(container)
        try:
            session = await run_in_threadpool(store.create_session, payload.name, binding)
            return session.model_dump(mode="json")
        except LedgerError as exc:
            raise _conversation_http_error(exc) from exc
        except Exception as exc:
            logger.exception("Conversation creation failed")
            raise HTTPException(status_code=503, detail="Conversation storage unavailable") from exc

    @app.get("/v1/conversations")
    async def list_conversations() -> list[dict[str, Any]]:
        store, binding = _conversation_scope(container)
        try:
            sessions = await run_in_threadpool(store.list_sessions, binding)
            return [session.model_dump(mode="json") for session in sessions]
        except LedgerError as exc:
            raise _conversation_http_error(exc) from exc
        except Exception as exc:
            logger.exception("Conversation listing failed")
            raise HTTPException(status_code=503, detail="Conversation storage unavailable") from exc

    @app.get("/v1/conversations/{session_id}/turns", response_model=ConversationDisplayPage)
    async def list_conversation_turns(
        session_id: UUID,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        cursor: UUID | None = None,
    ) -> ConversationDisplayPage:
        store, binding = _conversation_scope(container)
        try:
            return await run_in_threadpool(
                store.list_display_turns, session_id, binding, limit=limit, cursor=cursor
            )
        except LedgerError as exc:
            raise _conversation_http_error(exc) from exc
        except Exception as exc:
            logger.exception("Conversation display listing failed")
            raise HTTPException(status_code=503, detail="Conversation storage unavailable") from exc

    @app.post("/v1/conversations/{session_id}/turns")
    async def answer_conversation_turn(
        session_id: UUID, payload: ConversationTurnInput
    ) -> dict[str, Any]:
        workflow = container.conversation_workflow
        if workflow is None:
            raise HTTPException(status_code=503, detail="Conversation workflow is not configured")
        store, binding = _conversation_scope(container)
        try:
            result = await workflow.answer(
                session_id,
                binding,
                request_id=payload.request_id,
                payload=RequestPayload(
                    message=payload.message,
                    reply_to_question_id=payload.reply_to_question_id,
                    use_history=payload.use_history,
                ),
                expected_revision=payload.expected_revision,
            )
            response = result.model_dump(mode="json")
            if response.get("status") == "succeeded" and response.get("answer_not_retained"):
                answer = await run_in_threadpool(
                    store.get_display_answer, session_id, binding, result.request.turn_id
                )
                if answer is not None:
                    response["answer"] = answer
                    response["answer_not_retained"] = False
            return response
        except (LedgerError, ConversationWorkflowError) as exc:
            raise _conversation_http_error(exc) from exc
        except Exception as exc:
            logger.exception("Conversation turn failed")
            raise HTTPException(
                status_code=503, detail="Conversation workflow unavailable"
            ) from exc

    @app.get("/v1/conversations/{session_id}/questions")
    async def conversation_questions(session_id: UUID) -> list[dict[str, Any]]:
        store, binding = _conversation_scope(container)
        try:
            session = await run_in_threadpool(store.get_session, session_id, binding)
            questions = await run_in_threadpool(
                store.current_questions_at,
                session_id,
                binding,
                as_of_revision=session.revision,
            )
            return [item.model_dump(mode="json") for item in questions]
        except LedgerError as exc:
            raise _conversation_http_error(exc) from exc

    @app.get("/v1/conversations/{session_id}/questions/{question_id}/versions")
    async def conversation_question_versions(
        session_id: UUID, question_id: UUID
    ) -> list[dict[str, Any]]:
        store, binding = _conversation_scope(container)
        try:
            versions = await run_in_threadpool(
                store.list_question_versions, session_id, binding, question_id
            )
            return [item.model_dump(mode="json") for item in versions]
        except LedgerError as exc:
            raise _conversation_http_error(exc) from exc

    @app.post("/v1/conversations/{session_id}/archive")
    async def archive_conversation(
        session_id: UUID, payload: ConversationArchiveInput
    ) -> dict[str, Any]:
        store, binding = _conversation_scope(container)
        try:
            archived = await run_in_threadpool(
                store.archive_session,
                session_id,
                binding,
                expected_revision=payload.expected_revision,
            )
            return archived.model_dump(mode="json")
        except LedgerError as exc:
            raise _conversation_http_error(exc) from exc

    @app.post("/v1/conversations/{session_id}/index/recover")
    async def recover_conversation_index(session_id: UUID) -> dict[str, int]:
        store, binding = _conversation_scope(container)
        try:
            processed = await run_in_threadpool(store.process_history_index, session_id, binding)
            pending = await run_in_threadpool(store.pending_history_index, session_id, binding)
            return {"processed": processed, "pending": pending}
        except LedgerError as exc:
            raise _conversation_http_error(exc) from exc

    @app.post("/v1/conversations/{session_id}/requests/{request_id}/recover-orphan")
    async def recover_orphaned_request(session_id: UUID, request_id: str) -> dict[str, Any]:
        store, binding = _conversation_scope(container)
        try:
            request = await run_in_threadpool(
                store.recover_orphaned_request, session_id, binding, request_id
            )
            return request.model_dump(mode="json")
        except LedgerError as exc:
            raise _conversation_http_error(exc) from exc

    @app.post("/v1/conversations/{session_id}/requests/{request_id}/acknowledge-unknown")
    async def acknowledge_unknown_request(
        session_id: UUID, request_id: str, payload: ConversationUnknownAckInput
    ) -> dict[str, Any]:
        store, binding = _conversation_scope(container)
        try:
            request = await run_in_threadpool(
                store.acknowledge_unknown_request,
                session_id,
                binding,
                request_id,
                acknowledge_result_unknown=payload.acknowledge_result_unknown,
            )
            return request.model_dump(mode="json")
        except LedgerError as exc:
            raise _conversation_http_error(exc) from exc

    if container.settings.debug_enabled:
        _register_debug_routes(app, container)

    _register_document_routes(app, container)

    return app


def _conversation_scope(container: ApplicationContainer):
    if container.session_store is None or not container.settings.registered_db_id:
        raise HTTPException(status_code=503, detail="Conversation storage is not configured")
    try:
        database_id = uuid.UUID(container.settings.registered_db_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=503, detail="Conversation database binding is invalid"
        ) from exc
    return container.session_store, DatabaseBinding(
        database_id=database_id,
        storage_fingerprint=container.storage.database_id,
    )


def _conversation_http_error(exc: LedgerError | ConversationWorkflowError) -> HTTPException:
    if exc.code in {"SESSION_NOT_FOUND", "REQUEST_NOT_FOUND"}:
        status = 404
    elif exc.code in {
        "REVISION_CONFLICT",
        "IDEMPOTENCY_CONFLICT",
        "SESSION_BUSY",
        "REQUEST_IN_PROGRESS",
        "SCOPE_MISMATCH",
        "SESSION_ARCHIVED",
        "SESSION_UNCERTAIN",
        "REQUEST_NOT_RECOVERABLE",
    }:
        status = 409
    else:
        status = 400
    return HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)})


def _local_management_request(request: Request) -> bool:
    if request.url.hostname not in {"127.0.0.1", "localhost", "::1"} or not request.client:
        return False
    try:
        if not ip_address(request.client.host).is_loopback:
            return False
    except ValueError:
        return False
    origin = request.headers.get("origin")
    return not origin or origin == str(request.base_url).rstrip("/")


def _register_document_routes(app: FastAPI, container: ApplicationContainer) -> None:
    @app.get("/v1/documents", response_model=DocumentPage)
    async def list_documents(
        q: Annotated[str, Query(max_length=500)] = "",
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> DocumentPage:
        try:
            return await run_in_threadpool(container.documents.list_documents, q, offset, limit)
        except (DocumentManagementError, StorageUnavailableError):
            raise
        except Exception as exc:
            logger.exception("Document listing failed")
            raise StorageUnavailableError("Document listing failed") from exc

    @app.get("/v1/documents/{document_id}/preview", response_model=DocumentContentPreview)
    async def preview_document(
        document_id: str,
        response: Response,
        max_characters: Annotated[int, Query(ge=1, le=20000)] = 8000,
    ) -> DocumentContentPreview:
        try:
            result = await run_in_threadpool(
                container.documents.preview_document, document_id, max_characters
            )
        except (DocumentManagementError, StorageUnavailableError):
            raise
        except Exception as exc:
            logger.exception("Document content preview failed: %s", document_id)
            raise StorageUnavailableError("Document preview failed") from exc
        response.headers["ETag"] = result.etag
        return result

    @app.get("/v1/documents/{document_id}", response_model=DocumentDetail)
    async def get_document(document_id: str, response: Response) -> DocumentDetail:
        try:
            result = await run_in_threadpool(container.documents.get_document, document_id)
        except (DocumentManagementError, StorageUnavailableError):
            raise
        except Exception as exc:
            logger.exception("Document detail failed: %s", document_id)
            raise StorageUnavailableError("Document detail failed") from exc
        response.headers["ETag"] = result.etag
        return result

    @app.delete("/v1/documents/{document_id}", response_model=DocumentDeletionResult)
    async def delete_document(
        document_id: str,
        database_id: Annotated[str | None, Header(alias="X-Database-ID")] = None,
        if_match: Annotated[str | None, Header()] = None,
    ) -> DocumentDeletionResult:
        if not database_id or not if_match:
            raise HTTPException(status_code=428, detail="X-Database-ID and If-Match are required")
        try:
            return await run_in_threadpool(
                container.documents.delete_document,
                document_id,
                database_id=database_id,
                if_match=if_match,
            )
        except (DocumentManagementError, StorageUnavailableError):
            raise
        except Exception as exc:
            logger.exception("Document deletion failed: %s", document_id)
            raise StorageUnavailableError("Document deletion failed") from exc

    if container.settings.debug_enabled:

        @app.get(
            "/debug/v1/documents/{document_id}/delete-preview",
            response_model=DocumentDeletePreview,
        )
        async def delete_preview(document_id: str) -> DocumentDeletePreview:
            try:
                return await run_in_threadpool(container.documents.delete_preview, document_id)
            except (DocumentManagementError, StorageUnavailableError):
                raise
            except Exception as exc:
                logger.exception("Document deletion preview failed: %s", document_id)
                raise StorageUnavailableError("Document preview failed") from exc


def _register_debug_routes(app: FastAPI, container: ApplicationContainer) -> None:
    @app.post("/debug/conversations/{session_id}/preview")
    async def conversation_preview(
        session_id: UUID, payload: ConversationPreviewInput
    ) -> dict[str, Any]:
        if container.conversation_workflow is None:
            raise HTTPException(status_code=503, detail="Conversation workflow is not configured")
        store, binding = _conversation_scope(container)
        try:
            await run_in_threadpool(store.get_session, session_id, binding)
            return await run_in_threadpool(
                container.conversation_workflow.preview_turn,
                payload=RequestPayload(message=payload.message),
                plan=payload.plan,
                task_inputs=payload.task_inputs,
            )
        except LedgerError as exc:
            raise _conversation_http_error(exc) from exc

    @app.get("/debug/v1/config", response_model=DebugConfigSnapshot)
    async def debug_config() -> DebugConfigSnapshot:
        return DebugConfigSnapshot(**container.settings.debug_snapshot())

    @app.post("/debug/v1/retrieval/trace", response_model=RetrievalResult)
    async def retrieval_trace(request: QueryRequest) -> RetrievalResult:
        return await run_in_threadpool(container.retrieval.retrieve_trace, request)

    @app.post("/debug/v1/generation/trace")
    async def generation_trace(request: QueryRequest) -> dict[str, Any]:
        return await run_in_threadpool(container.generation.answer_trace, request)

    @app.post("/debug/v1/generation/input")
    async def generation_input(request: QueryRequest) -> dict[str, Any]:
        if container.settings.query_fusion_num_queries != 1:
            raise HTTPException(
                status_code=409,
                detail="LLM input preview requires QUERY_FUSION_NUM_QUERIES=1",
            )
        return await run_in_threadpool(container.generation.prepare_llm_input, request)

    @app.post("/debug/v1/ingestion/inspect")
    async def ingestion_inspect(request: QueryRequest) -> dict[str, Any]:
        return await run_in_threadpool(
            container.ingestion.inspect_bytes,
            request.query.encode("utf-8"),
            source_name="debug-input.txt",
        )

    @app.post("/debug/v1/workflow/trace")
    async def workflow_trace(request: QueryRequest) -> dict[str, Any]:
        events = await container.workflow.stream_trace(request)
        return {"events": events}
