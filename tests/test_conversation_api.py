from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from story_rag.api import create_app
from story_rag.conversation_domain import ConversationDisplayPage
from tests.test_task_planning import _case_messages


class _Record:
    def __init__(self, **values):
        self.values = values
        self.revision = values.get("revision", 0)

    def model_dump(self, **_kwargs):
        return self.values


class _Store:
    def __init__(self, session_id):
        self.session_id = session_id

    def create_session(self, name, _binding):
        return _Record(session_id=str(self.session_id), name=name, revision=0)

    def list_sessions(self, _binding):
        return [self.create_session("方案零", _binding)]

    def list_display_turns(self, _session_id, _binding, *, limit, cursor):
        assert limit == 50 and cursor is None
        return ConversationDisplayPage(items=[])

    def get_session(self, _session_id, _binding):
        return self.create_session("方案零", _binding)

    def current_questions_at(self, *_args, **_kwargs):
        return []

    def list_question_versions(self, *_args):
        return []

    def archive_session(self, _session_id, _binding, *, expected_revision):
        return _Record(session_id=str(self.session_id), revision=expected_revision + 1)

    def process_history_index(self, *_args):
        return 0

    def pending_history_index(self, *_args):
        return 0


class _Workflow:
    def __init__(self):
        self.calls = []

    async def answer(self, _session_id, _binding, **kwargs):
        self.calls.append(kwargs)
        return _Record(status="succeeded", revision=1, answer="无", items=[])


def _app():
    database_id = uuid4()
    session_id = uuid4()
    workflow = _Workflow()

    class _Storage:
        database_id = "b" * 64

        async def aclose(self):
            return None

    container = SimpleNamespace(
        settings=SimpleNamespace(debug_enabled=True, registered_db_id=str(database_id)),
        storage=_Storage(), session_store=_Store(session_id),
        conversation_workflow=workflow,
    )
    return create_app(container), session_id, workflow


@pytest.mark.unit
def test_phase0_conversation_http_is_local_only_and_validates_history() -> None:
    app, session_id, workflow = _app()
    message = _case_messages()[0]

    async def exercise() -> None:
        local = httpx.ASGITransport(app=app, client=("127.0.0.1", 1234))
        remote = httpx.ASGITransport(app=app, client=("10.0.0.8", 1234))
        async with httpx.AsyncClient(
            transport=local, base_url="http://127.0.0.1:18082"
        ) as client:
            created = await client.post("/v1/conversations", json={"name": "方案零"})
            assert created.status_code == 200
            listed = await client.get("/v1/conversations")
            assert listed.status_code == 200
            assert listed.json()[0]["session_id"] == str(session_id)
            answer = await client.post(
                f"/v1/conversations/{session_id}/turns",
                json={"request_id": "phase0-http", "expected_revision": 0,
                      "message": message},
            )
            assert answer.status_code == 200
            assert answer.json()["revision"] == 1
            display = await client.get(f"/v1/conversations/{session_id}/turns")
            assert display.status_code == 200
            assert display.json() == {"items": [], "next_cursor": None}
            invalid_page = await client.get(
                f"/v1/conversations/{session_id}/turns?limit=101"
            )
            assert invalid_page.status_code == 422
            invalid = await client.post(
                f"/v1/conversations/{session_id}/turns",
                json={"request_id": "invalid", "expected_revision": 0,
                      "message": message, "use_history": False,
                      "reply_to_question_id": str(uuid4())},
            )
            assert invalid.status_code == 422
            assert len(workflow.calls) == 1
            questions = await client.get(f"/v1/conversations/{session_id}/questions")
            assert questions.status_code == 200
        async with httpx.AsyncClient(
            transport=remote, base_url="http://127.0.0.1:18082"
        ) as client:
            denied = await client.post("/v1/conversations", json={"name": "blocked"})
            assert denied.status_code == 403
            denied_turn = await client.post(
                f"/v1/conversations/{session_id}/turns",
                json={"request_id": "blocked", "expected_revision": 0,
                      "message": message},
            )
            assert denied_turn.status_code == 403
            denied_display = await client.get(f"/v1/conversations/{session_id}/turns")
            assert denied_display.status_code == 403

    asyncio.run(exercise())
