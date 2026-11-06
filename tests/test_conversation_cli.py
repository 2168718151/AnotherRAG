from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from story_rag.cli import run_shell
from story_rag.cli_session import PreviewLLM, QuerySession
from story_rag.config import ConfigError, Settings
from story_rag.conversation_domain import DatabaseBinding
from story_rag.domain import NodeTrace, RetrievalResult
from tests.conftest import ScriptedLLM
from tests.test_task_planning import _case_messages


@pytest.mark.unit
def test_phase0_shell_selects_conversation_and_routes_ask(monkeypatch) -> None:
    calls = []

    class FakeSession:
        def __init__(self, _databases):
            self.record = None
            self.selected_conversation = None

        def select(self, name, db_id):
            self.record = SimpleNamespace(name=name or "方案零库", db_id=db_id or "db1")
            self.selected_conversation = None
            return {"name": self.record.name}

        def create_conversation(self, name):
            self.selected_conversation = SimpleNamespace(name=name, revision=0)
            return {"name": name, "session_id": "s1", "revision": 0}

        def ask(self, message, **kwargs):
            calls.append((message, kwargs))
            self.selected_conversation.revision = 1
            return {"status": "succeeded", "revision": 1, "answer": "已回答"}

        def list_questions(self):
            return [{"question_no": 1, "status": "answered"}]

        def archive_conversation(self, session_id):
            self.selected_conversation = None
            return {"session_id": session_id, "archived": True}

        def close(self):
            return None

    monkeypatch.setattr("story_rag.cli_session.QuerySession", FakeSession)
    message = _case_messages()[0]
    commands = iter([
        'db use "方案零库"',
        'conversation create "权限对话"',
        f'ask "{message}" --request-id phase0-cli',
        'question list',
        'conversation archive s1',
        'exit',
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    result = run_shell(SimpleNamespace())

    assert result == {"status": "closed"}
    assert calls == [(message, {
        "request_id": "phase0-cli", "expected_revision": None,
        "question_no": None, "use_history": True,
    })]


@pytest.fixture
def ask_session(tmp_path, monkeypatch):
    binding = DatabaseBinding(database_id=uuid4(), storage_fingerprint="a" * 64)
    settings = Settings(
        project_root=tmp_path,
        api_file=tmp_path / "custom-api.txt",
        deepseek_base_url="https://model.invalid",
        llm_model="preview-model",
        llm_api_key="unused-local-preview",
        registered_db_id=str(binding.database_id),
        collection_name="selected-collection",
        docstore_path=tmp_path / "selected-docstore.json",
        postgres_dsn="postgresql://localhost/test-only",
        postgres_schema="selected_schema",
    )
    session = QuerySession(SimpleNamespace(settings=settings))
    session.storage = SimpleNamespace(settings=settings)
    selected = SimpleNamespace(session_id=uuid4(), revision=0)
    store = SimpleNamespace(get_session=Mock(return_value=selected))
    monkeypatch.setattr(session, "_selected_scope", lambda: (store, binding, selected))
    monkeypatch.setattr(session, "_ensure_retrieval", Mock())
    return session


@pytest.mark.unit
def test_ask_loads_real_config_without_changing_bound_storage(ask_session, monkeypatch):
    from story_rag.adapters import model_factory

    settings = ask_session.storage.settings
    settings.api_file.write_text(
        "deepseek: sk-unit-test-only\nmodel: configured-conversation-model\n",
        encoding="utf-8",
    )
    factory = Mock(return_value=ScriptedLLM())
    monkeypatch.setattr(model_factory, "build_conversation_llm", factory)
    calls = []

    class FakeWorkflow:
        def __init__(self, **components):
            self.components = components

        async def answer(self, session_id, binding, **request):
            calls.append((session_id, binding, request))
            return SimpleNamespace(
                status="succeeded", answer_not_retained=False,
                model_dump=lambda **_: {"status": "succeeded", "answer": "模拟回答"},
            )

    monkeypatch.setattr("story_rag.conversation_workflow.ConversationWorkflow", FakeWorkflow)
    result = ask_session.ask("什么是RAG", request_id="credential-test")
    assert result["answer"] == "模拟回答"
    configured = factory.call_args.args[0]
    assert configured.llm_api_key == "sk-unit-test-only"
    assert configured.llm_model == "configured-conversation-model"
    for field in (
        "api_file", "deepseek_base_url", "registered_db_id", "collection_name",
        "docstore_path", "postgres_dsn", "postgres_schema",
    ):
        assert getattr(configured, field) == getattr(settings, field)
    assert settings.llm_api_key == "unused-local-preview"
    assert calls[0][2]["payload"].message == "什么是RAG"
    ask_session._ensure_retrieval.assert_called_once()

    settings.api_file.unlink()
    ask_session.ask("继续解释", request_id="credential-reuse")
    factory.assert_called_once()
    assert len(calls) == 2


@pytest.mark.unit
@pytest.mark.parametrize("config_text", [None, "model: test-model\n", "sk-unit-test-only\n"])
def test_ask_rejects_missing_config_before_models_or_workflow(
    ask_session, monkeypatch, config_text,
):
    if config_text is not None:
        ask_session.storage.settings.api_file.write_text(config_text, encoding="utf-8")
    factory = Mock()
    monkeypatch.setattr("story_rag.adapters.model_factory.build_conversation_llm", factory)

    with pytest.raises(ConfigError):
        ask_session.ask("什么是RAG")

    factory.assert_not_called()
    ask_session._ensure_retrieval.assert_not_called()
    assert ask_session.conversation_workflow is None


@pytest.mark.unit
def test_query_keeps_zero_generation_without_api_credentials(ask_session, monkeypatch):
    loader = Mock(side_effect=AssertionError("Preview must not load API credentials"))
    monkeypatch.setattr("story_rag.cli_session.load_api_config", loader)
    monkeypatch.setattr(ask_session, "current", lambda: {"name": "预览测试库"})
    ask_session.models = SimpleNamespace(llm=PreviewLLM())
    ask_session.retrieval = SimpleNamespace(retrieve=Mock(return_value=RetrievalResult(
        query="什么是RAG",
        nodes=[NodeTrace(
            node_id="mock-node", text="RAG 是检索增强生成。", retriever="mock", rank=1,
        )],
    )))

    result = ask_session.query("什么是RAG")

    assert not ask_session.storage.settings.api_file.exists()
    loader.assert_not_called()
    assert result["llm_calls"] == 0
    assert result["status"] == "context_found"
    assert result["llm_input"]["would_call_llm"] is True
    assert result["llm_input"]["messages"]
    assert "RAG 是检索增强生成。" in result["llm_input"]["context_str"]
