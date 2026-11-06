from __future__ import annotations

from types import SimpleNamespace

import pytest
from llama_index.core.embeddings import MockEmbedding

from story_rag.adapters.model_factory import ModelBundle
from story_rag.cli import build_parser, confirmation, execute, main, report_error, run_shell
from story_rag.config import Settings
from story_rag.domain import DatabaseError
from tests.conftest import KeywordReranker, ScriptedLLM
from tests.test_databases import databases as databases
from tests.test_databases import opened, seed, session_for


@pytest.mark.parametrize(
    "words",
    [
        ["db", "create", "--name", "算法面试"],
        ["db", "list", "--all", "--json"],
        ["db", "info", "--db-id", "uuid"],
        ["db", "rename", "--db", "A", "--name", "B", "--dry-run"],
        [
            "documents",
            "replace",
            "--db",
            "A",
            "--document-id",
            "id",
            "--path",
            "D:\\a.txt",
            "--dry-run",
        ],
        ["shell"],
        ["shell", "--db", "算法面试"],
        ["ingest", "--db", "A", "--path", "D:\\qa"],
    ],
)
def test_documented_commands_parse(words):
    assert build_parser().parse_args(words).command == words[0]


@pytest.mark.parametrize(
    "words",
    [
        ["ingest", "--path", "a.txt"],
        ["db", "info"],
        ["db", "create"],
        ["documents", "delete", "--db", "A"],
        ["shell", "--db", "A", "--db-id", "id"],
        ["db", "drop", "--db", "A", "--force"],
        ["query", "question"],
        ["retrieve", "--db", "A"],
    ],
)
def test_unsafe_or_ambiguous_commands_rejected(words):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(words)
    assert error.value.code == 2


def test_management_settings_do_not_require_api_or_reranker(tmp_path, monkeypatch):
    monkeypatch.setenv("RERANKER_MODEL_PATH", str(tmp_path / "missing"))
    configured = Settings.from_env(tmp_path, require_llm=False, require_reranker=False)
    assert not configured.api_file.exists()
    assert configured.llm_api_key == "unused-local-preview"


@pytest.mark.parametrize("action", ["drop", "rename", "delete", "txt_rename", "replace"])
def test_noninteractive_mutations_require_confirmation(databases, monkeypatch, tmp_path, action):
    record, document_id = seed(databases)
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("Q: new\nA: revised", encoding="utf-8")
    if action in {"drop", "rename"}:
        words = ["db", action, "--db", "A"]
        if action == "rename":
            words.extend(["--name", "B"])
    else:
        words = [
            "documents",
            "rename" if action == "txt_rename" else action,
            "--db",
            "A",
            "--document-id",
            document_id,
        ]
        if action == "txt_rename":
            words.extend(["--name", "changed.txt"])
        if action == "replace":
            words.extend(["--path", str(replacement)])
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(DatabaseError) as error:
        execute(build_parser().parse_args(words), databases)
    assert error.value.code == "CONFIRMATION_REQUIRED"
    assert databases.info(record)["document_count"] == 1


def test_document_confirmation_needs_exact_etag(databases):
    record, document_id = seed(databases)
    words = [
        "documents",
        "delete",
        "--db",
        "A",
        "--document-id",
        document_id,
        "--confirm-id",
        record.db_id,
    ]
    with pytest.raises(DatabaseError):
        execute(build_parser().parse_args(words), databases)
    assert databases.info(record)["document_count"] == 1


def test_cli_confirmed_file_delete_uses_selected_library(databases):
    first, document_id = seed(databases, "A")
    second, _ = seed(databases, "B")
    with opened(databases, first) as storage:
        from story_rag.pipelines.documents import DocumentManagementPipeline

        etag = DocumentManagementPipeline(storage).get_document(document_id).etag
    args = build_parser().parse_args(
        [
            "documents",
            "delete",
            "--db",
            "A",
            "--document-id",
            document_id,
            "--confirm-id",
            first.db_id,
            "--if-match",
            etag,
        ]
    )
    assert execute(args, databases).status == "deleted"
    assert databases.info(second)["document_count"] == 1


def test_interactive_cancel_does_not_change_library(databases, monkeypatch):
    record = databases.create("A")
    args = build_parser().parse_args(["db", "drop", "--db", "A"])
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _: "wrong")
    with pytest.raises(DatabaseError) as error:
        confirmation(args, record, databases.info(record))
    assert error.value.code == "CANCELLED"
    assert databases.info(record)["collection_exists"]


def test_shell_query_and_switch_remain_local(databases, monkeypatch, capsys):
    from story_rag import cli_session

    seed(databases, "A")
    seed(databases, "B", "Q: cobalt\nA: cobalt is metal")
    commands = iter(
        [
            'db use "A"',
            'query "amber" --trace',
            'db use "missing"',
            "db current",
            'db use "B"',
            'query "cobalt"',
            "db release",
            "exit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    monkeypatch.setattr(cli_session, "QuerySession", lambda manager: session_for(manager))
    result = run_shell(databases)
    output = capsys.readouterr()
    assert result["status"] == "closed"
    assert '"llm_calls": 0' in output.out and "DB_NOT_FOUND" in output.err
    assert "Amber" in output.out and "cobalt" in output.out
    for record in databases.registry.read():
        with opened(databases, record):
            pass


def test_cli_readonly_list_does_not_initialize_models(databases, monkeypatch, capsys):
    monkeypatch.setattr(Settings, "from_env", lambda *a, **k: databases.settings)
    monkeypatch.setattr(
        "story_rag.pipelines.databases.DatabasesPipeline.embedding",
        lambda *a: (_ for _ in ()).throw(AssertionError("no models")),
    )
    assert main(["db", "list", "--json"]) == 0
    assert capsys.readouterr().out.strip() == "[]"
    assert not databases.registry.path.exists()


def test_api_start_binds_selected_library_and_releases_lock(databases, monkeypatch):
    from story_rag import main as api_main
    from story_rag.adapters import model_factory

    record, _ = seed(databases)
    models = ModelBundle(
        llm=ScriptedLLM(), embedding=MockEmbedding(embed_dim=8), reranker=KeywordReranker(top_n=3)
    )
    monkeypatch.setattr(Settings, "from_env", lambda *a, **k: databases.settings)
    monkeypatch.setattr(model_factory, "model_bundle", lambda _: models)

    def server(app, **kwargs):
        assert app.state.container.settings.registered_db_id == record.db_id
        assert app.state.container.settings.collection_name == record.collection_name
        with pytest.raises(DatabaseError) as error:
            databases.drop(record, confirmation=record.db_id)
        assert error.value.code == "DB_BUSY"

    monkeypatch.setattr(api_main.uvicorn, "run", server)
    assert api_main.main(["--db", "A"]) == 0
    assert models.llm.prompts == []
    with opened(databases, record):
        pass


def test_api_busy_rejected_before_loading_models(databases, monkeypatch):
    from story_rag import main as api_main
    from story_rag.adapters import model_factory

    record = databases.create("A")
    monkeypatch.setattr(Settings, "from_env", lambda *a, **k: databases.settings)
    monkeypatch.setattr(
        model_factory,
        "model_bundle",
        lambda _: (_ for _ in ()).throw(AssertionError("must not load")),
    )
    with opened(databases, record):
        assert api_main.main(["--db", "A"]) == 4


def test_legacy_api_cannot_recreate_deleted_registered_default(container, databases, monkeypatch):
    from story_rag import main as api_main

    container.storage.persist()
    registered = databases.register_default()
    record = databases.resolve(db_id=registered["db_id"])
    databases.drop(record, confirmation=record.db_id)
    monkeypatch.setattr(Settings, "from_env", lambda *a, **k: databases.settings)
    assert api_main.main([]) == 3
    assert not container.storage.client.collection_exists(record.collection_name)


def test_api_configuration_error_returns_summary(monkeypatch, capsys):
    from story_rag import main as api_main
    from story_rag.config import ConfigError

    monkeypatch.setattr(
        Settings, "from_env", lambda *a, **k: (_ for _ in ()).throw(ConfigError("missing config"))
    )
    assert api_main.main([]) == 2
    assert capsys.readouterr().err.strip() == "INVALID_ARGUMENT: missing config"


def test_invalid_document_pagination_has_argument_exit_code(capsys):
    from story_rag.pipelines.documents import DocumentManagementError

    assert report_error(DocumentManagementError(422, "Invalid pagination")) == 2
    assert capsys.readouterr().err.strip() == "INVALID_ARGUMENT: Invalid pagination"


@pytest.mark.parametrize("missing", ["collection", "docstore"])
def test_legacy_registered_storage_is_not_implicitly_recreated(
    container, databases, monkeypatch, missing
):
    from pathlib import Path

    from story_rag import main as api_main
    from story_rag.adapters import model_factory

    container.storage.persist()
    registered = databases.register_default()
    if missing == "collection":
        container.storage.client.delete_collection(registered["collection_name"])
    else:
        Path(registered["docstore_path"]).unlink()
    monkeypatch.setattr(Settings, "from_env", lambda *a, **k: databases.settings)
    monkeypatch.setattr(
        model_factory,
        "model_bundle",
        lambda _: ModelBundle(
            llm=ScriptedLLM(),
            embedding=MockEmbedding(embed_dim=8),
            reranker=KeywordReranker(top_n=3),
        ),
    )
    assert api_main.main([]) == 7
    assert (
        not container.storage.client.collection_exists(registered["collection_name"])
        if missing == "collection"
        else not Path(registered["docstore_path"]).exists()
    )
