from __future__ import annotations

from dataclasses import replace

import pytest

from story_rag.config import ApiConfig, ConfigError, Settings, _validate_settings, load_api_config
from tests.conftest import FAKE_API_KEY


def test_load_api_config_reads_key_and_model(tmp_path) -> None:
    api_file = tmp_path / "api.txt"
    api_file.write_text(f"{FAKE_API_KEY}\n模型:  deepseek-flash\n", encoding="utf-8")

    config = load_api_config(api_file)

    assert config == ApiConfig(api_key=FAKE_API_KEY, model="deepseek-flash")


def test_load_api_config_supports_explicit_fields(tmp_path) -> None:
    api_file = tmp_path / "api.txt"
    api_file.write_text(
        f"deepseek_api_key: {FAKE_API_KEY}\nmodel = deepseek-chat\n",
        encoding="utf-8",
    )

    config = load_api_config(api_file)

    assert config.api_key == FAKE_API_KEY
    assert config.model == "deepseek-chat"


def test_load_api_config_fails_without_key(tmp_path) -> None:
    api_file = tmp_path / "api.txt"
    api_file.write_text("模型: deepseek-flash\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_api_config(api_file)


def test_api_key_never_appears_in_repr_or_debug_snapshot(settings: Settings) -> None:
    assert FAKE_API_KEY not in repr(settings)
    snapshot = settings.debug_snapshot()
    assert FAKE_API_KEY not in str(snapshot)
    assert "api_key" not in snapshot
    assert snapshot["deepseek_base_url"] == "https://api.deepseek.com"
    assert snapshot["llm_model"] == "deepseek-flash"


def test_settings_from_env_uses_api_file(tmp_path, monkeypatch) -> None:
    api_file = tmp_path / "api.txt"
    api_file.write_text(f"{FAKE_API_KEY}\n模型: deepseek-flash\n", encoding="utf-8")
    monkeypatch.setenv("API_FILE", str(api_file))
    monkeypatch.setenv("CHUNK_SIZE", "128")
    monkeypatch.setenv("CHUNK_OVERLAP", "16")
    monkeypatch.setenv("RERANK_TOP_N", "2")
    monkeypatch.setenv("RERANKER_MODEL_PATH", str(tmp_path))

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.llm_model == "deepseek-flash"
    assert settings.chunk_size == 128
    assert settings.rerank_top_n == 2
    assert settings.log_dir == tmp_path / "logs"


def test_qdrant_config_is_read_at_startup(tmp_path, monkeypatch) -> None:
    (tmp_path / "api.txt").write_text(f"{FAKE_API_KEY}\nmodel: deepseek-flash\n")
    monkeypatch.setenv("API_FILE", str(tmp_path / "api.txt"))
    monkeypatch.setenv("RERANKER_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:16333")
    monkeypatch.setenv("QDRANT_COLLECTION", "qa_config_test")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-secret-placeholder")

    settings = Settings.from_env(tmp_path)

    assert settings.qdrant_url == "http://127.0.0.1:16333"
    assert settings.collection_name == "qa_config_test"
    assert settings.qdrant_api_key == "qdrant-secret-placeholder"
    assert "qdrant-secret-placeholder" not in repr(settings)
    assert "qdrant-secret-placeholder" not in str(settings.debug_snapshot())


def test_postgres_dsn_is_not_exposed_in_config_snapshot(tmp_path, monkeypatch) -> None:
    (tmp_path / "api.txt").write_text(f"{FAKE_API_KEY}\nmodel: deepseek-flash\n")
    monkeypatch.setenv("API_FILE", str(tmp_path / "api.txt"))
    monkeypatch.setenv("RERANKER_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("STORY_RAG_POSTGRES_DSN", "postgresql://user:secret@localhost/test")
    monkeypatch.setenv("STORY_RAG_POSTGRES_SCHEMA", "conversation_test")

    settings = Settings.from_env(tmp_path)

    assert settings.postgres_schema == "conversation_test"
    assert "secret" not in repr(settings)
    assert "secret" not in str(settings.debug_snapshot())


@pytest.mark.parametrize(
    "url",
    ["file:///tmp/db", "http://user:secret@localhost:6333", "http://localhost:6333?api_key=secret"],
)
def test_qdrant_url_rejects_embedded_credentials(settings, url) -> None:
    with pytest.raises(ConfigError, match="QDRANT_URL"):
        _validate_settings(replace(settings, qdrant_url=url))
