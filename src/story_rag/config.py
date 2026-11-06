from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_KEY_PATTERN = re.compile(
    r"^\s*(?:deepseek|deepseep|deepseek[_-]?api[_-]?key)\s*[:=]\s*(\S+)\s*$",
    re.IGNORECASE,
)
_MODEL_PATTERN = re.compile(
    r"^\s*(?:模型|model|deepseek[_-]?model)\s*[:：=]\s*(.+?)\s*$",
    re.IGNORECASE,
)


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ApiConfig:
    api_key: str = field(repr=False)
    model: str


def _env_text(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value and value.strip() else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean value")


def load_api_config(path: Path) -> ApiConfig:
    if not path.is_file():
        raise ConfigError(f"API config file does not exist: {path}")

    api_key: str | None = None
    model: str | None = None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key_match = _KEY_PATTERN.match(line)
        if key_match:
            api_key = key_match.group(1).strip().strip("\"'")
            continue

        model_match = _MODEL_PATTERN.match(line)
        if model_match:
            model = model_match.group(1).strip().strip("\"'")
            continue

        stripped = line.strip().strip("\"'")
        if api_key is None and stripped.startswith("sk-"):
            api_key = stripped

    if not api_key:
        raise ConfigError(f"No DeepSeek API key found in {path}")
    if not model:
        raise ConfigError(f"No model name found in {path}")
    return ApiConfig(api_key=api_key, model=model)


def preview_model_name(path: Path, fallback: str) -> str:
    if not path.is_file():
        return fallback
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            match = _MODEL_PATTERN.match(line)
            if match:
                return match.group(1).strip().strip("\"'") or fallback
    return fallback


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    api_file: Path
    deepseek_base_url: str
    llm_model: str
    llm_api_key: str = field(repr=False)
    lmstudio_base_url: str = "http://127.0.0.1:1234/v1"
    lmstudio_api_key: str = field(default="lm-studio", repr=False)
    embedding_model: str = "text-embedding-qwen3-embedding-0.6b"
    reranker_model_path: Path = Path(r"D:\text_Project\Agentic RAG\data\models\Qwen3-Reranker-0.6B")
    data_dir: Path = PROJECT_ROOT / "data"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str | None = field(default=None, repr=False)
    qdrant_timeout_seconds: float = 10.0
    docstore_path: Path = PROJECT_ROOT / "data" / "docstore" / "docstore.json"
    log_dir: Path = PROJECT_ROOT / "logs"
    collection_name: str = "story_chunks"
    chunk_size: int = 512
    chunk_overlap: int = 64
    vector_top_k: int = 30
    bm25_top_k: int = 30
    fusion_top_k: int = 30
    rerank_top_n: int = 5
    query_fusion_num_queries: int = 1
    reranker_max_length: int = 1024
    request_timeout_seconds: float = 120.0
    app_host: str = "127.0.0.1"
    app_port: int = 18082
    debug_enabled: bool = True
    log_level: str = "INFO"
    llama_index_log_level: str = "DEBUG"
    registered_db_id: str | None = None
    expected_embedding_dimension: int | None = None
    allow_storage_create: bool = True
    postgres_dsn: str | None = field(default=None, repr=False)
    postgres_schema: str = "story_rag"

    @classmethod
    def from_env(
        cls,
        project_root: Path | None = None,
        *,
        require_llm: bool = True,
        require_reranker: bool = True,
    ) -> Settings:
        root = (project_root or PROJECT_ROOT).resolve()
        api_file = Path(_env_text("API_FILE", "api.txt"))
        if not api_file.is_absolute():
            api_file = root / api_file

        api_config = load_api_config(api_file) if require_llm else None
        data_dir = root / "data"
        settings = cls(
            project_root=root,
            api_file=api_file,
            deepseek_base_url=_env_text("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            llm_model=api_config.model if api_config else "deepseek-flash",
            llm_api_key=api_config.api_key if api_config else "unused-local-preview",
            lmstudio_base_url=_env_text("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
            lmstudio_api_key=_env_text("LMSTUDIO_API_KEY", "lm-studio"),
            embedding_model=_env_text(
                "LMSTUDIO_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b"
            ),
            reranker_model_path=Path(
                _env_text(
                    "RERANKER_MODEL_PATH",
                    r"D:\text_Project\Agentic RAG\data\models\Qwen3-Reranker-0.6B",
                )
            ),
            data_dir=data_dir,
            qdrant_url=_env_text("QDRANT_URL", "http://127.0.0.1:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            qdrant_timeout_seconds=float(_env_text("QDRANT_TIMEOUT_SECONDS", "10")),
            docstore_path=data_dir / "docstore" / "docstore.json",
            log_dir=root / "logs",
            collection_name=_env_text("QDRANT_COLLECTION", "story_chunks"),
            chunk_size=_env_int("CHUNK_SIZE", 512),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 64),
            vector_top_k=_env_int("VECTOR_TOP_K", 30),
            bm25_top_k=_env_int("BM25_TOP_K", 30),
            fusion_top_k=_env_int("FUSION_TOP_K", 30),
            rerank_top_n=_env_int("RERANK_TOP_N", 5),
            query_fusion_num_queries=_env_int("QUERY_FUSION_NUM_QUERIES", 1),
            reranker_max_length=_env_int("RERANKER_MAX_LENGTH", 1024),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")),
            app_host=_env_text("APP_HOST", "127.0.0.1"),
            app_port=_env_int("APP_PORT", 18082),
            debug_enabled=_env_bool("DEBUG_ENABLED", True),
            log_level=_env_text("LOG_LEVEL", "INFO").upper(),
            llama_index_log_level=_env_text("LLAMA_INDEX_LOG_LEVEL", "DEBUG").upper(),
            postgres_dsn=os.getenv("STORY_RAG_POSTGRES_DSN") or None,
            postgres_schema=_env_text("STORY_RAG_POSTGRES_SCHEMA", "story_rag"),
        )
        _validate_settings(settings, require_reranker=require_reranker)
        return settings

    def debug_snapshot(self) -> dict[str, object]:
        return {
            "project_root": str(self.project_root),
            "api_file": str(self.api_file),
            "deepseek_base_url": self.deepseek_base_url,
            "llm_model": self.llm_model,
            "lmstudio_base_url": self.lmstudio_base_url,
            "embedding_model": self.embedding_model,
            "reranker_model_path": str(self.reranker_model_path),
            "vector_backend": "qdrant",
            "qdrant_url": self.qdrant_url,
            "collection_name": self.collection_name,
            "docstore_path": str(self.docstore_path),
            "log_dir": str(self.log_dir),
            "debug_enabled": self.debug_enabled,
        }


def _validate_settings(settings: Settings, *, require_reranker: bool = True) -> None:
    url = urlsplit(settings.qdrant_url)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ConfigError("QDRANT_URL must be an HTTP(S) URL without embedded credentials")
    if settings.qdrant_timeout_seconds <= 0:
        raise ConfigError("QDRANT_TIMEOUT_SECONDS must be greater than zero")
    if settings.chunk_size <= 0:
        raise ConfigError("CHUNK_SIZE must be greater than zero")
    if settings.chunk_overlap < 0 or settings.chunk_overlap >= settings.chunk_size:
        raise ConfigError("CHUNK_OVERLAP must be in [0, CHUNK_SIZE)")
    if not 1 <= settings.rerank_top_n <= settings.fusion_top_k:
        raise ConfigError("RERANK_TOP_N must be between 1 and FUSION_TOP_K")
    if require_reranker and not settings.reranker_model_path.is_dir():
        raise ConfigError(f"Reranker model path does not exist: {settings.reranker_model_path}")


def ensure_directories(settings: Settings) -> None:
    for path in (
        settings.data_dir,
        settings.docstore_path.parent,
        settings.log_dir,
        settings.log_dir / "traces",
    ):
        path.mkdir(parents=True, exist_ok=True)
