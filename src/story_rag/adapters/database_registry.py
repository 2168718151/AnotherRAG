from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from story_rag.config import Settings
from story_rag.domain import DatabaseError


def normalize_name(name: str) -> str:
    result = unicodedata.normalize("NFC", name).strip()
    if not 1 <= len(result) <= 64 or any(unicodedata.category(c).startswith("C") for c in result):
        raise DatabaseError("INVALID_ARGUMENT", "Database name must be 1-64 visible characters", 2)
    return result


def storage_fingerprint(settings: Settings) -> str:
    scope = [
        settings.qdrant_url.rstrip("/"),
        settings.collection_name,
        str(settings.docstore_path.resolve()).casefold(),
    ]
    return hashlib.sha256(json.dumps(scope).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class FileLease:
    """持有操作系统锁；锁文件保留，避免删除锁文件造成竞争。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream: Any = None

    def acquire(self) -> FileLease:
        if self._stream is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise DatabaseError("DB_BUSY", "Database or registry is in use", 4) from exc
        self._stream = stream
        return self

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> FileLease:
        return self.acquire()

    def __exit__(self, *args: Any) -> None:
        self.close()


class DatabaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    db_id: str
    name: str
    collection_name: str
    docstore_path: str
    qdrant_url: str
    embedding_model: str
    embedding_base_url: str
    embedding_dimension: int = Field(ge=1)
    status: str = "creating"
    legacy: bool = False
    operation: dict[str, Any] | None = None


class DatabaseRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.data_dir / "databases"
        self.path = self.root / "registry.json"

    def lease(self, settings: Settings) -> FileLease:
        return FileLease(self.root / "locks" / f"{storage_fingerprint(settings)}.lock")

    def is_busy(self, record: DatabaseRecord) -> bool:
        lease = self.lease(self.bind(record))
        if not lease.path.exists():
            return False
        try:
            with lease:
                return False
        except DatabaseError as exc:
            if exc.code == "DB_BUSY":
                return True
            raise

    def read(self) -> list[DatabaseRecord]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload["version"] != 1:
                raise ValueError("Unsupported registry version")
            records = [DatabaseRecord.model_validate(item) for item in payload["databases"]]
            ids, scopes, paths, names = set(), set(), set(), set()
            for record in records:
                if str(uuid.UUID(record.db_id)) != record.db_id:
                    raise ValueError("Database UUID must be canonical")
                if record.legacy and record.collection_name != self.settings.collection_name:
                    raise ValueError("Legacy collection differs from the configured scope")
                if record.status not in {"creating", "ready", "deleting", "deleted"}:
                    raise ValueError("Invalid database status")
                expected = (
                    self.settings.docstore_path
                    if record.legacy
                    else self.root / record.db_id.replace("-", "")[:12] / "docstore.json"
                )
                if Path(record.docstore_path).resolve() != expected.resolve():
                    raise ValueError("Database path is outside its registered scope")
                if record.qdrant_url != self.settings.qdrant_url.rstrip("/"):
                    raise ValueError("Database belongs to another Qdrant endpoint")
                if (
                    not record.legacy
                    and record.collection_name != "d_" + record.db_id.replace("-", "")[:12]
                ):
                    raise ValueError("Unexpected collection name")
                scope = (record.qdrant_url, record.collection_name)
                path = str(expected.resolve()).casefold()
                if record.db_id in ids or scope in scopes or path in paths:
                    raise ValueError("Duplicate database storage identity")
                ids.add(record.db_id)
                scopes.add(scope)
                paths.add(path)
                name = normalize_name(record.name).casefold()
                if record.status != "deleted":
                    if name in names:
                        raise ValueError("Duplicate database name")
                    names.add(name)
            return records
        except (ValueError, KeyError, TypeError) as exc:
            raise DatabaseError("STORAGE_INCONSISTENT", "Registry validation failed", 7) from exc

    @contextmanager
    def edit(self) -> Iterator[list[DatabaseRecord]]:
        with FileLease(self.root / "registry.lock"):
            records = self.read()
            yield records
            atomic_json(self.path, {"version": 1, "databases": [r.model_dump() for r in records]})

    def resolve(
        self, name: str | None = None, db_id: str | None = None, *, ready: bool = True
    ) -> DatabaseRecord:
        if bool(name) == bool(db_id):
            raise DatabaseError("INVALID_ARGUMENT", "Specify exactly one of --db and --db-id", 2)
        records = self.read()
        if db_id:
            matches = [r for r in records if r.db_id == db_id]
        else:
            key = normalize_name(name or "").casefold()
            matches = [r for r in records if normalize_name(r.name).casefold() == key]
            active = [r for r in matches if r.status != "deleted"]
            matches = active or matches
        if not matches:
            raise DatabaseError("DB_NOT_FOUND", "Database not found")
        if len(matches) != 1:
            raise DatabaseError("INVALID_ARGUMENT", "Use --db-id for historical duplicate names", 2)
        record = matches[0]
        if ready and record.status != "ready":
            raise DatabaseError("DB_NOT_READY", f"Database state: {record.status}")
        return record

    def bind(self, record: DatabaseRecord) -> Settings:
        return replace(
            self.settings,
            collection_name=record.collection_name,
            docstore_path=Path(record.docstore_path),
            registered_db_id=record.db_id,
            expected_embedding_dimension=record.embedding_dimension,
            allow_storage_create=False,
        )

    def update(self, record: DatabaseRecord) -> None:
        with self.edit() as records:
            for index, current in enumerate(records):
                if current.db_id == record.db_id:
                    records[index] = record
                    return
            raise DatabaseError("DB_NOT_FOUND", "Database record disappeared")

    def check_name(self, records: list[DatabaseRecord], name: str, exclude: str = "") -> None:
        if any(
            r.status != "deleted"
            and r.db_id != exclude
            and normalize_name(r.name).casefold() == name.casefold()
            for r in records
        ):
            raise DatabaseError("DB_NAME_EXISTS", "Database name is already in use")
