from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from story_rag.config import ConfigError, Settings
from story_rag.domain import DatabaseError


def selector(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("--db")
    group.add_argument("--db-id")


def mutation_options(parser: argparse.ArgumentParser, *, document: bool = False) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-id")
    if document:
        parser.add_argument("--if-match")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="story-rag")
    commands = parser.add_subparsers(dest="command", required=True)
    db = commands.add_parser("db").add_subparsers(dest="action", required=True)
    create = db.add_parser("create")
    create.add_argument("--name", required=True)
    listing = db.add_parser("list")
    listing.add_argument("--all", action="store_true")
    listing.add_argument("--json", action="store_true")
    info = db.add_parser("info")
    selector(info)
    info.add_argument("--json", action="store_true")
    register = db.add_parser("register-default")
    register.add_argument("--dry-run", action="store_true")
    for action in ("rename", "drop", "recover"):
        command = db.add_parser(action)
        selector(command)
        mutation_options(command)
        if action == "rename":
            command.add_argument("--name", required=True)
        if action == "recover":
            command.add_argument("--operation-id")
    docs = commands.add_parser("documents").add_subparsers(dest="action", required=True)
    for action in ("list", "info", "delete", "rename", "replace"):
        command = docs.add_parser(action)
        selector(command)
        if action != "list":
            command.add_argument("--document-id", required=True)
        if action in {"list", "info"}:
            command.add_argument("--json", action="store_true")
        else:
            mutation_options(command, document=True)
        if action == "list":
            command.add_argument("--q", default="")
            command.add_argument("--offset", default=0, type=int)
            command.add_argument("--limit", default=50, type=int)
        if action == "rename":
            command.add_argument("--name", required=True)
        if action == "replace":
            command.add_argument("--path", required=True, type=Path)
    ingest = commands.add_parser("ingest")
    selector(ingest)
    ingest.add_argument("--path", required=True, type=Path)
    ingest.add_argument("--title")
    shell = commands.add_parser("shell")
    selector(shell, required=False)
    return parser


def emit(value: Any, *, compact: bool = False) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    print(json.dumps(value, ensure_ascii=False, indent=None if compact else 2, default=str))


def confirmation(args, record, preview, *, document: bool = False) -> tuple[str, str | None]:
    if args.confirm_id:
        if args.confirm_id != record.db_id:
            raise DatabaseError("DB_ID_MISMATCH", "Database confirmation differs", 5)
        if document and not args.if_match:
            raise DatabaseError("CONFIRMATION_REQUIRED", "Provide --if-match from the preview", 5)
        if args.action == "recover" and not args.operation_id:
            raise DatabaseError(
                "CONFIRMATION_REQUIRED", "Provide --operation-id from the preview", 5
            )
        return args.confirm_id, getattr(args, "if_match", None)
    if not sys.stdin.isatty():
        raise DatabaseError(
            "CONFIRMATION_REQUIRED", "Noninteractive changes require --confirm-id", 5
        )
    emit(preview)
    if args.action == "drop" and input("输入完整库名确认整库删除: ") != record.name:
        raise DatabaseError("CANCELLED", "Database deletion cancelled", 9)
    if input("输入完整数据库 UUID: ") != record.db_id:
        raise DatabaseError("CANCELLED", "Operation cancelled", 9)
    if document:
        if input("输入目标 Document ID: ") != args.document_id:
            raise DatabaseError("CANCELLED", "Document change cancelled", 9)
        if (
            args.action in {"rename", "replace"}
            and input("输入当前 ETag（包括双引号）: ") != preview["etag"]
        ):
            raise DatabaseError("CANCELLED", "Version confirmation cancelled", 9)
    if args.action == "recover":
        operation = preview.get("operation")
        if operation and input("输入操作 ID: ") != operation["operation_id"]:
            raise DatabaseError("CANCELLED", "Recovery cancelled", 9)
        args.operation_id = operation["operation_id"] if operation else ""
    return record.db_id, preview.get("etag")


def execute(args, databases) -> Any:
    if args.command == "db":
        if args.action == "create":
            return databases.create(args.name)
        if args.action == "list":
            return databases.list_databases(args.all)
        if args.action == "register-default":
            return databases.register_default(dry_run=args.dry_run)
        record = databases.resolve(
            args.db, args.db_id, ready=args.action not in {"info", "recover"}
        )
        if args.action == "info":
            return databases.info(record)
        if args.action == "rename":
            preview = databases.rename(record, args.name, confirmation="", dry_run=True)
        elif args.action == "drop":
            preview = databases.drop(record, confirmation="", dry_run=True)
        else:
            preview = databases.recover(record, dry_run=True)
        if args.dry_run:
            return preview
        confirm, _ = confirmation(args, record, preview)
        if args.action == "rename":
            return databases.rename(record, args.name, confirmation=confirm)
        if args.action == "drop":
            return databases.drop(record, confirmation=confirm)
        return databases.recover(record, confirmation=confirm, operation_id=args.operation_id or "")
    if args.command == "shell":
        return run_shell(databases, args.db, args.db_id)
    record = databases.resolve(args.db, args.db_id)
    embedding = databases.embedding() if args.command == "ingest" else None
    storage = databases.open(record, embedding=embedding, probe=args.command == "ingest")
    try:
        from story_rag.pipelines.documents import DocumentManagementPipeline
        from story_rag.pipelines.ingestion import StoryIngestionPipeline

        documents = DocumentManagementPipeline(storage)
        if args.command == "ingest":
            ingestion = StoryIngestionPipeline(storage.settings, storage)
            if args.path.is_dir():
                if args.title:
                    raise DatabaseError("INVALID_ARGUMENT", "--title is only for a single TXT", 2)
                return ingestion.ingest_directory(args.path)
            if args.path.suffix.casefold() != ".txt":
                raise DatabaseError("INVALID_ARGUMENT", "Input must be TXT", 2)
            return ingestion.ingest_bytes(
                args.path.read_bytes(), source_name=str(args.path.resolve()), title=args.title
            )
        if args.action == "list":
            return documents.list_documents(args.q, args.offset, args.limit)
        if args.action == "info":
            return documents.get_document(args.document_id)
        if args.action == "rename":
            preview = documents.rename_preview(args.document_id, args.name)
        elif args.action == "replace":
            preview = documents.replace_preview(args.document_id, args.path)
        else:
            preview = documents.delete_preview(args.document_id).model_dump()
        preview["db_id"] = record.db_id
        if args.dry_run:
            return preview
        confirm, etag = confirmation(args, record, preview, document=True)
        databases.confirm(record, confirm)
        kwargs = {"database_id": storage.database_id, "if_match": etag}
        if args.action == "rename":
            return documents.rename_document(args.document_id, args.name, **kwargs)
        if args.action == "replace":
            storage.index._embed_model = databases.embedding()
            return documents.replace_document(args.document_id, args.path, **kwargs)
        return documents.delete_document(args.document_id, **kwargs)
    finally:
        asyncio.run(storage.aclose())


def run_shell(databases, name=None, db_id=None) -> dict[str, str]:
    from story_rag.cli_session import QuerySession, split_command

    session = QuerySession(databases)
    parser = build_parser()
    use = argparse.ArgumentParser(prog="db use")
    group = use.add_mutually_exclusive_group(required=True)
    group.add_argument("name", nargs="?")
    group.add_argument("--db-id")
    query = argparse.ArgumentParser(prog="query")
    query.add_argument("query")
    query.add_argument("--top-k", type=int)
    query.add_argument("--trace", action="store_true")
    query.add_argument("--export", type=Path)
    conversation_create = argparse.ArgumentParser(prog="conversation create")
    conversation_create.add_argument("name")
    conversation_use = argparse.ArgumentParser(prog="conversation use")
    conversation_use.add_argument("session_id")
    conversation_archive = argparse.ArgumentParser(prog="conversation archive")
    conversation_archive.add_argument("session_id")
    conversation_recover = argparse.ArgumentParser(prog="conversation recover")
    conversation_recover.add_argument("request_id")
    recovery_mode = conversation_recover.add_mutually_exclusive_group(required=True)
    recovery_mode.add_argument("--acknowledge-unknown", action="store_true")
    recovery_mode.add_argument("--orphan-to-unknown", action="store_true")
    ask = argparse.ArgumentParser(prog="ask")
    ask.add_argument("message")
    ask.add_argument("--request-id")
    ask.add_argument("--expected-revision", type=int)
    ask.add_argument("--question")
    ask.add_argument("--no-history", action="store_true")
    question = argparse.ArgumentParser(prog="question")
    question.add_argument("number")
    try:
        if name or db_id:
            emit(session.select(name, db_id))
        while True:
            try:
                if session.record and session.selected_conversation:
                    selected = session.selected_conversation
                    prompt = (
                        f"story-rag[{session.record.name}|{selected.name}|r{selected.revision}]> "
                    )
                else:
                    prompt = f"[{session.record.name}]> " if session.record else "story-rag> "
                words = split_command(input(prompt))
                if not words:
                    continue
                if words == ["exit"]:
                    break
                if words == ["help"]:
                    print(
                        'db list | db use "名称" | db current | db release | documents list/info | '
                        'query "问题" [--top-k N] [--trace] [--export PATH] | '
                        "conversation create/list/use/archive/index recover/recover | "
                        'question list/show/history | ask "问题" [--question Q001] '
                        "[--no-history] [--request-id ID] | exit"
                    )
                elif words == ["db", "list"]:
                    emit(databases.list_databases())
                elif words[:2] == ["db", "use"]:
                    selected = use.parse_args(words[2:])
                    emit(session.select(selected.name, selected.db_id))
                elif words == ["db", "current"]:
                    emit(session.current())
                elif words == ["db", "release"]:
                    emit(session.release())
                elif words[0] == "query":
                    request = query.parse_args(words[1:])
                    emit(
                        session.query(
                            request.query,
                            top_k=request.top_k,
                            trace=request.trace,
                            export=request.export,
                        )
                    )
                elif words[:2] == ["conversation", "create"]:
                    args = conversation_create.parse_args(words[2:])
                    emit(session.create_conversation(args.name))
                elif words == ["conversation", "list"]:
                    emit(session.list_conversations())
                elif words[:2] == ["conversation", "use"]:
                    args = conversation_use.parse_args(words[2:])
                    emit(session.use_conversation(args.session_id))
                elif words[:2] == ["conversation", "archive"]:
                    args = conversation_archive.parse_args(words[2:])
                    emit(session.archive_conversation(args.session_id))
                elif words == ["conversation", "index", "recover"]:
                    emit(session.recover_conversation_index())
                elif words[:2] == ["conversation", "recover"]:
                    args = conversation_recover.parse_args(words[2:])
                    if args.orphan_to_unknown:
                        emit(session.recover_orphaned_request(args.request_id))
                        continue
                    if not args.acknowledge_unknown:
                        raise DatabaseError(
                            "CONFIRMATION_REQUIRED",
                            "Use --acknowledge-unknown after checking the remote outcome",
                            5,
                        )
                    emit(session.acknowledge_unknown_request(args.request_id))
                elif words == ["question", "list"]:
                    emit(session.list_questions())
                elif words[:2] in (["question", "show"], ["question", "history"]):
                    args = question.parse_args(words[2:])
                    number = _question_no(args.number)
                    versions = session.question_versions(number)
                    emit(versions[-1] if words[1] == "show" else versions)
                elif words[0] == "ask":
                    args = ask.parse_args(words[1:])
                    emit(
                        session.ask(
                            args.message,
                            request_id=args.request_id,
                            expected_revision=args.expected_revision,
                            question_no=_question_no(args.question) if args.question else None,
                            use_history=not args.no_history,
                        )
                    )
                elif words[:2] in (["documents", "list"], ["documents", "info"]):
                    session.current()
                    if "--db" in words or "--db-id" in words:
                        raise DatabaseError(
                            "INVALID_ARGUMENT", "Use db use to switch the session database", 2
                        )
                    command = parser.parse_args([*words, "--db-id", session.record.db_id])
                    docs = session.documents()
                    emit(
                        docs.list_documents(command.q, command.offset, command.limit)
                        if command.action == "list"
                        else docs.get_document(command.document_id)
                    )
                else:
                    raise DatabaseError(
                        "INVALID_ARGUMENT", "Unsupported session command; use help", 2
                    )
            except EOFError:
                break
            except (KeyboardInterrupt, SystemExit):
                print("CANCELLED: 当前输入或查询已取消")
            except Exception as exc:
                report_error(exc)
        return {"status": "closed"}
    finally:
        session.close()


def _question_no(value: str) -> int:
    if value.upper().startswith("Q"):
        value = value[1:]
    if not value.isdecimal() or int(value) < 1:
        raise DatabaseError("INVALID_ARGUMENT", "Use a positive question number such as Q001", 2)
    return int(value)


def report_error(exc: Exception) -> int:
    from story_rag.adapters.storage import StorageUnavailableError
    from story_rag.pipelines.documents import DocumentManagementError

    if isinstance(exc, DatabaseError):
        code, message, exit_code = exc.code, str(exc), exc.exit_code
    elif isinstance(exc, DocumentManagementError):
        code = {
            404: "DOCUMENT_NOT_FOUND",
            409: "DB_ID_MISMATCH",
            412: "DOCUMENT_VERSION_MISMATCH",
        }.get(exc.status_code, "INVALID_ARGUMENT")
        message = str(exc)
        exit_code = 5 if exc.status_code in {409, 412} else 3 if exc.status_code == 404 else 2
    elif isinstance(exc, StorageUnavailableError):
        code, message, exit_code = "RECOVERY_REQUIRED", str(exc), 7
    elif isinstance(exc, (ValueError, FileNotFoundError, ConfigError)):
        code, message, exit_code = "INVALID_ARGUMENT", str(exc), 2
    else:
        logging.getLogger(__name__).exception("Command failed")
        code, message, exit_code = "BACKEND_UNAVAILABLE", "Operation failed; see logs/error.log", 6
    print(f"{code}: {message}", file=sys.stderr)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from story_rag.logging_config import configure_logging
        from story_rag.pipelines.databases import DatabasesPipeline

        settings = Settings.from_env(require_llm=False, require_reranker=False)
        configure_logging(settings)
        result = execute(args, DatabasesPipeline(settings))
        if isinstance(result, list):
            result = [r.model_dump() if hasattr(r, "model_dump") else r for r in result]
        emit(result, compact=getattr(args, "json", False))
        return 0
    except KeyboardInterrupt:
        return report_error(DatabaseError("CANCELLED", "Operation cancelled", 9))
    except Exception as exc:
        return report_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
