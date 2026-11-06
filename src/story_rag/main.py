from __future__ import annotations

import argparse
import asyncio

import uvicorn

from story_rag.api import create_app
from story_rag.cli import report_error, selector
from story_rag.config import Settings
from story_rag.container import build_container
from story_rag.domain import DatabaseError
from story_rag.logging_config import configure_logging
from story_rag.pipelines.databases import DatabasesPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="story-rag-api")
    selector(parser, required=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    storage = lease = None
    try:
        settings = Settings.from_env()
        configure_logging(settings)
        databases = DatabasesPipeline(settings)
        from story_rag.adapters.model_factory import model_bundle
        from story_rag.adapters.storage import build_storage

        if args.db or args.db_id:
            record = databases.resolve(args.db, args.db_id)
            if (
                record.embedding_model != settings.embedding_model
                or record.embedding_base_url != settings.lmstudio_base_url
            ):
                raise DatabaseError("STORAGE_INCONSISTENT", "Embedding configuration changed", 7)
            settings = databases.registry.bind(record)
            lease = databases.registry.lease(settings).acquire()
            databases.resolve(db_id=record.db_id)
        else:
            lease = databases.legacy_lease()
            registered = next((r for r in databases.registry.read() if r.legacy), None)
            if registered:
                if (
                    registered.embedding_model != settings.embedding_model
                    or registered.embedding_base_url != settings.lmstudio_base_url
                ):
                    raise DatabaseError(
                        "STORAGE_INCONSISTENT", "Embedding configuration changed", 7
                    )
                settings = databases.registry.bind(registered)
        models = model_bundle(settings)
        storage = build_storage(settings, models.embedding, lease=lease)
        container = build_container(settings, models=models, storage=storage)
        app = create_app(container)
        uvicorn.run(
            app, host=settings.app_host, port=settings.app_port, log_config=None, access_log=False
        )
        return 0
    except KeyboardInterrupt:
        return report_error(DatabaseError("CANCELLED", "API startup cancelled", 9))
    except Exception as exc:
        return report_error(exc)
    finally:
        if storage:
            asyncio.run(storage.aclose())
        elif lease:
            lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
