from __future__ import annotations

import argparse
import json

from story_rag.adapters.session_store import build_session_store
from story_rag.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the conversation ledger migration")
    parser.add_argument("--apply", action="store_true", help="Apply schema changes")
    args = parser.parse_args()
    if not args.apply:
        parser.error("pass --apply to run the migration")
    settings = Settings.from_env(require_llm=False, require_reranker=False)
    store = build_session_store(settings)
    print(json.dumps({"schema": store.schema, "version": store.migrate()}))


if __name__ == "__main__":
    main()
