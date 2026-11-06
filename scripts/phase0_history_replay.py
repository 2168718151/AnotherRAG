"""Replay saved real provider responses without calling any model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from llama_index.core.output_parsers import PydanticOutputParser

from scripts.phase0_history_probe import BindingPlan, catalogue_for, prepare_binding
from scripts.phase0_probe_support import CASE, ROOT, dump, isolated_postgres, now
from story_rag.adapters.session_store import PostgresSessionStore
from story_rag.conversation_domain import DatabaseBinding, RequestPayload
from story_rag.pipelines.history_resolution import HistoryResolutionPipeline, HistoryScope
from story_rag.pipelines.task_planning import TaskPlan, TaskPlanningPipeline


def provider_result(directory, stage, model):
    events = json.loads((directory / "llm_events.json").read_text(encoding="utf-8"))
    last = [
        event
        for event in events
        if event.get("provider", {}).get("id") and event["stage"].endswith(stage)
    ][-1]
    content = last["choices"][0]["content"]
    return PydanticOutputParser(model).parse(content), {
        "provider_response_id": last["provider"]["id"],
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    original = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if original["case_sha256"] != hashlib.sha256(CASE.read_bytes()).hexdigest():
        raise ValueError("Source case changed")
    output.mkdir(parents=True, exist_ok=False)
    log_dir = ROOT / "logs" / output.name
    log_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(
        started_at=now(),
        mode="saved_provider_response_replay",
        source=str(source),
        llm_calls=0,
        turns=[],
    )
    # No LLM object is constructed; only the planner's diagnostic validation is used.
    planner = TaskPlanningPipeline(None)
    with isolated_postgres(output, log_dir) as dsn:
        store = PostgresSessionStore(dsn, schema="phase0_replay")
        store.migrate()
        binding = DatabaseBinding(
            database_id=uuid4(),
            storage_fingerprint=hashlib.sha256(str(output).encode()).hexdigest(),
        )
        session = store.create_session("saved actual provider responses", binding)
        resolver = HistoryResolutionPipeline(store)
        for original_turn in original["turns"]:
            number, message = original_turn["input_turn"], original_turn["message"]
            previous = store.get_session(session.session_id, binding)
            scope = HistoryScope(
                session_id=session.session_id, binding=binding, as_of_revision=previous.revision
            )
            payload = RequestPayload(message=message)
            request = store.claim_request(
                session.session_id,
                binding,
                request_id=f"replay-{number}",
                payload=payload,
                expected_revision=previous.revision,
            )
            directory = output / f"turn_{number}"
            plan, plan_provenance = provider_result(
                source / f"turn_{number}", ".planning", TaskPlan
            )
            validated = planner.validate_candidate(message, plan)
            if not validated.valid:
                raise ValueError(validated.errors)
            candidate, link_provenance = provider_result(
                source / f"turn_{number}", ".binding", BindingPlan
            )
            catalogue, identities = catalogue_for(store, scope)
            batch, links, specs = prepare_binding(
                candidate, plan, message, identities, resolver, scope, request, payload
            )
            committed = store.commit_request(
                session.session_id,
                binding,
                request.request_id,
                questions=batch.questions,
                objects=batch.objects,
                payload=payload,
            )
            row = dict(
                input_turn=number,
                message=message,
                plan_provenance=plan_provenance,
                binding_provenance=link_provenance,
                links=links,
                committed_request=committed,
                binding=candidate,
                catalogue=catalogue,
            )
            dump(directory / "result.json", row)
            dump(directory / "prepared.json", batch)
            dump(directory / "specs.json", specs)
            dump(
                directory / "ledger_after.json", store.inspect_versions(session.session_id, binding)
            )
            summary["turns"].append(row)
            dump(output / "summary.json", summary)
        summary["final_session"] = store.get_session(session.session_id, binding)
        dump(output / "final_ledger.json", store.inspect_versions(session.session_id, binding))
    summary.update(ended_at=now(), postgres_stopped=True)
    dump(output / "summary.json", summary)
    print(
        json.dumps(
            dict(
                turns=len(summary["turns"]), llm_calls=0, revision=summary["final_session"].revision
            )
        )
    )


if __name__ == "__main__":
    main()
