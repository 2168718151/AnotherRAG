from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from scripts.phase0_history_probe import BindingPlan, BoundTarget, prepare_binding
from story_rag.pipelines.history_resolution import RevisionBatch
from tests.test_conversation_ledger import _case_messages

pytestmark = pytest.mark.contract


def test_preserved_reference_does_not_require_new_question_text():
    message = _case_messages()[4]
    item = BoundTarget(
        task_keys=["t1"],
        operation="preserve",
        target_alias="h1",
        source_excerpt=message[-50:],
    )
    previous = SimpleNamespace(
        question_id=uuid4(), version=1, model_dump=lambda **_: {"version": 1}
    )

    def prepare(scope, request, payload, specs):
        assert specs == []
        return RevisionBatch()

    resolver = SimpleNamespace(
        resolve=lambda *_: SimpleNamespace(decision="matched", selected=previous),
        prepare_batch=prepare,
    )
    batch, links, specs = prepare_binding(
        BindingPlan(targets=[item]),
        SimpleNamespace(tasks=[SimpleNamespace(task_key="t1")]),
        message,
        {"h1": previous},
        resolver,
        None,
        None,
        None,
    )
    assert not batch.questions and not batch.objects and not specs
    assert links[0]["operation"] == "preserve"


def test_mutation_still_requires_question_text():
    with pytest.raises(ValidationError, match="requires question_text"):
        BoundTarget(
            task_keys=["t1"],
            operation="supplement",
            target_alias="h1",
            source_excerpt=_case_messages()[1][:30],
        )


@pytest.mark.parametrize("failure", ["alias", "source", "value", "coverage"])
def test_probe_rejects_unverified_inputs_before_preparing_revisions(failure):
    message = _case_messages()[1]
    target = dict(
        task_keys=["t1"],
        operation="new",
        object_kind="test_object",
        question_text=message,
        source_excerpt=message[:30],
        patches=[
            dict(field="tenancy", op="supplement", value="多租户", source_excerpt="系统改成多租户")
        ],
    )
    if failure == "alias":
        target.update(operation="supplement", target_alias="h999")
    elif failure == "source":
        target["source_excerpt"] = _case_messages()[4][:30]
    elif failure == "value":
        target["patches"][0]["value"] = "single-tenant"
    else:
        target["task_keys"] = ["t2"]
    with pytest.raises(ValueError):
        prepare_binding(
            BindingPlan(targets=[BoundTarget(**target)]),
            SimpleNamespace(tasks=[SimpleNamespace(task_key="t1")]),
            message,
            {},
            None,
            None,
            None,
            None,
        )
