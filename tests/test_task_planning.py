from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from story_rag.logging_config import TraceWriter
from story_rag.pipelines.task_planning import TaskPlanningError, TaskPlanningPipeline
from tests.conftest import ScriptedLLM

ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "docs" / "多轮询问模块重建子方案" / "00-唯一连续回归用例.md"


def _case_messages() -> list[str]:
    source = CASE_PATH.read_text(encoding="utf-8")
    messages = re.findall(r"(?ms)^## 第 [1-5] 轮\s*\n\s*> (.+?)(?=\n\n## |\Z)", source)
    assert len(messages) == 5
    return [message.strip() for message in messages]


def _fact(text: str, source_excerpt: str) -> dict[str, str]:
    return {"text": text, "source_excerpt": source_excerpt}


def _goal(
    message: str, key: str, text: str, excerpt: str, task_keys: list[str]
) -> dict[str, object]:
    assert excerpt in message
    return {
        "goal_key": key,
        "goal_text": text,
        "source_excerpt": excerpt,
        "task_keys": task_keys,
    }


def _task(
    key: str,
    intent: str,
    objective: dict[str, str],
    *,
    scope: dict[str, str] | None = None,
    constraints: list[dict[str, str]] | None = None,
    prohibited: list[dict[str, str]] | None = None,
    outputs: list[dict[str, str]] | None = None,
    information_gaps: list[dict[str, str]] | None = None,
    history_need: str = "none",
    history_reference: dict[str, str] | None = None,
    dependencies: list[str] | None = None,
    relations: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "task_key": key,
        "intent": intent,
        "objective": objective,
        "object_scope": scope,
        "constraints": constraints or [],
        "prohibited_actions": prohibited or [],
        "output_requirements": outputs or [],
        "information_gaps": information_gaps or [],
        "history_need": history_need,
        "history_reference": history_reference,
        "dependencies": dependencies or [],
        "relation_candidates": relations or [],
    }


def _frozen_plan(message: str, turn: int) -> dict[str, object]:
    if turn == 1:
        goals = [
            _goal(
                message,
                "g1",
                "说明提示词约束、用户确认和服务端鉴权各自的作用与局限",
                "Agent 写操作权限怎么控制？提示词约束、用户确认和服务端鉴权分别能解决什么问题，"
                "又有哪些局限？",
                ["t1"],
            ),
            _goal(
                message,
                "g2",
                "设计单租户同步场景下仅修改本人草稿标题的流程",
                "We are building an internal assistant that may change only the title of the "
                "current user's "
                "draft. It must not publish, delete, or notify anyone.",
                ["t2"],
            ),
            _goal(
                message,
                "g3",
                "说明草稿变更审批执行流程和模型外检查",
                "How would you design the approval and execution workflow, and "
                "which checks must be "
                "enforced outside the model?",
                ["t2"],
            ),
            _goal(
                message,
                "g4",
                "判断独立工单环境失败原因是否可确定并指出缺失材料",
                "另一个独立测试环境里的助手提交工单一直失败，tool call failed，err@@锟斤拷??。"
                "你能确定到底是参数校验、权限拦截还是业务执行出了问题吗？调用记录和错误日志我还没提供，"
                "后面再补。",
                ["t3"],
            ),
            _goal(
                message,
                "g5",
                "分别回答多个问题，缺信息时说明且不猜测",
                "请分别回答，信息不够就说明缺什么，不要把猜测当成结论。",
                ["t1", "t2", "t3"],
            ),
        ]
        tasks = [
            _task(
                "t1",
                "answer",
                _fact("解释三种写操作控制机制的职责和局限", "Agent 写操作权限怎么控制？"),
                constraints=[
                    _fact(
                        "缺少信息时明确说明，不把猜测当结论",
                        "信息不够就说明缺什么，不要把猜测当成结论",
                    )
                ],
                outputs=[_fact("逐项说明机制", "分别能解决什么问题")],
            ),
            _task(
                "t2",
                "answer",
                _fact(
                    "设计只修改本人草稿标题的审批与执行流程", "the approval and execution workflow"
                ),
                scope=_fact("当前认证用户本人的草稿标题", "the title of the current user's draft"),
                constraints=[
                    _fact("单租户环境", "single-tenant application"),
                    _fact("请求同步执行", "requests run synchronously"),
                    _fact("请求处于认证用户会话内", "within the authenticated user's session"),
                ],
                prohibited=[
                    _fact("不得发布", "must not publish"),
                    _fact("不得删除", "delete"),
                    _fact("不得通知任何人", "notify anyone"),
                ],
                outputs=[
                    _fact(
                        "指出必须在模型外实施的检查",
                        "which checks must be enforced outside the model?",
                    )
                ],
            ),
            _task(
                "t3",
                "diagnose",
                _fact("诊断独立工单测试环境的失败环节", "助手提交工单一直失败"),
                scope=_fact("独立测试环境", "另一个独立测试环境"),
                constraints=[
                    _fact(
                        "当前不能确定失败环节",
                        "你能确定到底是参数校验、权限拦截还是业务执行出了问题吗？",
                    ),
                    _fact("调用记录与错误日志尚未提供", "调用记录和错误日志我还没提供"),
                ],
                information_gaps=[
                    {
                        "description": "工单调用记录和错误日志尚未提供，当前不能确定失败环节",
                        "state": "explicitly_missing",
                        "basis_excerpt": "调用记录和错误日志我还没提供",
                    }
                ],
                outputs=[_fact("说明需要的调用记录和错误日志", "调用记录和错误日志")],
            ),
        ]
    elif turn == 2:
        goals = [
            _goal(
                message,
                "g1",
                "将通用写权限原则扩展到多租户、异步队列和共享服务账号",
                "如果系统改成多租户，用户发起的操作不再立即执行，而是进入后台队列，并且执行器使用共享服务账号访问业务接口，原来的权限控制方案需要怎样调整？",
                ["t1"],
            ),
            _goal(
                message,
                "g2",
                "关联发起用户、租户、目标资源与执行身份",
                "发起用户、租户、目标资源和实际执行身份之间应该怎样关联",
                ["t1", "t2"],
            ),
            _goal(
                message,
                "g3",
                "同步修订草稿流程并覆盖入队和执行前检查",
                "这些变化也适用于刚才那个修改草稿标题的助手，请同步修订通用的权限控制方案和具体业务流程，不要只考虑请求进入队列时的检查。",
                ["t1", "t2"],
            ),
        ]
        history = _fact("关联上一轮写权限原则", "接着刚才关于Agent写权限的解释")
        tasks = [
            _task(
                "t1",
                "revise",
                _fact("修订多租户异步执行下的通用写权限控制", "原来的权限控制方案需要怎样调整？"),
                scope=_fact(
                    "发起用户、租户、目标资源和实际执行身份",
                    "发起用户、租户、目标资源和实际执行身份",
                ),
                constraints=[
                    _fact("用户请求进入后台队列", "进入后台队列"),
                    _fact("执行器使用共享服务账号", "执行器使用共享服务账号访问业务接口"),
                    _fact("入队检查之外还要考虑后续检查", "不要只考虑请求进入队列时的检查"),
                ],
                outputs=[_fact("同步修订通用权限控制原则", "同步修订通用的权限控制方案")],
                history_need="required",
                history_reference=history,
            ),
            _task(
                "t2",
                "revise",
                _fact("将多租户异步权限控制用于本人草稿标题助手", "修改草稿标题的助手"),
                constraints=[_fact("同步修订草稿业务流程", "具体业务流程")],
                outputs=[_fact("同步更新具体业务流程", "具体业务流程")],
                history_need="required",
                history_reference=_fact("关联上一轮的草稿标题场景", "刚才那个修改草稿标题的助手"),
            ),
        ]
    elif turn == 3:
        goals = [
            _goal(
                message,
                "g1",
                "判断参数、目标、授权或资源变化后原确认是否仍有效",
                "如果模型重新生成了工具参数、更换了目标草稿，或者任务排队期间用户撤回授权、草稿本身发生变化，之前的确认结果还能直接使用吗？",
                ["t2"],
            ),
            _goal(
                message,
                "g2",
                "说明用户确认与服务端授权的关系及确认不可复用范围",
                "用户确认和服务端授权之间究竟是什么关系，确认结果能不能作为后续其他写操作的通行证？",
                ["t1"],
            ),
            _goal(
                message,
                "g3",
                "补充确认范围、提交前复核、失效处理并同步修订原则与流程",
                "请在同一轮里补充确认适用范围、提交前复核以及失效后的处理方式，并同步更新前面的权限控制原则和草稿修改流程。",
                ["t1", "t2"],
            ),
        ]
        tasks = [
            _task(
                "t1",
                "revise",
                _fact("说明确认与服务端授权的边界", "用户确认和服务端授权之间究竟是什么关系"),
                constraints=[
                    _fact("确认不能成为其他写操作的通行证", "能不能作为后续其他写操作的通行证？"),
                    _fact("补充确认适用范围", "确认适用范围"),
                ],
                outputs=[_fact("同步更新通用权限控制原则", "前面的权限控制原则")],
                history_need="required",
                history_reference=_fact("承接上一轮的通用写权限原则", "你上一轮对写权限控制的解释"),
            ),
            _task(
                "t2",
                "revise",
                _fact("复核草稿修改流程中的既有确认", "之前的确认结果还能直接使用吗？"),
                constraints=[
                    _fact("参数或目标变化后重新核验确认", "重新生成了工具参数、更换了目标草稿"),
                    _fact("队列期间撤权后重新核验", "用户撤回授权"),
                    _fact("草稿变化后重新核验", "草稿本身发生变化"),
                    _fact("提交前复核", "提交前复核"),
                    _fact("明确确认失效后的处理方式", "失效后的处理方式"),
                ],
                outputs=[_fact("同步更新草稿修改流程", "草稿修改流程")],
                history_need="required",
                history_reference=_fact("承接上一轮的草稿修改流程", "刚才给出的草稿修改流程"),
            ),
        ]
    elif turn == 4:
        goals = [
            _goal(
                message,
                "g1",
                "草稿标题修改必须由人工审批平台生成并执行变更单",
                "它只能生成待审批的变更单，真正的修改必须由现有人工审批平台执行",
                ["t1"],
            ),
            _goal(
                message,
                "g2",
                "禁止发布、删除、通知及以聊天确认绕过平台",
                "发布、删除和通知仍然不允许。即使用户在聊天里说“同意修改”，也不能跳过这个平台。",
                ["t1"],
            ),
            _goal(
                message,
                "g3",
                "说明助手、审批平台职责及如何核验执行结果",
                "请说明助手与审批平台各自负责什么，以及怎样确认申请最终是否执行成功。",
                ["t1"],
            ),
            _goal(
                message,
                "g4",
                "仅调整草稿业务流程并保持通用权限原则不变",
                "这次只调整草稿业务的实现流程，前面关于Agent写权限的通用原则保持不变。",
                ["t1"],
            ),
        ]
        tasks = [
            _task(
                "t1",
                "revise",
                _fact(
                    "修订本人草稿标题变更的人工审批平台流程", "真正的修改必须由现有人工审批平台执行"
                ),
                scope=_fact("只调整草稿业务实现流程", "只调整草稿业务的实现流程"),
                constraints=[
                    _fact("助手只生成待审批变更单", "它只能生成待审批的变更单"),
                    _fact("保持通用写权限原则不变", "通用原则保持不变"),
                    _fact("通过平台最终状态或回执确认执行", "怎样确认申请最终是否执行成功"),
                ],
                prohibited=[
                    _fact("禁止发布、删除或通知", "发布、删除和通知仍然不允许"),
                    _fact("聊天确认不得绕过审批平台", "不能跳过这个平台"),
                ],
                outputs=[_fact("说明助手与审批平台的职责", "助手与审批平台各自负责什么")],
                history_need="required",
                history_reference=_fact("沿用前述草稿流程", "刚才那个修改草稿标题的业务"),
            )
        ]
    elif turn == 5:
        goals = [
            _goal(
                message,
                "g1",
                "根据本轮调用记录定位工单创建失败环节",
                "权限策略返回deny，拒绝原因是missing_write_scope",
                ["t1"],
            ),
            _goal(
                message,
                "g2",
                "区分已经通过的检查与尚未执行的业务步骤",
                "参数Schema校验通过，身份认证通过，租户和目标资源归属检查通过",
                ["t1"],
            ),
            _goal(
                message,
                "g3",
                "在不绕过权限控制的前提下区分原因定位与授予权限",
                "区分“定位失败原因”和“是否应该授予写权限”这两件事",
                ["t2"],
            ),
            _goal(
                message,
                "g4",
                "保持独立工单环境与草稿人工审批流程隔离",
                "这里说的是独立工单环境，不改变前面草稿业务必须通过人工审批平台执行的要求。",
                ["t1", "t2"],
            ),
        ]
        tasks = [
            _task(
                "t1",
                "diagnose",
                _fact("定位创建工单请求的权限策略拒绝", "权限策略返回deny"),
                scope=_fact("独立工单测试环境", "独立工单测试环境"),
                constraints=[
                    _fact("Schema 校验已经通过", "参数Schema校验通过"),
                    _fact("身份认证已经通过", "身份认证通过"),
                    _fact("租户及资源归属检查已经通过", "租户和目标资源归属检查通过"),
                    _fact("授权上下文只有读取权限", "授权上下文只有ticket.read"),
                    _fact("创建动作缺少写权限", "请求动作是ticket.create"),
                    _fact("拒绝原因是缺少写权限范围", "missing_write_scope"),
                    _fact("业务处理接口未调用", "执行器没有调用后面的业务处理接口"),
                    _fact("未创建工单记录", "也没有创建工单记录"),
                    _fact("不存在网络超时", "不存在网络超时"),
                    _fact("不存在执行后响应丢失", "已经执行但响应丢失"),
                ],
                outputs=[_fact("说明已经通过和未执行步骤", "哪些检查已经通过，哪些步骤尚未执行？")],
                history_need="required",
                history_reference=_fact("沿用此前独立工单环境的边界", "刚才那个独立工单测试环境"),
            ),
            _task(
                "t2",
                "answer",
                _fact(
                    "说明合规处理方式并区分定位与授权决策",
                    "接下来应该怎样在不绕过权限控制的前提下处理",
                ),
                constraints=[
                    _fact("不绕过现有权限控制", "不绕过权限控制"),
                    _fact("原因已经定位不等于应授予写权限", "定位失败原因”和“是否应该授予写权限"),
                    _fact(
                        "保持与草稿审批流程独立", "不改变前面草稿业务必须通过人工审批平台执行的要求"
                    ),
                ],
                information_gaps=[
                    {
                        "description": "是否授予 ticket.create 权限需要独立的业务授权决策",
                        "state": "uncertain",
                        "basis_excerpt": "是否应该授予写权限",
                    }
                ],
                dependencies=["t1"],
                history_need="required",
                history_reference=_fact("保持独立工单环境", "独立工单环境"),
            ),
        ]
    else:
        raise ValueError(f"Unknown fixed case turn: {turn}")
    return {"schema_version": 1, "goals": goals, "tasks": tasks}


def _plan_text(plan: dict[str, object]) -> str:
    return json.dumps(plan, ensure_ascii=False)


@pytest.mark.unit
def test_five_turn_fixed_plans_preserve_goals_scope_and_boundaries() -> None:
    messages = _case_messages()
    planner = TaskPlanningPipeline(ScriptedLLM())
    plans = []

    for turn, message in enumerate(messages, 1):
        candidate = _frozen_plan(message, turn)
        result = planner.validate_candidate(message, candidate)
        assert result.valid, (turn, result.errors)
        assert result.llm_calls == 0
        assert result.plan is not None
        assert all(goal.task_keys for goal in result.plan.goals)
        plans.append(_plan_text(candidate))

    assert len(messages) == len(plans) == 5
    assert len(planner._llm.prompts) == 0

    first = plans[0]
    assert all(text in first for text in ("服务端鉴权", "current user's draft", "must not publish"))
    assert all(
        text in first
        for text in (
            "delete",
            "notify anyone",
            "single-tenant application",
            "requests run synchronously",
            "authenticated user's session",
            "调用记录和错误日志我还没提供",
            "explicitly_missing",
        )
    )

    second = plans[1]
    assert all(text in second for text in ("多租户", "后台队列", "共享服务账号"))
    assert all(text in second for text in ("发起用户", "租户", "目标资源", "实际执行身份"))

    third = plans[2]
    assert all(
        text in third
        for text in (
            "重新生成了工具参数",
            "用户撤回授权",
            "草稿本身发生变化",
            "提交前复核",
            "失效后的处理方式",
        )
    )
    assert "后续其他写操作的通行证" in third

    fourth = plans[3]
    assert all(text in fourth for text in ("待审批的变更单", "人工审批平台", "通用原则保持不变"))
    assert all(text in fourth for text in ("发布", "删除", "通知", "不能跳过这个平台"))

    fifth = plans[4]
    assert all(text in fifth for text in ("ticket.read", "ticket.create", "missing_write_scope"))
    assert all(
        text in fifth
        for text in (
            "业务处理接口",
            "没有创建工单记录",
            "不存在网络超时",
            "已经执行但响应丢失",
            "不绕过权限控制",
            "是否授予 ticket.create 权限需要独立的业务授权决策",
        )
    )
    assert "是否应该授予写权限" in fifth


@pytest.mark.unit
def test_llamaindex_structured_predict_uses_one_call_for_valid_whole_message() -> None:
    message = _case_messages()[0]
    candidate = _frozen_plan(message, 1)
    llm = ScriptedLLM(responses=[json.dumps(candidate, ensure_ascii=False)])
    result = TaskPlanningPipeline(llm).plan(message)

    assert result.plan.schema_version == 1
    assert len(result.plan.goals) == 5
    assert len(result.plan.tasks) == 3
    assert result.llm_calls == result.attempts == 1
    assert len(llm.prompts) == 1
    assert message in llm.prompts[0]
    assert "task_key" in llm.prompts[0]


@pytest.mark.unit
def test_invalid_structured_response_gets_only_one_bounded_repair() -> None:
    message = _case_messages()[0]
    candidate = _frozen_plan(message, 1)
    llm = ScriptedLLM(responses=["not valid JSON", json.dumps(candidate, ensure_ascii=False)])
    result = TaskPlanningPipeline(llm).plan(message)

    assert result.llm_calls == result.attempts == 2
    assert len(llm.prompts) == 2
    assert "validation feedback" in llm.prompts[1].casefold()


@pytest.mark.unit
def test_repeated_invalid_output_fails_closed_after_two_calls() -> None:
    message = _case_messages()[0]
    llm = ScriptedLLM(responses=["not valid JSON", "still not valid JSON"])

    with pytest.raises(TaskPlanningError) as raised:
        TaskPlanningPipeline(llm).plan(message)

    assert raised.value.code == "INVALID_PLAN"
    assert raised.value.attempts == 2
    assert len(llm.prompts) == 2


@pytest.mark.unit
def test_provider_error_is_not_retried_as_a_format_error() -> None:
    class ProviderFailureLLM(ScriptedLLM):
        calls: int = 0

        def structured_predict(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("provider unavailable")

    llm = ProviderFailureLLM()

    with pytest.raises(RuntimeError, match="provider unavailable"):
        TaskPlanningPipeline(llm).plan(_case_messages()[0])

    assert llm.calls == 1


@pytest.mark.unit
def test_debug_port_rejects_unknown_fields_and_identity_claims() -> None:
    message = _case_messages()[0]
    candidate = _frozen_plan(message, 1)
    candidate["session_id"] = "must-not-be-model-selected"
    planner = TaskPlanningPipeline(ScriptedLLM())

    result = planner.validate_candidate(message, candidate)

    assert not result.valid
    assert result.plan is None
    assert result.llm_calls == 0
    assert any("extra_forbidden" in error for error in result.errors)
    assert not planner._llm.prompts


@pytest.mark.unit
def test_application_container_exposes_task_planner(container) -> None:
    assert container.task_planner is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("dangling_goal", "unknown"),
        ("missing_task_mapping", "every task"),
        ("dependency_cycle", "cycle"),
        ("unknown_history_id", "extra_forbidden"),
    ],
)
def test_debug_port_rejects_broken_mappings_graph_and_history_identity(
    mutation: str, expected_error: str
) -> None:
    message = _case_messages()[0]
    candidate = copy.deepcopy(_frozen_plan(message, 1))
    if mutation == "dangling_goal":
        candidate["goals"][0]["task_keys"] = ["t99"]
    elif mutation == "missing_task_mapping":
        candidate["goals"] = candidate["goals"][:-1]
        candidate["goals"][3]["task_keys"] = ["t1"]
    elif mutation == "dependency_cycle":
        candidate["tasks"][0]["dependencies"] = ["t2"]
        candidate["tasks"][1]["dependencies"] = ["t1"]
    else:
        candidate["tasks"][0]["selected_question_id"] = "forbidden"

    result = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(message, candidate)

    assert not result.valid
    assert any(expected_error in error for error in result.errors)


@pytest.mark.unit
def test_debug_port_rejects_source_quotes_not_present_in_current_message() -> None:
    message = _case_messages()[0]
    candidate = copy.deepcopy(_frozen_plan(message, 1))
    candidate["tasks"][0]["objective"]["source_excerpt"] = "未曾出现在用户消息中的事实"

    result = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(message, candidate)

    assert not result.valid
    assert result.errors == ["SOURCE_EXCERPT_NOT_IN_MESSAGE"]


@pytest.mark.unit
def test_ambiguous_task_relation_is_preserved_without_merging() -> None:
    message = _case_messages()[4]
    candidate = _frozen_plan(message, 5)
    candidate["tasks"][1]["relation_candidates"] = [{"kind": "ambiguous", "target_task_key": "t1"}]

    result = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(message, candidate)

    assert result.valid
    assert result.plan is not None
    assert len(result.plan.tasks) == 2
    assert result.plan.tasks[1].relation_candidates[0].kind == "ambiguous"


@pytest.mark.unit
def test_phase0_invalid_advisory_relation_is_dropped_without_losing_task() -> None:
    message = _case_messages()[3]
    candidate = _frozen_plan(message, 4)
    candidate["tasks"][0]["relation_candidates"] = [
        {"kind": "dependency", "target_task_key": "t99"}
    ]
    result = TaskPlanningPipeline(ScriptedLLM()).validate_candidate(message, candidate)
    assert result.valid
    assert result.plan is not None
    assert len(result.plan.tasks) == 1
    assert result.plan.tasks[0].relation_candidates == []


@pytest.mark.unit
def test_validation_trace_does_not_write_message_or_prompt(tmp_path: Path) -> None:
    message = _case_messages()[0]
    candidate = _frozen_plan(message, 1)
    trace_writer = TraceWriter(tmp_path / "logs")
    llm = ScriptedLLM(responses=[json.dumps(candidate, ensure_ascii=False)])

    TaskPlanningPipeline(llm, trace_writer).plan(message)

    traces = list((tmp_path / "logs" / "traces").glob("*.jsonl"))
    content = "\n".join(path.read_text(encoding="utf-8") for path in traces)
    assert traces
    assert message not in content
    assert "task_plan_validated" in content


@pytest.mark.unit
def test_empty_message_is_rejected_without_model_call() -> None:
    llm = ScriptedLLM()

    with pytest.raises(TaskPlanningError) as raised:
        TaskPlanningPipeline(llm).plan("  \n ")

    assert raised.value.code == "EMPTY_MESSAGE"
    assert not llm.prompts
