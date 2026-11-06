# 任务解析与目标覆盖接口

> 当前实现核对：2026-11-06。规划器已接入多轮 CLI/API；原子方案的测试边界仍保留，不代表本次重新完成真实模型验收。

## 工作流位置

```text
当前整条用户消息
  -> TaskPlanningProductionPort.plan()
  -> Schema、原文片段、目标映射和依赖校验
  -> ConversationWorkflow：历史定位 / 逐题检索 / 生成与提交
```

规划器使用项目现有 LlamaIndex LLM 的 `structured_predict()` 与 Pydantic 结构化输出。一次调用处理完整消息，不为每个任务单独规划。成功时正常为一次 LLM 调用；若结构化输出或程序校验失败，最多发起一次携带错误类型反馈的修复调用。网络或服务异常不作为格式错误重试。

## 生产端口

```python
result = container.task_planner.plan(message)
result.plan.schema_version
result.plan.goals
result.plan.tasks
result.llm_calls
result.attempts
result.elapsed_ms
```

计划中的目标与任务键仅在当前计划内有效，不是稳定题号或数据库 ID。每个目标映射到至少一个任务；每个任务也必须映射到至少一个目标。任务可保留约束、禁止动作、输出要求、对象范围、历史需求、信息缺口、关系候选及局部依赖。信息缺口区分 `explicitly_missing`、`not_provided` 和 `uncertain`，并要求用当前消息原文说明缺口依据；不能把未提供的事实补成推测。模糊关系仍保留为候选，不会自动去重或合并。

所有目标和带语义的任务事实均要求携带当前消息中的精确 `source_excerpt`。程序检查精确子串、未知字段、重复/缺失映射、悬空引用、自引用和依赖环。校验失败两次后抛出 `TaskPlanningError(code="INVALID_PLAN")`；调用方不得继续检索或生成。空消息以 `EMPTY_MESSAGE` 拒绝。

规划器本身返回结构化计划，不直接写 PostgreSQL、分配稳定题号、执行检索或生成。
当前 `ConversationWorkflow` 已消费该计划并连接历史定位、证据映射、生成和账本提交；
规划结果中的临时键不能直接当作历史身份。现有零生成的 `query` 预览链路保持不变。
历史正文查询和当前规划契约的扩展见[历史原文读取](历史原文读取补丁与验收_20261031.md)。

## 测试诊断端口

```python
diagnostic = container.task_planner.validate_candidate(message, fixed_plan)
diagnostic.valid
diagnostic.plan
diagnostic.errors
diagnostic.llm_calls  # 固定为 0
```

该端口接受冻结计划或字典，进行与生产一致的结构及原文校验，不调用模型。它适合程序契约和故障注入测试；不能证明 DeepSeek 的任务理解质量。

规划器默认 trace 记录校验结果、任务/目标数量与尝试次数，不包含 API 凭据。
完整工作流的显式诊断可另外保存 Prompt、节点和模型输出；这些产物属于敏感运行数据，不随发布副本提供。

## 验收边界

- 快速回归仅使用 `docs/多轮询问模块重建子方案/00-唯一连续回归用例.md` 的五轮原文。
- 固定结构化结果用于验证五轮任务分离、范围、禁止动作、历史需求和目标覆盖。
- Schema 非法、原文引用不存在、悬空目标/关系及依赖环均须拒绝；有效冻结计划不得调用 LLM。
- 单次有效结构化模型输出的契约测试只证明 LlamaIndex 适配和调用次数，不代表真实模型理解质量已验收。
- 上述模块契约测试不运行付费 DeepSeek 质量评估；当前业务集成通过 `ask` 和会话 HTTP 接口使用规划器。原测试不能据此扩大为真实理解质量验收。
