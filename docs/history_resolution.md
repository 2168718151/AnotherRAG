# 历史定位与条件修订接口：子方案三

> 当前实现核对：2026-11-06。历史定位已接入 `ConversationWorkflow`，由 CLI `ask` 和会话 API 使用；原子方案三仍提供独立 Python 诊断端口。

## 阶段边界

本模块提供历史身份定位与修订草稿，已由生产会话工作流编排；不承担外部工单、审批平台等业务写执行。
模块测试使用冻结计划和可信定位选择器，不能据此声称自然语言指代在真实模型下始终稳定。
旧单轮 `query` 的零生成预览边界不变。PostgreSQL 是历史身份权威，不新增历史向量索引。

应用不会自动迁移。核对 `STORY_RAG_POSTGRES_DSN` 与 `STORY_RAG_POSTGRES_SCHEMA` 后，显式执行 `uv run python scripts/migrate_conversations.py --apply`。
当前完整应用要求 schema 版本 5；版本 2 是历史定位阶段引入 `changes`、`derived_from`、`revision_kind` 的历史增量，不是当前部署目标版本。
迁移与升级说明见[会话账本](conversation_ledger.md)。不要对主库执行隔离测试命令。

## 生产端口

`PostgresSessionStore` 增加 `read_questions_at()`、`read_objects_at()`、`current_object_at()` 和 `next_question_no()`。每次读取先校验会话、知识库 UUID、存储指纹和 `as_of_revision`，只读取该范围内已成功提交且在指定 revision 可见的版本。`read_questions_at()` 必须带题号、轮次、题目 UUID 或 `pending_only`，不提供无边界全量扫描。轮次+题号按原轮次锁定身份，再返回 as-of 的当前版本；无匹配返回空，不改用相似题。整轮包含多个题时返回候选，由调用者澄清。

`HistoryResolutionPipeline.resolve(scope, lookup)` 返回 `matched/ambiguous/not_found`、当前题目版本及其对象版本。
`use_history=false` 时不能通过该端口读取历史或准备依赖历史的修订；当前会话工作流另行处理关闭历史时的新问题，不将其解释为允许回读旧内容。
请求的 `use_history` 与解析范围不一致也会被拒绝。模型产生的题号、UUID、相似度分数不能直接作为可信身份；上游需提供有来源的明确引用或程序确认的唯一待补任务。

`prepare_batch(scope, request, payload, specs)` 为同一已认领请求准备追加版本，不立即提交。它验证请求摘要、revision、来源片段和逐字段值；补充不得覆盖旧字段，更正必须匹配旧值。同一请求内共享对象的新版本会传递给后续题目；同一对象的两次写入会被拒绝。当前题目指向旧对象版本时，事实修改返回 `STALE_OBJECT_REF`，不会静默基于过期条件继续。`new/extend/compare` 分配新题号；`supplement/replace/format/cancel` 沿用题号并追加版本；`ambiguous/not_found` 不产生草稿。比较分支的对象使用 `derived_from` 保留原来源，不能覆盖原对象。

准备结果须交由 `commit_request(..., payload=原始请求, questions=..., objects=...)` 在短事务中提交。带逐字段变化的对象必须传入原始请求，账本复核摘要、原文片段、旧值、字段差异、来源轮次与版本。旧账本直接调用端口仍兼容无 `changes` 的历史用法；新修订应通过本端口，不要绕过来源校验。`source_excerpt` 仅证明字面出处；不能凭引用片段自动推断公式、同义数值或真实权限状态。

## 确认与授权预检

`check_write_preflight()` 是纯函数，不执行工具调用。确认记录必须由可信服务端/审批平台取得，而非相信模型生成的 JSON。确认绑定主体、租户、动作、资源、参数 SHA-256、资源版本、授权版本及有效期；撤回、任一字段变化或过期均令确认失效。预检**另外**核验当前授权上下文的主体、租户、资源、撤回状态及所需 scope，并检查动作和实际执行器是否符合业务策略。草稿业务要求人工审批平台时，聊天确认不能替代平台批准，助手执行器也不能直接修改。工单 `ticket.read` 不满足 `ticket.create`；诊断出拒绝原因不等于自动授予写权限。

此端口不签发或持久化确认，不实现审批平台回执、服务端授权适配器或最终业务执行。真正接入写操作时必须在执行前重新加载可信确认、授权和资源版本，并验证平台最终状态/回执；仅调用纯函数不能声称完成安全闭环。

## 诊断和测试

`HistoryResolutionDebugPort` 与生产端口接受相同的冻结定位/修订输入，可检查候选、草稿、来源和范围，不调用 DeepSeek。日志仍由应用配置写入 `logs/`，本模块不打印原文、Prompt 或凭据。

`tests/test_history_resolution.py` 只使用方案 0 的五轮原文，注入方案二冻结计划并在独立临时 PostgreSQL 实例验证：第二/三轮双对象修订、第四轮仅草稿变化、第五轮独立工单及缺口补全；跨会话/库/指纹、as-of、轮次/题号精确定位、歧义不猜测、来源和旧值拒绝、确认失效与权限/平台分离。它不测真实模型理解、知识检索或生成质量。

用户原话和已保存回答的正文回读由 `HistoryContentPipeline` 承担，与题目/对象身份定位分开，见[历史原文读取](历史原文读取补丁与验收_20261031.md)。
