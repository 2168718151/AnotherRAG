# 会话账本接口：子方案一及后续集成补充

> 当前实现核对：2026-11-06。会话 schema 为版本 5，CLI/API 已集成；本文同时补充 2026-10-31 历史正文读取后的边界。

## 当前范围

PostgreSQL 账本保存长期会话身份、请求摘要、稳定题号/版本、对象条件和证据引用；独立展示表保存每轮用户原话与最终可见回答。
当前 `HistoryContentPipeline` 可在范围与版本核验后定向读取正文，用于原样复述、格式转换或追问；完整正文不会默认作为整段聊天历史回灌，也不作为知识证据。
账本不保存模型中间稿、完整 Prompt 或检索节点正文。不改变 Qdrant、JSON Docstore 与 `query` 预览链路。入口见[多轮询问端口](interfaces.md#多轮询问端口)。

## 配置和迁移

| 配置                          | 用途                                                                                   |
| --------------------------- | ------------------------------------------------------------------------------------ |
| `STORY_RAG_POSTGRES_DSN`    | PostgreSQL 连接串，Python 存储配置从进程环境读取；不要写入日志、快照或 Git。`run_api.ps1` 另可先从本机被忽略的连接文件填充此环境变量 |
| `STORY_RAG_POSTGRES_SCHEMA` | 账本 schema，默认 `story_rag`；开发/测试应使用独立 schema                                           |

应用不会隐式迁移。将 DSN 放入当前进程环境并核对目标后，明确执行 `uv run python scripts/migrate_conversations.py --apply`。
迁移在指定实例中创建或升级 schema、业务表和版本记录；当前版本 5 支持从版本 1–4 增量升级，重复运行不重建业务表，迁移失败时事务回滚。
发布副本不带原机集群、连接文件或既有会话；本次文档更新没有执行任何数据库迁移。

基础四张业务表是 `conversation_sessions`、`turn_requests`、`question_versions` 和 `object_versions`；版本 3 增加 `history_index_outbox`，版本 4 增加有效状态，版本 5 增加独立的 `conversation_display_turns`。`turn_requests` 只存请求 SHA-256、状态、顺序、题目引用与数值用量；题目/对象采用不可覆盖的版本行。展示表以轮次关联账本，保存 `user_message` 与可空的 `assistant_answer`。题目目标、短来源片段及条件仍属于业务数据，调用者不得把整段聊天或生成答案塞进这些字段。

## 生产端口

`PostgresSessionStore` 实现 `ConversationLedgerProductionPort`。调用者先由 `DatabaseRegistry.resolve()` 取得已登记且 `ready` 的库，再用 `binding_for_database(registry, record)` 构建包含数据库 UUID 和物理存储指纹的 `DatabaseBinding`。名称只用于展示，不能替代 UUID/指纹。

| 方法                                                                                                                        | 输入与行为                                                                            |
| ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `create_session(name, binding)`                                                                                           | 创建 UUID 会话，revision 为 0；同名不是身份                                                   |
| `get_session(session_id, binding, include_archived=False)`                                                                | 选取/恢复已绑定会话，核对库 UUID 与指纹                                                          |
| `list_sessions(binding, include_archived=False)`                                                                          | 只列出绑定范围内的会话                                                                      |
| `archive_session(session_id, binding, expected_revision=...)`                                                             | 归档但不删除知识库或账本；成功时 revision 加 1，重复归档不再递增                                           |
| `claim_request(session_id, binding, request_id=..., payload=..., expected_revision=...)`                                  | 认领请求；按完整请求字段及 expected revision 计算摘要，同事务将原话写入独立展示表                               |
| `get_request(session_id, binding, request_id)`                                                                            | 返回执行状态、轮次 ID、题目版本引用与用量元数据，不返回答案正文                                                |
| `list_display_turns(session_id, binding, limit=..., cursor=...)`                                                          | 按轮次 UUID 游标分页读取原话、可空最终回答与账本状态                                                    |
| `get_display_answer(session_id, binding, turn_id)`                                                                        | API/CLI 在成功请求重放时读取最终回答；受约束的历史正文任务使用专门的 `read_history_content()`                  |
| `read_history_content(session_id, binding, as_of_revision=..., before_turn_id=..., query=...)`                            | 按会话、库绑定、时间边界和选择条件回读已保存正文，返回来源与内容快照                                               |
| `commit_request(session_id, binding, request_id, questions=..., objects=..., usage=..., payload=..., visible_answer=...)` | 在单一短事务中追加题目/对象版本、更新请求状态与 revision，并写入最终可见回答；新条件修订需携带原始 `payload` 校验来源，不跨模型调用持有事务 |

`RequestPayload` 当前包括 `message`、可选 `reply_to_question_id` 与 `use_history`，均参与摘要。同一会话内相同 request ID 和相同摘要返回已知状态；相同 ID 配不同消息、选项或 expected revision 返回 `IDEMPOTENCY_CONFLICT`。已成功提交的重复 `commit_request` 返回原记录，不重复增加 revision；API/CLI 展示层可从独立表读取该轮已存的最终回答。迁移前的旧轮次无法从摘要或哈希恢复正文，须明确标记缺失。

`QuestionVersionDraft` 保存稳定 `question_id`、不可重用的 `question_no`、递增 `version`、目标、状态、关系、对象/证据引用和可选答案哈希。`ObjectVersionDraft` 保存同会话对象条件及短来源片段。跨会话引用、题号改绑和跳版本被拒绝。
生产 `ConversationWorkflow` 在执行必要的生成/校验后调用 `commit_request`；存储方法本身不负责评判答案语义。模块测试可以注入冻结结果或 Mock，不能据此声称真实质量通过。

## 诊断端口

`inspect_scope(session_id, binding)` 返回会话与关联表行数；`inspect_versions(session_id, binding)` 返回该会话的题目及对象版本；`inspect_display_scope(session_id, binding)` 仅统计展示轮次及已存回答数。诊断端口与生产读取一样强制核对绑定，不绕过身份边界，也不导出正文答案。

| 错误码                                             | 含义                     |
| ----------------------------------------------- | ---------------------- |
| `SESSION_NOT_FOUND` / `SESSION_ARCHIVED`        | 会话不存在或已归档              |
| `SCOPE_MISMATCH`                                | 会话绑定或对象/题目引用越出当前会话/知识库 |
| `REVISION_CONFLICT`                             | 认领或提交时 revision 已变化    |
| `IDEMPOTENCY_CONFLICT`                          | request ID 被另一请求内容占用   |
| `VERSION_CONFLICT`                              | 稳定题号或版本链不合法            |
| `REQUEST_NOT_FOUND` / `REQUEST_NOT_COMMITTABLE` | 请求不存在或需后续恢复流程处理        |
| `MIGRATION_CONFLICT`                            | 账本 schema 版本不受当前代码支持   |

同一会话执行阶段保持顺序，提交仍须核对 revision 和请求身份；冲突显式返回。确定失败/unknown 恢复、可靠历史索引更新和 API/CLI 生命周期已由子方案六补齐；账本端口本身仍不负责理解或生成答案。

## 验证边界

`tests/test_conversation_ledger.py` 从[唯一连续回归用例](多轮询问模块重建子方案/00-唯一连续回归用例.md)逐字读取五轮输入，以冻结的任务版本 Mock 测试会话绑定、五次提交、幂等、跨范围引用、并发 revision、重启恢复、归档和迁移回滚。测试自建并停止独立临时 PostgreSQL 实例；不调用 LLM、不打开主 Qdrant 集合，也不把 Mock 通过等同于真实多轮答案质量。
