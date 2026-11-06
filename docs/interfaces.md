# 端口与接口文档

> 当前实现核对：2026-11-06。首次运行见[项目 README](../README.md)，文档分类见[导航](README.md)。以下区分单轮 API、零生成 CLI 预览与已集成的多轮会话。

## 生产端口

| 端口 | 方法 |
|---|---|
| `IngestionProductionPort` | `ingest_bytes()`、`ingest_path()` |
| `RetrievalProductionPort` | `retrieve()` |
| `GenerationProductionPort` | `answer()` |
| `WorkflowProductionPort` | `answer()` |
| `EvaluationProductionPort` | `run()` |
| `DocumentsProductionPort` | `list_documents()`、`get_document()`、`preview_document()`、`delete_document()`、`rename_document()`、`replace_document()` |
| `DatabasesProductionPort` | `create()`、`list_databases()`、`info()`、`resolve()`、`register_default()`、`open()`、`rename()`、`drop()`、`recover()` |
| `QuerySessionProductionPort` | `select()`、`query()`、`release()`、`close()` |
| `ConversationLedgerProductionPort` | `create_session()`、`get_session()`、`list_sessions()`、`archive_session()`、`claim_request()`、`get_request()`、`commit_request()`、`read_questions_at()`、`read_objects_at()` |
| `TaskPlanningProductionPort` | `plan()`：整条用户消息一次结构化规划；校验失败最多一次有界修复 |
| `HistoryResolutionProductionPort` | `resolve()`、`prepare_batch()`：按绑定和 revision 定位历史，准备可追溯的追加修订 |
| `EvidenceMappingProductionPort` | `map_evidence()`：绑定库内逐题混合检索、版本核验与许可证据映射 |

## 测试诊断端口

| 端口 | 方法 |
|---|---|
| `ConfigDebugPort` | `snapshot()` |
| `IngestionDebugPort` | `inspect_bytes()`、`parse_nodes()` |
| `RetrievalDebugPort` | `retrieve_trace()` |
| `GenerationDebugPort` | `prepare_llm_input()`、`answer_trace()` |
| `WorkflowDebugPort` | `stream_trace()` |
| `EvaluationDebugPort` | `generate_dataset()`、`preview_case()` |
| `DocumentsDebugPort` | `delete_preview()`、`rename_preview()`、`replace_preview()` |
| `DatabasesDebugPort` | `info()`、`delete_preview()`、`rename_preview()`、`recovery_preview()` |
| `QuerySessionDebugPort` | `current()`；`query(trace=True)` 复用一次检索导出阶段快照 |
| `ConversationLedgerDebugPort` | `inspect_scope()`、`inspect_versions()`；仅返回所选会话范围内的记录与计数 |
| `TaskPlanningDebugPort` | `validate_candidate()`：注入固定计划并检查 Schema、原文片段、目标映射及依赖，不调用 LLM |
| `HistoryResolutionDebugPort` | `resolve()`、`prepare_batch()`：注入固定定位与修订，检查范围、歧义、条件来源和版本，不调用 LLM |
| `EvidenceMappingDebugPort` | `map_trace()`：原检索诊断链路的逐题许可、节点引用、排名和耗时 |

会话账本的参数、幂等、错误码、迁移及阶段边界见[会话账本接口](conversation_ledger.md)。
会话账本、任务规划、历史定位、证据映射和生成已由 `ConversationWorkflow` 编排并接入 CLI/API。
当前 PostgreSQL schema 要求版本 5；历史定位阶段的版本 2 只是增量迁移来源。定位、确认/授权预检与模块边界见[历史定位与条件修订接口](history_resolution.md)。
子方案四的逐题证据契约、候选与来源状态见[逐题检索与证据映射接口](evidence_mapping.md)。

## 依赖注入端口

`build_container(settings, *, models: ModelBundle | None = None, storage: StorageBundle | None = None)`

- `models=None`：生产模式，构建 OpenAIEmbedding（LM Studio）、OpenAILike（DeepSeek）、
  SentenceTransformerRerank（本地 Qwen3）。
- `models=ModelBundle(...)`：测试模式，注入假 LLM / 假 Embedding / 假 Reranker。
- `storage=...`：注入已验证且持有占用锁的指定库，不重新打开默认集合。

`ApplicationContainer` 暴露 `settings`、`llm`、`storage`、`ingestion`、`retrieval`、
`generation`、`workflow`、`documents`、`trace_writer`、`ready`。
`ready` 为实时 Qdrant 检查属性，不再是固定布尔值；`storage.aclose()` 关闭数据库客户端并释放持有的库锁。
`task_planner` 提供任务规划，`history_resolver` 在配置 PostgreSQL DSN 时提供历史定位；未配置存储时后者为 `None`。
`evidence_mapper` 要求会话存储和注册知识库 UUID，否则为 `None`。
容器另暴露 `session_store`、`conversation_generation` 与 `conversation_workflow`；会话工作流仅在必需依赖具备时构建，并按规划、定位、证据、生成与提交顺序调用模块。应用不隐式迁移数据库，旧 CLI `query` 预览不受影响。

`DatabasesPipeline(settings, *, client_factory=..., embedding_factory=None)` 支持替换管理后端和
Embedding 工厂；`QuerySession(databases, *, models_factory=None)` 支持注入模型束。测试使用
原生 `MockEmbedding` 和假 LLM；生产 `query` 使用禁用生成的 `PreviewLLM`，`ask` 则按需读取配置并构建真实会话模型。
内部直接调用 `build_storage()` 不会自行取得跨进程锁；支持的 CLI/API 入口负责取得并转交锁。

## 命令行接口

执行位置为项目根。入门示例见[项目 README](../README.md)，管理入口和恢复边界见[文档导航](README.md#库与文档管理)及[文件管理](file_management.md)。可通过 `uv run story-rag --help` 和各子命令 `--help` 查看当前参数。

| 命令组 | 生产连接端口 | 诊断接口 |
|---|---|---|
| `db` | 创建、列库、详情、登记旧主库、改名、整库删除、恢复 | 改名/删除/恢复的 `--dry-run`；详情含 `busy` / `requires_recovery` |
| `documents` | 指定库的列表、详情、精准删除、库内 TXT 改名、正文替换 | 删除/改名/替换的 `--dry-run`，返回目标身份、ETag、节点范围 |
| `ingest` | 明确 `--db` 或 `--db-id` 后导入 TXT 或目录 | 原有解析诊断端口保持不变 |
| `shell` | `db use` 绑定库后连续 `query`；`db release` / `exit` 释放 | `db current`、`query --trace`、`query --export` |

`name` 为展示/选择名称，`db_id` 为注册 UUID，`database_id` 为物理存储指纹。
`--db-id` / `--confirm-id` 接受完整注册 UUID；HTTP `X-Database-ID` 保留指纹含义，不是切库参数。
所有独立数据命令选择参数互斥且必须明确，不存在、非 ready 或缺失配套存储时不回退其他库。

会话查询返回 `db_id`、`name`、`status`、`storage_fingerprint`、数量、`request_id`、`query`、
请求/有效 Top-K、到 Prompt 耗时、`retrieval`、`llm_input` 与 `llm_calls=0`。
顶层 `status` 在查询结果中为 `context_found` / `no_context`；`db current` 中为注册表状态。
`retrieval` 包含最终节点、RRF 候选和重排耗时，`llm_input` 包含完整 `model/messages`、Prompt、
上下文和节点；`would_call_llm` 只是正常生成时是否存在可发送上下文，不表示会话实际调用。
`--trace` 额外返回 `stages` 和 `stage_metrics`；原生回调从同一次 Vector/BM25/RRF 检索取得
快照，重排前记录融合分数，不重复检索。导出必须为父目录已存在的绝对路径，拒绝覆盖已有文件。

库内改名保存 `display_name`，不动源 TXT、文档/节点 ID 或向量；正文替换保留文档 ID、
原始 `source` 与展示名，提前分块和 Embedding 后保存完整更新记录。两者都校验强 ETag，
刷新 revision，让 BM25/RRF 下次重建；未完成更新阻止继续查询。命名库的删除/更新恢复必须
经 `db recover` 确认库 UUID 与操作 ID，只读打开不会自动恢复。

API 新增启动参数 `story-rag-api --db "名称"` / `--db-id "UUID"`，绑定集合与 docstore，
进程存续期间持有物理库锁。无参数保留旧配置，但取得相同物理库锁；不会自动登记旧主库。
旧主库一旦登记，无参数启动也遵循注册模型/schema 和禁止补建/隐式恢复的规则。
CLI 会话不会改变 API 绑定，请求体不增加选库字段；本文未新增数据库管理 HTTP 路由。
各库间可分别启动进程，同一库只允许一个支持的 API/CLI 入口持有；外部直接操作 Qdrant
不受这些锁保护，也不提供分布式事务或多用户权限。

## HTTP API

### 健康与就绪

`GET /health` 返回 HTTP 200 和 `{"status":"ok"}`，只检测进程存活。

`GET /ready` 返回模型名、集合名、`debug_enabled`、`vector_backend="qdrant"` 和
`database_ready`。数据库连通、集合 schema/状态及点数与本地节点一致时返回 HTTP 200、
`status="ready"`；失败时返回 HTTP 503、`status="not_ready"`、`database_ready=false`。
此接口不返回底层错误、连接凭据，也不进行付费 LLM 探测。

### 导入 TXT

`POST /v1/qa/ingest`，`multipart/form-data`。旧路径 `/v1/stories/ingest` 暂保留兼容，
不再出现在 OpenAPI 文档中。

字段：`file`、可选 `title`、可选 `user_instruction`。

返回：`document_id`、`source_name`、`title`、`content_hash`、`node_count`、`duplicate`、
`replaced`、`warnings`。

| 状态码 | 场景 |
|---|---|
| 415 | 扩展名不是 `.txt` |
| 413 | 文件超过 50 MB |
| 422 | 编码不是 UTF-8 BOM / UTF-8 / GB18030，或内容为空 |
| 400 | 文本无法切分为任何节点 |
| 503 | 向量化或持久化失败（LM Studio 不可用等） |

### 查询

`POST /v1/chat/query`

```json
{"query":"什么是向量检索？","top_k":5}
```

无答案时返回：

```json
{"status":"no_answer","answer":"无"}
```

### 文件管理

前端网页连接已停用；不再提供 GET /、/static/ 或 GET /v1/ui/context。
应用进程仅绑定一座知识库，操作通过 CLI 或以下业务 API 完成。
TXT 导入接口同样受本机同源门禁保护。
文件管理接口只允许本机回环客户端，
Host 必须为 `127.0.0.1`、`localhost` 或 `::1`；若有 Origin，必须与访问地址完全同源。
不接受客户端指定任意 Qdrant 地址或集合，不提供跨域授权。
这不是用户身份认证，不能作为公网或转发代理后的权限系统使用。

| 接口 | 用途 |
|---|---|
| `GET /v1/documents/{document_id}/preview?max_characters=8000` | 只读资料正文预览，最多 20,000 字符 |
| `GET /v1/documents?q=&offset=0&limit=50` | 当前库的文件目录，按文件名、来源、ID 稳定排序 |
| `GET /v1/documents/{document_id}` | 删除确认信息，响应头包含 `ETag` |
| `DELETE /v1/documents/{document_id}` | 删除一个文档，必须附带 `X-Database-ID` 和 `If-Match` |
| `GET /debug/v1/documents/{document_id}/delete-preview` | 只读节点/向量 ID 与一致性预览，仅在调试开启时注册 |

`q` 对文件名和来源做不区分大小写的子串搜索，最长 500 字符；`offset >= 0`，
`1 <= limit <= 200`。`total` 为搜索后的总数，不是当前页数量。
正文预览默认最多返回 8,000 字符，max_characters 范围为 1–20,000，包含全文字符数和
truncated 标记；不得把正文放入会话标题、诊断日志或查询参数。
目录来自持久化 LlamaIndex `Document` 与 `get_all_ref_doc_info()`，一份 TXT 一行，
不将每个 Q&A/分块列成文件，也不将自定义 title 当成文件名。
列表读取前检查集合就绪及点数；每个文件的 `status=indexed` 表示 docstore 节点映射完整，
不意味着每次列表都全量比对 Qdrant 点 ID。精确范围通过只读预览和删除前检查核对。

列表响应示例（ID 和 ETag 为示意值）：

```json
{
  "database_id": "<当前服务的数据库标识>",
  "collection_name": "story_chunks",
  "total": 1,
  "offset": 0,
  "limit": 50,
  "items": [{
    "document_id": "<64位文档ID>",
    "filename": "面试.txt",
    "source": "D:\\qa\\面试.txt",
    "node_count": 6,
    "status": "indexed",
    "etag": "\"<版本摘要>\""
  }]
}
```

详情在上述单条字段基础上增加 `database_id` 和 `collection_name`。
数据库标识由服务端地址、集合及配套 docstore 的绝对路径生成，不包含服务密钥。
客户端必须原样回传详情中的带双引号强 ETag，不支持 `*`、弱 ETag 或多个候选值。
预览还包含 `node_ids`、`qdrant_point_ids`、`consistent`，不输出原文、不执行删除。

```powershell
$base = 'http://127.0.0.1:18082'
$page = Invoke-RestMethod "$base/v1/documents"
$page.items | Select-Object filename, source, node_count, document_id | Format-Table -AutoSize -Wrap
# 先人工选定一条并核对来源；不要对 items 做批量删除。
$id = Read-Host '请输入要核对的 document_id'
$detail = Invoke-RestMethod "$base/v1/documents/$id"
$detail | Select-Object collection_name, filename, source, node_count, document_id | Format-List
Invoke-RestMethod "$base/debug/v1/documents/$id/delete-preview" | ConvertTo-Json -Depth 6
```

仅在核对上述单个文件及删除范围后，才执行删除命令；预览需要开启调试接口：

```powershell
# 仅在确认删除这一个文件后执行下一条命令。
Invoke-RestMethod "$base/v1/documents/$id" -Method Delete -Headers @{
    'X-Database-ID' = $detail.database_id
    'If-Match' = $detail.etag
}
```

成功返回 HTTP 200，字段为 `database_id`、`collection_name`、`document_id`、`filename`、
`deleted_nodes`、`status="deleted"`、`source_file_deleted=false`。
删除范围为该文档全部 Q&A/分块向量、docstore 原文及节点、内存 IndexStruct；
增加存储 revision，使下一次查询重建 BM25/RRF。源 TXT、其他文档、历史报告和日志不删除。
同名但不同来源通过既有 Document ID 区分，不使用文件名模糊匹配。
上传只保留上传接口收到的文件名，不能据此还原客户端绝对路径；相同上传名仍遵循既有覆盖规则。

| 状态码 | 场景 |
|---|---|
| 403 | 非本机客户端、非法 Host 或跨源请求 |
| 404 | 文档不存在，或重复删除已删除文档 |
| 409 | 当前库与确认时的数据库标识不同 |
| 412 | 文件内容/元数据/节点版本已变化，或 ETag 格式不受支持 |
| 422 | 列表参数不合法 |
| 428 | 缺少 `X-Database-ID` 或 `If-Match` |
| 503 | 存储不可用、节点范围不一致或删除尚未完成 |

删除过程先持久化 `docstore.delete.json` 恢复记录，再调用原生
`VectorStoreIndex.delete_ref_doc(..., delete_from_docstore=True)`，检查残留，原子替换
JSON docstore，最后移除恢复记录。中途失败不返回成功，并阻止该进程继续检索/入库/管理。
未登记的旧配置库仍在重启时恢复未完成删除；已登记库需显式 `db recover` 后才能打开，
然后执行正常集合/docstore 一致性校验。损坏或不匹配的恢复记录
不会强行应用，需人工检查日志及成对备份。不要手动移除恢复记录以绕过保护。
此机制不是跨存储事务，也不是回收站；一般入库失败仍需检查并恢复匹配存储。

单进程共享锁协调入库、删除与完整检索阶段，不跨 LLM 生成持锁。
同一库不支持多个 API/CLI 进程或多 worker，共有物理锁在入口拒绝并发持有；
服务运行期间应通过同一服务导入，或先正常停止服务再用 CLI。删除成功后的新检索不会返回目标节点，已经组装好的在途上下文
不会被追溯撤回。目录和删除本身不调用 LLM、Embedding、Reranker；正常应用启动仍需本地模型。

命令行验证方法与隔离规则见 [文件管理验证说明](file_management.md)。

### 诊断

调试接口仅在 `DEBUG_ENABLED=true` 时注册。响应不包含 API Key、请求头或模型凭据。

| 接口 | 用途 |
|---|---|
| `GET /debug/v1/config` | 非敏感配置快照（不含密钥） |
| `POST /debug/v1/retrieval/trace` | 返回向量 / BM25 / RRF / Rerank 四段候选与分数 |
| `POST /debug/v1/generation/input` | 返回最终 Prompt、chat messages、上下文和节点，不调用 LLM |
| `POST /debug/v1/generation/trace` | 返回 Prompt、原始回答、门控 Prompt 与门控原始输出 |
| `POST /debug/v1/ingestion/inspect` | 对请求体文本返回解码信息与分块结果，不落库 |
| `POST /debug/v1/workflow/trace` | 返回检索与生成两个事件的完整快照 |

配置快照删除 `chroma_dir`，新增 `vector_backend`、`qdrant_url`、`collection_name`。
`docstore_path` 保留；Qdrant API Key 不进入快照。节点 ID 由旧 SHA-256 改为稳定 UUID v5，
Document ID 保持不变。向量候选 `score` 为 `1/(1+Euclid distance)`，越大越相关；
RRF 和 Rerank 分数定义不变，不能将此向量分数与旧 Chroma 数值直接比较。

离线诊断脚本：`uv run python scripts/verify_local_models.py`（检查 LM Studio Embedding 与本地 Reranker，不调用 DeepSeek）。

Q&A 全链路基准：`uv run python scripts/run_qa_benchmark.py --output-dir reports/qa_v14d1_run`。
该脚本使用拒绝生成的 LLM 替身，查询融合固定为一条原始 Query；Prompt 与 messages
由生产代码的 `prepare_llm_input()` 组装，结果写入本地 `results.jsonl`。
诊断接口在 `QUERY_FUSION_NUM_QUERIES` 不为 `1` 时返回 409，以保证预览不会触发
Query 改写的 LLM 调用。

## 多轮询问端口

启用条件：服务绑定已登记知识库，并配置 `STORY_RAG_POSTGRES_DSN`；迁移需显式执行
`uv run python scripts/migrate_conversations.py --apply`。会话固定绑定知识库 UUID 和
存储指纹；这些本地回环接口不提供公网用户鉴权。
`POST /v1/chat/query` 是会调用生成链路的单轮问答；CLI `query` 才是零生成检索预览。多轮业务使用下列会话接口。

| 生产接口 | 作用 |
| --- | --- |
| `POST /v1/conversations` | 以 `{"name":"会话名"}` 创建长期会话 |
| `GET /v1/conversations` | 列出当前绑定库的未归档会话 |
| `POST /v1/conversations/{session_id}/turns` | 规划、定位、检索、生成、逐题门控和提交一轮 |
| `GET /v1/conversations/{session_id}/questions` | 读取稳定题号及当前版本 |
| `GET /v1/conversations/{session_id}/questions/{question_id}/versions` | 读取该题版本链 |
| `POST /v1/conversations/{session_id}/archive` | 附 `expected_revision` 归档会话；隐藏历史但保留账本 |
| `POST /v1/conversations/{session_id}/index/recover` | 幂等补齐历史派生索引 |
| `POST /v1/conversations/{session_id}/requests/{request_id}/acknowledge-unknown` | 显式确认 unknown 请求结果仍未知，解除阻断；不会重发原请求 |

已实现展示读取端口：GET /v1/conversations/{session_id}/turns?limit=50&cursor=...。
返回 items 和 next_cursor，每项包含原始用户消息、可空的最终可见回答、请求状态、
失败码、轮次顺序及时间；limit 范围为 1–100，cursor 使用上一页的轮次 UUID。
完整表结构、失败轮次、幂等与验收契约见
[对话展示记录与预留端口](对话展示记录与预留端口_20261028.md)。
展示正文由独立的定向历史读取端口用于历史定位和受控 Prompt 组装。此端口需要 PostgreSQL
迁移版本 5；当前生产库不会自动迁移，使用前须明确执行迁移。

提问请求示意：

```json
{"request_id":"client-unique-001","expected_revision":0,"message":"请分别回答这三题","use_history":true,"reply_to_question_id":null}
```

`request_id` 与完整请求摘要、`expected_revision` 共同决定幂等；相同 ID 和内容重放不再次生成，
相同 ID 配不同内容返回冲突。`reply_to_question_id` 仅作为默认历史目标，不能与
`use_history=false` 同用。响应包含 `status`（`succeeded/failed/unknown`）、
`request`、`revision`、`answer`、按输入任务顺序排列的 `items`、`error_code`、
`answer_not_retained` 和摘要 `trace`。每项有题目 ID/版本、状态、答案及引用；
`waiting_input`、`no_evidence`、`technical_failure` 不混同。
确定失败或结果未知作为响应状态返回，不表示已提交新 revision；重放已提交请求时，
API/CLI 展示层从独立展示表读取最终可见回答。迁移前旧轮次若缺少正文，
`answer_not_retained=true`，不伪造旧回答。

历史原文回读补丁说明和六类场景验收见
[历史原文读取补丁与验收](历史原文读取补丁与验收_20261031.md)。请求体未增加必填字段；
规划器可选择 read_user、read_answer、transform 或 followup，然后程序按会话、知识库、
as-of revision 与轮次/题号读取。响应 item 可含 response_kind 和 history_refs。原样复述
不走知识生成/门控；格式转换仍执行生成及历史忠实性校验；新增事实仍需当前知识证据。
读取失败、未保留正文、歧义和缺失分题映射各有独立状态，不会伪装成知识库无答案。
普通自然语言 ask 仍会调用规划模型。

多题的原始用户消息没有可靠的逐题字节区间，因此用户原话读取返回整轮原文；旧助手答案
没有可靠逐题映射时，单题读取要求返回整轮或澄清。新产生的逐题答案保存字符区间和哈希
于该题版本 JSONB，复述/格式转换创建独立题目记录，不改变来源题状态。旧回答是历史记录，
不是当前事实证据。

诊断接口 `POST /debug/conversations/{session_id}/preview` 仅在
`DEBUG_ENABLED=true` 时提供，输入为 `message`、已验证的 `plan` 和按题对齐的
`task_inputs`，输出计划、每批保守预算、题数、最终生成 messages；不调用 LLM、
不检索或提交。生产 `turns` 的内部诊断可通过受控测试端口注入 Mock，检查阶段
trace、失败码、usage 与答案数量；完整 Prompt 和原文只按需保存于忽略 Git 的测试报告。

CLI：`uv run story-rag shell` 后使用 `db use "库名"` → `conversation create "会话名"`
或 `conversation use UUID` → `ask "消息"`。`ask --question Q001` 为明确引用，
`ask --no-history` 完全关闭历史读取，二者冲突时拒绝。`question list/show/history`
展示稳定题号；`conversation index recover` 只补索引，`conversation recover ID
--acknowledge-unknown` 仅用于人工确认 unknown 状态后的恢复。

CLI 首次 `ask` 创建会话工作流时，从选定配置的 `api_file` 读取真实模型名和密钥，
只替换 LLM 字段，保留知识库绑定与 PostgreSQL 配置；已有工作流不热加载凭据，
修改配置后需退出并重新启动 CLI。列库、文档管理和 `query` 不因此要求生成密钥。
规划调用收到明确的 HTTP 401 鉴权拒绝时，返回 `status=failed`、
`error_code=LLM_AUTHENTICATION_FAILED`，不增加会话 revision；纠正配置后可使用新请求 ID
重试。结果不确定的超时仍返回 `unknown` 并保留会话阻断。旧版已记录为 unknown 的请求
不会被自动改写，需核对具体请求后通过上述恢复命令解除；恢复本身不调用模型。

2026-10-26 起，多轮会话中的计算类子题仅检索已有 Q&A，不执行计算。
HTTP/CLI 请求格式不变；无适用已有答案时该题为 `no_evidence/无`，不影响其他题。
内部 `answer_mode=calculation_lookup`、`qa_match` 调用计量与预览诊断详见
[计算类子题仅检索补丁说明](多轮询问模块重建子方案/计算类子题仅检索补丁说明_20261026.md)。
