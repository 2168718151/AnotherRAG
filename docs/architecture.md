# 架构说明

> 当前实现核对：2026-11-06。安装和操作入口见[项目 README](../README.md)，文档状态与历史口径见[文档导航](README.md)。

## 库选择与生命周期

命名库由注册 UUID 绑定独立 Qdrant 集合和 JSON docstore；名称不参与物理路径。
CLI 独立数据操作明确选库，查询会话通过 `db use` 先绑定再连续查询；API 在启动时固定绑定。
支持的入口持有跨进程物理库锁，同一库只允许一个持有者，不提供外部 Qdrant 写入保护或分布式事务。
库创建/删除保留生命周期记录，文件更新保留可重放输入；已登记库打开不补建缺失存储，
待完成文件删除/更新需显式恢复。完整命令与边界见[命令行接口](interfaces.md#命令行接口)。

## 入库链路

1. 读取 TXT 字节并按 UTF-8 BOM、严格 UTF-8、GB18030 顺序解码。
2. 一个 TXT 文件映射为一个 `Document`，原文全文持久化，文件名作为默认标题。
3. 有有效 `问题：/答案：` 对时只为问答块建检索节点；长块由 LlamaIndex
   `SentenceSplitter` 切分。无问答对时回退到全文 `SentenceSplitter`。
4. 调用 LM Studio 的 `text-embedding-qwen3-embedding-0.6b` 生成向量。
5. 通过官方 `QdrantVectorStore` 的薄适配层和 LlamaIndex document store 持久化。
   Qdrant point ID 使用稳定 UUID v5，原 Document ID 不变，便于按 `doc_id` 删除。

## 检索链路

1. `VectorIndexRetriever` 在 Qdrant 上执行语义检索；同步和异步客户端共享同一服务及集合。
2. `BM25Retriever` 使用支持 CJK 的 token pattern 执行关键词检索。
3. `QueryFusionRetriever(mode="reciprocal_rerank")` 融合结果。
   默认只使用原始 Query；显式设置 `QUERY_FUSION_NUM_QUERIES>1` 时才调用 LLM 改写。
   集合使用 Euclid 距离。`EuclidQdrantVectorStore.parse_to_query_result()` 将距离转换为
   `score = 1 / (1 + distance)`，避免官方 RRF 按分数降序排列时颠倒向量排名。
   该公开转换入口同时覆盖同步/异步查询，不重写检索或融合算法。
4. 本地 `Qwen3-Reranker-0.6B` 通过 `SentenceTransformerRerank` 重排。
5. Top-K 节点作为最终 Prompt 的上下文。

## 生成链路

1. `prepare_llm_input()` 组装 Prompt、chat messages、上下文和节点快照；此步骤不调用 LLM。
2. 正常问答接口有上下文时才调用 DeepSeek `deepseek-flash`。
3. 第二个 LLM 判定回答与 Query 是否相关，只接受 Yes 或 No。
4. Yes 返回答案；No 或无上下文时固定返回“无”。

## 依赖注入点

`build_container(settings, models=ModelBundle(...), storage=...)` 可注入模型与已绑定存储：
常规生产构建真实模型，测试可注入假 LLM、假 Embedding、假 Reranker。
库管理和查询会话分别有独立工厂注入点，不因列库或文件改名而加载完整模型。
CLI `query` 使用禁用生成的 `PreviewLLM`，单 Query 检索后仅组装最终输入，不运行生成链路。
CLI `ask` 在首次构建会话工作流时读取选定的 API 配置，创建真实会话模型；两者不能混为同一条调用链。

## 多轮会话与历史正文

1. 当前消息经任务规划、来源与结构校验，形成逐题任务及历史查询意图。
2. 会话绑定、数据库身份、as-of revision 与题目/对象版本共同限制历史定位和条件修订。
3. 知识任务按当前有效条件执行证据映射，复用现有混合检索与重排；历史原文任务走定向正文回读。
4. `ConversationAnswerPipeline` 按容量预算生成并执行逐题门控；工作流合并业务结果与状态。
5. `PostgresSessionStore` 通过短事务提交请求状态、题目/对象版本和独立展示记录。模型调用期间不保持跨远端请求的数据库事务。

当前 PostgreSQL 会话 schema 为版本 5。用户原话与最终成功返回的回答在展示表中持久化，
历史补丁可在严格范围内回读这些正文；旧轮次未保存的正文不能从哈希或引用补回。
历史助手回答和知识库证据分开处理；原样回读无需再次生成正文，但 `ask` 仍可能调用规划模型。
具体行为见[会话账本](conversation_ledger.md)和[历史原文读取](历史原文读取补丁与验收_20261031.md)。

## 入库后的检索刷新

`StorageBundle` 维护 `revision` 计数器：入库、改名、替换或删除文档后自增。
`StoryRetrievalPipeline` 在检索前比较 `revision`，变化时重建 BM25 与融合检索器，
保证服务运行期间新导入的 TXT 立即可被检索。

分块节点通过 `store_nodes_override=True` 写入项目自有 docstore，供 BM25 和原文恢复使用；
向量检索从 Qdrant payload 恢复节点。两边来自相同入库节点，启动时校验 ID 配对，
备份与恢复必须同时保留两份存储。

## Qdrant 存储边界

- `enable_hybrid=False`，不启用 Qdrant 稀疏向量，保留现有 BM25 和 RRF。
- 启动时先检查服务，再通过真实 Embedding 探测维度；新集合显式使用 `text-dense` 和 Euclid。
- 已有集合维度/距离不匹配、docstore 缺少对应集合、节点数量或 ID 不一致时拒绝启动；不自动删库。
- `/ready` 实时检查数据库连通、集合 schema、状态和精确节点数量，异常返回 HTTP 503。
- `/health` 只表示 HTTP 进程存活；`/ready` 不探测 DeepSeek 或 Reranker 的运行时可用性。
- Qdrant 和 docstore 不是跨存储事务。备份/恢复必须配对，多个 API 进程不能共写同一 JSON docstore。
- API lifespan 和 CLI/基准退出时关闭同步、异步客户端。

## 日志

应用和 LlamaIndex 均不使用控制台作为主要日志出口。日志由 `logging_config.py` 写入：

`logging_config.silence_third_party_console()` 设置日志环境变量、关闭 transformers 进度条，
并为 transformers / sentence-transformers / huggingface_hub 添加 `NullHandler`，
避免第三方库通过 logging 的 last-resort 通道向控制台输出；记录仍会传播到文件 handler。

- `logs/app.log`
- `logs/llama_index.log`
- `logs/error.log`
- `logs/traces/*.jsonl`

密钥字段使用受限配置快照，业务代码不应将 API Key 或 DSN 写入 payload。
目前日志处理器与 TraceWriter 没有统一密钥脱敏过滤器，不能把“不在快照中暴露密钥”当作任意日志内容都已自动脱敏。

## 发布与访问边界

当前提供 CLI 和业务 API，不挂载首页、静态网页或 `/v1/ui/context`；静态文件仅作为参考稿。
发布副本保留必要 Windows 数据库程序，但不带原机知识库、历史会话、模型权重或真实凭据。
应用按本机访问设计，不提供完整公网多用户鉴权与分布式一致性保障。
