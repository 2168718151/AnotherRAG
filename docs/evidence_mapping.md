# 逐题检索与证据映射接口：子方案四

> 当前实现核对：2026-11-06。证据映射已接入生产多轮工作流；下述模块测试的判定范围不因集成而扩大。

## 阶段边界

本模块是独立 Python 适配器，由 `ConversationWorkflow` 调用并连接后续生成/提交，CLI `ask` 与会话 API 使用同一生产编排。
上游提供已核验的题目版本、对象版本与当前请求；证据映射本身不负责拆题、历史定位、答案生成或直接提交 PostgreSQL。
它复用 Qdrant + Docstore、向量/BM25/RRF/Reranker 检索，不新增索引；原单轮 `query` 预览接口不变。

## 生产端口

`EvidenceMappingPipeline.map_evidence(scope, request, payload, tasks) -> EvidenceBatch` 对每个可执行任务生成独立检索文本并调用现有 `StoryRetrievalPipeline.retrieve()`。

| 参数 | 含义 |
| --- | --- |
| `scope` | `HistoryScope`：会话 UUID、知识库 UUID/存储指纹、请求开始的 revision |
| `request` / `payload` | 已认领请求与本轮原文；用于核验摘要、revision 和本轮事实摘录 |
| `tasks` | 逐题 `EvidenceTaskInput`：稳定题目版本、对象版本、可选可信文档 allowlist、旧证据及本轮事实摘录 |

调用前核验注册库 UUID、物理存储指纹、会话 revision、请求摘要和对象引用。不匹配抛出 `DATABASE_MISMATCH`、`STALE_REVISION`、`REQUEST_MISMATCH`、`OBJECT_MISMATCH` 等错误，不回退默认库。可选 `allowed_document_ids` 必须由可信服务端策略/来源选择器提供，不能直接信任用户或模型输出；缺省表示允许在绑定库内召回候选，**不表示候选相关或充分**。

检索文本由有效题目文本、对象条件和题目要求组成；不包含原文审计记录、整段会话历史或旧助手回答。相同查询在本批次复用一次检索结果，各题仍独立检查 allowlist，并保留自己的排序和 `allowed_evidence_ids`。检索及核验期间持有现有存储锁，不承诺并行提速。

候选须匹配 Docstore 文档/节点、`content_hash`、候选正文及 Qdrant 点 ID。`EvidenceRef` 含文档 ID/版本和节点 ID/正文哈希；本轮 `evidence_pool` 按知识库/文档版本/节点/正文哈希去重正文。同一节点可服务多题，但不能因此让某题引用其他题独有节点。旧对象版本不同或旧文档/节点已变化时，标记 `prior_evidence_status=needs_refresh`；新检索仍可得到有效候选，旧引用不构成已验证答案。

`TaskEvidence.decision` 为 `candidates`、`no_candidates`、`waiting_input`、`needs_refresh`、`cancelled` 或 `technical_failure`。只有 `candidates` 带 `sufficiency=unverified`；召回与重排分数不是充分度、准确率或自动答复许可。`waiting_input` 不检索；`no_candidates` 不等于最终 `no_evidence`；技术故障不伪装成无依据。每题有查询、排名分数、总耗时、原检索及 rerank 耗时和失败码。默认日志不记录正文或 Prompt。

第五轮直接提供的调用记录以本轮事实摘录进入该题 `user_facts`，摘录须逐字存在于请求并标注 `current_fact_turn_id`，**不进入知识证据池**。逐字存在不自动证明 `field` 标签的语义正确，后续仍须校验。第一轮工单日志未给出时为 `waiting_input`，不得按知识库相似内容猜测具体失败层。

## 诊断与测试

`EvidenceMappingPipeline.map_trace(...)` 使用原检索器的 `retrieve_trace()`，仅显式诊断时获取向量、BM25、RRF 阶段候选。与生产端口共用身份与版本核验，不调用生成模型。原检索诊断可能写入 `logs/traces/` 的节点和评分，须按敏感数据保留策略使用。

`tests/test_evidence_mapping.py` 使用方案 0 五轮原文、冻结上游版本、控制的权威节点和 Mock 检索器，检查缺日志不诊断、旧证据刷新、逐题过滤、去重、事实来源、身份/revision 拒绝、节点失效及技术故障；另用隔离 Qdrant 和 Mock 模型验证节点元数据。
这些模块测试不测真实 DeepSeek 的历史关联或最终生成质量；生产编排是否调用该模块可从源码核对，质量结论仍须单独验收。
