# 评估说明

> 2026-11-06 发布说明：本文区分无付费模型的程序回归、真实本地模型测试与可能计费的生成/评估。旧基准结果仍按原日期解读，本次没有重新运行质量验收。

评估链路使用 LlamaIndex 原生组件：

- `DatasetGenerator`
- `BatchEvalRunner`
- `ContextRelevancyEvaluator`
- `CorrectnessEvaluator`
- `RelevancyEvaluator`
- `FaithfulnessEvaluator`

基准数据集应生成后冻结，再复用同一输入比较中文、英文和中英混合场景。当前副本不提供原机 `data/`、`reports/` 或外部基准语料，不能仅下载代码就重放全部历史实验。

## 生成器选择

`llama_index.core.evaluation.DatasetGenerator` 在 0.14 已被标记为 deprecated，
官方后续推荐 `RagDatasetGenerator`。V0 按需求继续使用 `DatasetGenerator`，
升级 LlamaIndex 时需要同步迁移生成器与数据集格式。

## 运行方式

```powershell
# 生成中英双语测试集（调用 api.txt 中的 LLM）
uv run story-rag-eval generate --path "D:\stories" --output reports\dataset_v0.json

# 批量评估并输出报告
uv run story-rag-eval run --dataset reports\dataset_v0.json
```

报告写入 `reports/evaluation_report.json`，按评测器和语言分别统计均分与通过数。

上面两条命令均可能调用远端模型并产生费用；必须自行准备正确目标和凭据。配置方式见[项目 README](../README.md)。本次文档维护没有执行这些命令。

## 回归测试

- `uv run pytest -m "not integration" -q -p no:cacheprovider --basetemp reports\evaluation_tests_new`：程序与契约回归，模型可使用替身；存储测试会使用隔离 Qdrant/PostgreSQL。basetemp 必须是尚不存在的新路径，不能指向业务数据。
- `uv run pytest -m integration -q`：需要实际依赖的集成用例。普通本地模型用例检查 LM Studio/Reranker，另有显式开关控制的真实会话用例；开启后可能调用付费模型，不能将整个 integration 标记理解为零费用。
- 生成测试集的用例标记为 `evaluation`，使用脚本化 LLM，不产生外部费用。

## V1.4 冻结 Q&A 基准

使用 Agentic RAG 的 `benchmark_v14_20260918/cases.jsonl` 中的 128 题，
对应 `D:\text_Project\找面经八股\raw` 中的 133 个 TXT。运行：

这里的路径是原实验环境，不是发布副本预置文件；先准备同一冻结数据并检查脚本的输入参数。缺少原语料或题集时不能把运行失败当成检索质量失败。

```powershell
uv run python scripts\run_qa_benchmark.py --output-dir reports\qa_v14d1_run
```

基准创建独立 Qdrant 集合/docstore，走 TXT 入库、LM Studio Embedding、向量与 BM25、
RRF、Qwen3 Reranker、Top-5 上下文和最终 Prompt 组装。生成模型被替身拦截，
`llm_generation_calls` 必须为 0。每题完整输入位于 `results.jsonl` 的
`llm_input.prompt` 和 `llm_input.messages`，另导出无参考答案的 `llm_inputs.jsonl`。

原基准的 `expected_chunk_id` 属于 Agentic RAG 的分块策略，不能直接与本项目节点 ID 比较。
这里分别统计参考答案全文命中和来源文档命中，且按阶段输出 Hit@1/3/5。

## 数据库替换回归

新增数据库 schema/距离、UUID 稳定性、同步/异步共享、跨事件循环、RRF 距离方向、
重复导入、替换删除、重启恢复、维度不兼容、集合丢失、节点数量不一致与就绪降级测试。
测试夹具自动启动临时 Qdrant；必须先安装项目内原生发行包，或设置专用 `QDRANT_TEST_URL`。
基准保存 backend、集合名、距离和实际维度，报告目录与历史 Chroma 结果隔离。
GPU 重排延迟不能与历史 CPU 延迟直接归因比较；这 128 题属于入库语料，不代表独立问题准确率。

## 新语料真实检索与资源测试

2026-10-13 采用 `D:\data\nowcoder_raw` 排除产品、机械后的 174 个 TXT，
执行原冻结 128 题和新语料 128 道改写题。详细口径与统一报告入口见
[qa_benchmark_history.md](qa_benchmark_history.md)。

新增 128 题均有可追溯的完整参考节点；旧题仅 22/128 的参考答案可在新语料逐字确认，
其余题不计入可判分分母，不当成无答案检测成功。
使用原生 `HitRate` / `MRR`，同时保存 Vector、BM25、RRF、Rerank 的 Hit@1/3/5。

```powershell
.venv\Scripts\python.exe -m scripts.run_nowcoder_benchmark --output-dir reports\qa_history_new
```

采用一次生产检索和原生回调，输出最终 LLM 输入，不调用生成、门控或 LLM 评估。
`DatasetGenerator`、`BatchEvalRunner` 与四个生成质量 Evaluator 保留在原评估模块，
本轮因没有生成答案，不报告 correctness、faithfulness 等生成质量分数。
单题计时到 Prompt；按设置间隔采样工作集、提交内存及整卡显存，实际间隔包含测量开销。
本轮实际间隔 P50 为 4.302 秒，报告不得当成严格 1 Hz 采样，另记录显存缺测数。
历史 JSON 只复制不覆盖；旧报告未测内存，且语料/计时不同，不直接计算加速比。
