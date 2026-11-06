# Qdrant 替换与 V1.4 冻结基准验证

> 历史资料说明（2026-11-06）：以下设计、实施过程、基线与验收结果按文中原日期保留，不作为当前功能状态或可直接执行的部署步骤。当前说明见[文档导航](README.md)。本次未重新运行原验收，也未把历史失败改成通过。

## 结论

入库和检索向量存储已从 Chroma 替换为 Qdrant；Q&A 解析、JSON docstore、BM25、
原生 RRF、Qwen3 Reranker 与最终 LLM 输入组装保持原链路。
完整测试 78 passed、0 skipped；冻结 128 题四个阶段的参考答案排名与旧 Chroma 报告逐题一致。
本次没有调用 DeepSeek 生成或 LLM 评估，不代表最终生成回答的正确率。

## 环境与实现

- Qdrant 服务：官方 Windows 原生 1.19.2，项目内安装，localhost HTTP/gRPC。
- Python Qdrant Client：1.19.1；LlamaIndex Qdrant 集成：0.10.4。
- CUDA PyTorch：2.14.0+cu130，CUDA 可用，GPU 为 RTX 4060 Laptop GPU。
- Embedding：LM Studio Qwen3 Embedding 0.6B，实际返回 1024 维。
- 距离：Euclid；融合前转换为 `score=1/(1+distance)`，兼容官方 RRF 降序排序。
- 节点 ID：稳定 UUID v5；Document ID 与按文档删除语义不变。
- 旧主库为空，不进行旧数据迁移；旧 Chroma 与历史报告未删除或覆盖。

## 自动化与运行验证

```powershell
uv sync --frozen --group dev
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp reports\qt03 --junitxml reports\pytest_qdrant_20261011_final.xml
.venv\Scripts\python.exe scripts\run_qa_benchmark.py --output-dir reports\qa_v14d1_qdrant_20261011_01
```

- Ruff 和 `git diff --check` 通过。
- 完整测试：78 passed、5 warnings、0 skipped，耗时 129.28 秒。
- Warning 为既有 Starlette/httpx 与 DatasetGenerator/QueryResponseDataset 弃用提示。
- 覆盖 UUID、实际维度/距离、同步/异步共享、跨事件循环、RRF 排名方向、重复导入、
  替换删除、docstore 重载、集合缺失、错误维度/距离、点数及节点 ID 配对等。
- `/ready` 对数据库故障或集合丢失返回 503；主集合健康冒烟返回 200，仍保持空库。
- 在冻结基准结束后重启本机 Qdrant，再加载配套 docstore，恢复 133 个文档和 1056 个节点；
  向量、BM25 和 RRF 各返回 30 个候选，最终输入可组装，数据库就绪检测通过。
  此次恢复冒烟使用真实 Embedding、禁用生成的 LLM 和假 Reranker，不是再跑一次完整基准。

## 冻结基准

语料为 `D:\text_Project\找面经八股\raw` 的 133 个 TXT，测试为 Agentic RAG
`data/benchmark_v14_20260918/cases.jsonl` 的 128 题。
运行时间为 2026-10-11 23:34:52 至 23:53:17（Asia/Shanghai）。

| 阶段 | 参考答案 Hit@1 | Hit@3 | Hit@5 | 来源 Hit@5 |
|---|---:|---:|---:|---:|
| Vector | 128/128 | 128/128 | 128/128 | 128/128 |
| BM25 | 126/128 | 128/128 | 128/128 | 128/128 |
| RRF | 127/128 | 128/128 | 128/128 | 128/128 |
| Rerank | 128/128 | 128/128 | 128/128 | 128/128 |

与 `reports/qa_v14d1_20261009_01/results.jsonl` 按 case ID 比较，
四个阶段的参考答案排名变化数量均为 0。
检索诊断链路 P50 为 8.21 秒、P95 为 10.88 秒；历史为 CPU 重排，本次为 CUDA 环境，
且测试时存在其他 GPU 活动，不能将延迟变化直接归因于数据库替换。

## 输出与完整性

报告目录：`reports/qa_v14d1_qdrant_20261011_01/`。
基准集合：`qa_benchmark_c3614545fa41ffb90102`，与空的主集合 `story_chunks` 隔离。

- `summary.json`：命中率、模型/数据库与数据集元信息。
- `results.jsonl`：128 题完整四阶段候选、分数、参考指标和最终 Prompt/messages。
- `llm_inputs.jsonl`：128 个唯一 case ID 对应的最终 `model` 与 `messages`。
- 所有导出 messages/model 与逐题生产代码组装结果完全一致，没有单独的标准答案字段。
- 每题 Top-5 上下文和 Prompt 均已组装，生成调用为 0；LLM 替身任何生成调用都会报错。
- `logs/error.log` 为空，应用和 LlamaIndex 日志仍只写入文件。
- `qdrant_before_restart.log` 保存服务重启前的运行日志。

这 128 题来自已入库语料，参考答案可在源文本中精确出现；本次结果用于数据库替换回归，
不代表独立问题、无答案问题或生产环境的回答准确率。Qdrant 和 JSON docstore 仍需成对备份，
且没有跨存储事务或多进程共写保障。
