# 文档导航与当前状态

核对日期：2026-11-06。本文按发布副本的当前源码整理；首次运行从[项目 README](../README.md)开始。

## 当前功能

- TXT 入库、命名知识库、文档管理、向量/BM25/RRF 混合检索与本地 Reranker。
- CLI `query` 仅检索并预览模型输入；`ask` 与 HTTP 问答接口可调用真实模型。
- PostgreSQL 会话 schema 版本 5；多轮规划、历史定位、证据映射、生成/门控已接入 CLI/API。
- 用户原话和成功返回的可见回答保存于独立展示表。历史正文回读已有实现，不将旧助手回答直接视为知识证据。
- API 在启动时绑定一座库；当前不托管首页、静态网页或网页专用上下文接口。

以上是实现范围，不能替代模型质量、业务准确率或生产部署验收。

## 当前使用与接口

| 文档 | 主要内容 |
| --- | --- |
| [项目 README](../README.md) | 环境准备、模型配置、知识库导入、query/ask 与 API 启动 |
| [架构](architecture.md) | 存储、单轮检索/生成、多轮编排和日志边界 |
| [接口](interfaces.md) | CLI、HTTP、Python 端口、身份与错误语义 |
| [Qdrant](qdrant.md) | 启动、配置、存储一致性与恢复边界 |
| [文档管理](file_management.md) | 列表、预览、修改与精准删除 |
| [会话账本](conversation_ledger.md) | PostgreSQL、schema v5、请求幂等与展示记录 |
| [任务规划](task_planning.md) | 当前消息规划与诊断校验 |
| [历史定位](history_resolution.md) | 身份、条件修订、历史范围与版本 |
| [证据映射](evidence_mapping.md) | 逐题检索、证据许可与版本核查 |
| [历史原文读取](历史原文读取补丁与验收_20261031.md) | 原话/原答回读、转换与追问的实现和验收范围 |
| [评估](evaluation.md) | 本地测试、模型调用与外部语料前置条件 |

## 库与文档管理

在项目根目录执行 `uv run story-rag --help` 查看当前 CLI；命令组包括 `db`、`documents`、`ingest`、`shell`。数据操作显式指定 `--db` 或 `--db-id`；交互会话用 `db use` 选库。

```powershell
uv run story-rag db list --json
uv run story-rag documents list --db "学习资料"
uv run story-rag db info --db "学习资料" --json
```

删除、改名、替换和恢复先使用相应 `--dry-run`，再核对库 UUID、Document ID、ETag 或操作 ID。完整语义见[命令行接口](interfaces.md#命令行接口)及[文档管理](file_management.md)。`db recover` 是继续未完成操作，不是撤销删除。

## 历史方案与验证记录

`releases/`、带日期的设计/报告以及[多轮子方案](多轮询问模块重建子方案/README.md)保留当时的版本与验收结论。历史文件中的“尚未接入”、旧 schema、旧分支、失败项或当时服务状态只适用于所注明阶段，不能当作当前运行配置，也不因本次文档更新自动改判为通过。

部分固定回归用例位于这些 Markdown 文件中并被测试读取；不改动其业务原文。

## 发布副本未包含的资源

- 原机 `data/`、`logs/`、`reports/`、虚拟环境、模型、真实凭据与 Git 历史未随副本提供。
- 原报告的本地文件引用会保留为路径说明，并注明“未随副本提供”，不伪装为可访问链接。
- 原机三份根目录中文说明及辅助规则文件当前不在发布副本；使用本导航和现有接口文档，不自动恢复已删除文件。
- 历史 Git 标签命令仅适用于保留原仓库历史的环境；本副本不能据此假定本地存在旧标签。
- 根目录 `COPY_MANIFEST.json` 与 `COPY_VALIDATION.json` 是首次复制时的快照，不代表后续文档更新后的重新验收结果。

运行、安装和迁移以当前 README 为入口。`.env.example` 不会自动加载；真实凭据仅保存在被忽略的本机配置或环境变量中。
