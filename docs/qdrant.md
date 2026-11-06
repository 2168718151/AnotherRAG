# Qdrant 部署与替换说明

> 2026-11-06 发布说明：当前副本保留 `tools/qdrant/qdrant.exe`，数据、日志与下载 ZIP 不随副本提供。首次运行见[项目 README](../README.md)，历史替换背景不表示新副本已含原机数据。

## 替换范围

向量存储由 Chroma 替换为官方 LlamaIndex `QdrantVectorStore`。
JSON docstore 仍负责原文和 BM25 节点；BM25、RRF、Qwen3 Reranker、生成/门控链路不更换。
原替换验证时，旧主库为 0 个向量且 docstore 为空，因此当时没有执行旧数据迁移。
旧 `data/chroma/` 和历史报告仅留在原机；`story_chunks` 是默认配置名，不代表副本包含该集合或任何原机业务库。

## Windows 原生部署

```powershell
.\scripts\install_qdrant.ps1
.\scripts\run_qdrant.ps1
.\scripts\run_api.ps1
```

安装脚本下载固定的官方 Windows x64 发行包，校验官方 SHA256，再解压到
`tools/qdrant/`。不安装系统服务、不修改系统 PATH、不安装 Docker/WSL。
启动脚本仅监听 localhost；数据位于 `data/qdrant/`，进程日志位于 `logs/qdrant.*.log`。
启动输出给出 PID 和 `Stop-Process` 命令；停止前先停止 API 和入库写入。
重启使用同一启动脚本，已有存储不会被清理。HTTP 端口默认 6333，gRPC 端口为 HTTP 端口加 1。
本发布副本允许必要 Qdrant 程序进入 Git；下载 ZIP、数据库内容和日志仍被排除。
Windows 原生服务的存储路径和集合名应保持简短，深层 payload 索引可能触及路径长度限制。
测试使用短临时目录和短随机集合名，不修改系统注册表。

## 启动时配置

| 环境变量                     | 用途                              |
| ------------------------ | ------------------------------- |
| `QDRANT_URL`             | 服务地址，默认 `http://127.0.0.1:6333` |
| `QDRANT_COLLECTION`      | 主集合名，默认 `story_chunks`          |
| `QDRANT_API_KEY`         | 可选服务鉴权，不进入日志或诊断快照               |
| `QDRANT_TIMEOUT_SECONDS` | 数据库请求超时，默认 10 秒                 |
| `QDRANT_TEST_URL`        | 测试专用服务；不配置时测试自动启动隔离进程           |

这些变量由 Python 在启动时读取；`.env.example` 不自动加载。
切换地址后需重启应用，不实现热加载。Qdrant Key 不应写入 URL；localhost 开发服务默认无鉴权，
需要远程/生产部署时须单独配置服务端鉴权、TLS 和网络访问控制。

集合采用命名向量 `text-dense`，距离为 Euclid，以保持旧 Chroma L2 的检索方向。
启动时通过本地 Embedding 实际返回长度确定维度；模型或维度变化必须重新建库导入 TXT。
适配层将 Euclid 距离转换为 `1/(1+distance)`，以满足 LlamaIndex RRF 的降序分数约定。
节点 ID 是由原 SHA-256 派生的 UUID v5，Document ID 和按文档删除语义不变。

## 数据保护与恢复

- 不复用旧 Chroma docstore 直接跳过导入；集合缺失、点数或节点 ID 不匹配时应用拒绝启动。
- Qdrant 集合与 JSON docstore 必须成对备份和恢复；二者之间没有跨存储事务。
- 不支持多个应用进程共写同一 JSON docstore；本次替换不扩展分布式一致性能力。
- 已有错误维度/距离的集合不会自动删掉或重建，需选择新集合并配套空 docstore 后重导。
- 回退代码不等于完成数据回滚：新 UUID docstore 不能混用旧 SHA-256 Chroma 库。
  回退必须恢复同一时点的旧代码、依赖、Chroma 与旧 docstore，不删新 Qdrant 数据。

## 官方依据

- [LlamaIndex QdrantVectorStore](https://developers.llamaindex.ai/python/framework-api-reference/storage/vector_store/qdrant/)
- [Qdrant 官方发行包](https://github.com/qdrant/qdrant/releases)
- [Qdrant Point ID 规则](https://qdrant.tech/documentation/concepts/points/)
