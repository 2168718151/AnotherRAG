# 前端接入历史（已停用）

> 当前运行入口见[项目 README](../README.md)与[文档导航](README.md)。以下页面路由与启动步骤仅供历史追溯，不应照作当前操作。

> 2026-10-30：按用户要求，一键配置启动入口和前端连接均已撤下。API 不再托管首页、
> /static/ 静态资源或 /v1/ui/context。CLI、资料管理和多轮会话业务 API 保留；
> 网页设计文件仅作为未托管的参考稿保留。以下内容是历史接入说明，不是当前操作步骤。

## 访问链路

浏览器 → 同源 FastAPI `127.0.0.1:18082` → 当前绑定知识库 / PostgreSQL 会话。
最终版 HTML 和业务脚本由 API 提供，不再使用 `file://` 预览作为运行入口。
静态文件位于 `src/story_rag/static/`；独立 E 盘预览稿保持原样。

| 服务 | 默认端口或位置 | 用途 |
|---|---|---|
| FastAPI + 页面 | `127.0.0.1:18082` | `/` 为工作台，`/v1/` 为业务接口 |
| LM Studio | `127.0.0.1:1234` | Embedding，启动前加载既有模型 |
| Qdrant | `127.0.0.1:6333` | 当前绑定库的向量存储 |
| PostgreSQL | 由 `STORY_RAG_POSTGRES_DSN` 指定 | 长期会话与独立展示记录；schema 必须为版本 5 |
| Reranker | 既有本地模型目录 | 应用进程内加载，无独立 HTTP 端口 |
| DeepSeek | 既有远程配置 | 问答生成；浏览器不接收供应商凭据 |

Host、客户端 IP 必须属于本机回环地址；请求的 Origin 如存在，必须与访问地址完全一致。
不增加通配 CORS，不在页面存放 DSN、API Key 或连接任意数据库地址。

## 启动准备

1. 启动既有 Qdrant、LM Studio Embedding 和 PostgreSQL。
2. `scripts/run_api.ps1` 优先使用已设置的 `STORY_RAG_POSTGRES_DSN`，否则读取
   `data/postgres/session-url.txt`；它只在进程环境中把 `postgresql+asyncpg://`
   转为 psycopg 支持的 `postgresql://`，不改写连接文件，也不会打印 DSN。
   会话 schema 默认是 `story_rag`，可用 `-PostgresSchema` 显式指定其他 schema。
3. 只读核对目标实例、数据库与 schema。首次使用时必须经确认后单独执行迁移；
   API 和工作台都不会自动迁移数据库。迁移前在 PowerShell 中设置与目标一致的 DSN：

```powershell
$env:STORY_RAG_POSTGRES_DSN = (Get-Content -LiteralPath '.\data\postgres\session-url.txt' -Raw -Encoding UTF8).Trim().Replace('postgresql+asyncpg://', 'postgresql://')
$env:STORY_RAG_POSTGRES_SCHEMA = 'story_rag'
uv run python scripts/migrate_conversations.py --apply
```

4. 启动 API，显式绑定一座已登记知识库。例如：

```powershell
Set-Location -LiteralPath 'D:\text_Project\Another RAG'
uv run story-rag db list --json
.\scripts\run_api.ps1 -Database '牛客面试'
```

迁移命令仅供核对目标实例并决定升级后使用：

```powershell
uv run python scripts/migrate_conversations.py --apply
```

启动后访问 `http://127.0.0.1:18082/`。API 启动需要 Embedding 在线，
不把单独打开 HTML 或 `/health` 正常当作完整问答链路已通过。

## 页面能力与现有接口

| 页面操作 | 接口与语义 |
|---|---|
| 显示当前知识库及就绪状态 | `GET /v1/ui/context` |
| 选择 / 拖放 TXT | `POST /v1/qa/ingest`；逐份导入并报告失败 |
| 搜索 / 分页查看资料 | `GET /v1/documents` |
| 查看正文 | `GET /v1/documents/{id}/preview`；默认最多 8,000 字符 |
| 删除资料 | 先取详情，再携带数据库标识和 ETag 精准删除；源文件保留 |
| 新建会话 / 历史会话 | `/v1/conversations` 及 `/{id}/turns` |
| 提交 / 追问 | `request_id` 幂等与 `expected_revision` 冲突检查 |
| 隐藏历史会话 | `/{id}/archive`；页面明确标为归档，账本仍保留 |
| 核对未知结果 | 读取展示记录；显式核对原进程是否退出，再按后端状态处理 |
| 断线重发 | 仅当读回未发现原记录时，允许用同一请求 ID 重发 |
| 查看回答依据 | 读取该轮题目版本的资料引用；已删除引用显示不可读取 |

会话展示正文与业务账本分开保存；旧轮次缺失的正文不能补回。页面用
`sessionStorage` 临时保留未核对请求，核对后清除；这不替代 PostgreSQL 持久化。

当前 HTTP 服务只绑定一座库。广场显示该库，选择器不可切换；建库、整库删除和
切换仍走既有 CLI。后续要把这些操作开放到页面，需另行实现数据库生命周期 HTTP
路由及进程绑定方案。当前页面不提供模拟成功的建库 / 删库交互。

## 验收边界

API 契约使用隔离 Qdrant / PostgreSQL 测试，页面可使用模拟响应验证交互与布局；
这两者都不能替代正式依赖在线时的真实模型问答验收。实际启动和完整问答验收
还要求依赖服务、目标库与 schema 都就绪。不得自动升级正式库或调用付费模型测试。
