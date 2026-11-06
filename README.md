# AnotherRAG

**面向中英双语 Q&A 文本的本地知识库与多轮问答项目。**

AnotherRAG 基于 LlamaIndex 构建，将 TXT 文档转为可检索的知识库，通过向量检索、BM25、RRF 融合与本地模型重排寻找依据，再连接 DeepSeek 生成回答。项目提供命令行和 FastAPI 接口，并使用 PostgreSQL 保存多轮会话、题目版本与对话展示记录。

适合用于整理技术问答、面试知识、个人学习材料，以及研究 RAG 的检索、上下文组装和多轮追问流程。当前以 Windows 本机运行环境为主，前端网页托管已停用。

## 核心能力

| 能力     | 说明                                                     |
| ------ | ------------------------------------------------------ |
| TXT 导入 | 支持 UTF-8、UTF-8 BOM、GB18030；识别 `问题：/答案：` 结构，普通文本按分块规则处理 |
| 混合检索   | 向量召回与 BM25 关键词检索结合，通过 RRF 融合，再由 Qwen3 Reranker 重排      |
| 命名知识库  | 创建、列出、重命名知识库，并在 CLI 会话中显式选库、切库                         |
| 文档管理   | 列表、搜索、正文预览、改名、内容替换和按身份/版本确认的精准删除                       |
| 检索预览   | `query` 展示检索结果与最终 LLM 输入，不调用生成或相关性门控模型                 |
| 多轮问答   | `ask` 支持任务规划、追问关联、条件修订、逐题检索、生成和门控                      |
| 历史读取   | 在限定会话范围内回读用户原话或已保存回答，支持复述、格式转换与知识追问                    |
| 请求恢复   | 使用 request ID、revision 和请求状态区分重放、冲突及结果未知，不自动重复未知请求     |
| 诊断与评估  | 提供阶段追踪、输入预览、隔离测试与基准脚本，便于核查检索和生成链路                      |

## 工作流程

### 知识入库与检索

```text
TXT 文档
  → 编码识别与问答解析
  → 节点切分与 Embedding
  → Qdrant 向量存储 + JSON docstore 原文/节点持久化

用户问题
  → 向量检索 + BM25
  → RRF 融合
  → 本地 Qwen3 Reranker 重排
  → 上下文组装
  → 检索预览，或生成回答并执行相关性门控
```

### 多轮会话

```text
当前消息 + 受范围约束的历史信息
  → 任务规划与历史定位
  → 条件修订、逐题证据检索或历史正文回读
  → 按预算生成、校验并合并结果
  → PostgreSQL 提交请求状态、版本与可见对话记录
```

历史助手回答与知识库证据分开处理。知识问答缺少依据时按链路返回“无”或相应状态；历史正文缺失、歧义和读取故障会单独区分。

## 技术栈

| 部分        | 使用技术                                                             |
| --------- | ---------------------------------------------------------------- |
| 运行环境      | Python 3.12 / 3.13、uv                                            |
| 应用与编排     | LlamaIndex、FastAPI、Uvicorn                                       |
| 向量存储      | Qdrant                                                           |
| 原文与检索节点   | 持久化 JSON docstore                                                |
| 关键词召回与融合  | BM25、Reciprocal Rank Fusion                                      |
| Embedding | LM Studio 的 OpenAI 兼容接口，默认 `text-embedding-qwen3-embedding-0.6b` |
| Reranker  | 本地 `Qwen3-Reranker-0.6B`                                         |
| 生成模型      | DeepSeek，模型名与 API Key 从本机配置文件读取                                  |
| 会话存储      | PostgreSQL、psycopg，显式 SQL 迁移                                     |

## 环境要求

- **Windows x64**：仓库附带的数据库程序和 PowerShell 脚本面向该环境。
- **Python 3.12 或 3.13**，并安装 [uv](https://docs.astral.sh/uv/)。
- **LM Studio**：加载指定 Embedding 模型，并启用本地 HTTP 服务。
- **本地 Reranker 模型目录**：模型权重需要自行准备。
- **Qdrant**：副本保留了 Windows 程序，也可连接自行部署的服务。
- **PostgreSQL**：仅多轮会话需要；副本保留必要 Windows 运行组件。
- **DeepSeek API Key**：调用生成、规划等远端模型能力时需要，检索预览不需要。

Windows 依赖当前锁定 CUDA 13.0 版 PyTorch，安装后的 Python 环境会比仓库本身大得多。硬件与驱动需满足所选运行方式；本项目没有提供经过本轮验证的 CPU 专用锁文件或跨平台部署流程。

## 快速开始

以下命令均在项目根目录的 PowerShell 中执行。首次建议将仓库放在较短路径，避免 Windows 下数据库深层文件路径过长。

### 1. 安装 Python 依赖

```powershell
uv sync --frozen --group dev
```

项目的依赖文件分工：

- `pyproject.toml`：项目元数据、直接依赖和 CLI 入口。
- `uv.lock`：uv 使用的锁文件，推荐按它重建环境。
- `requirements.txt`：由 uv 导出的 pip 兼容依赖清单，不含开发依赖。

使用 pip 时，可在支持的 Python 虚拟环境中执行 `python -m pip install -r requirements.txt`。日常开发优先使用上面的 uv 命令。

### 2. 准备本地模型

在 LM Studio 中加载 `text-embedding-qwen3-embedding-0.6b`，启用默认地址为 `http://127.0.0.1:1234/v1` 的服务。

在当前 PowerShell 中设置自己的 Reranker 路径：

```powershell
$env:RERANKER_MODEL_PATH = 'D:\models\Qwen3-Reranker-0.6B'
$env:LMSTUDIO_BASE_URL = 'http://127.0.0.1:1234/v1'
$env:LMSTUDIO_EMBEDDING_MODEL = 'text-embedding-qwen3-embedding-0.6b'
```

将示例路径改为模型实际位置。源码和 `.env.example` 中保留了原开发机的默认模型路径，新机器应通过环境变量覆盖。

### 3. 启动 Qdrant

```powershell
.\scripts\run_qdrant.ps1
```

默认 HTTP 地址为 `http://127.0.0.1:6333`，数据写入本项目的 `data/qdrant/`。如果端口已有服务占用，先核实其用途，不要重复启动。如果缺少程序，可通过 `.\scripts\install_qdrant.ps1` 下载并校验固定版本。

### 4. 创建知识库并导入 TXT

```powershell
uv run story-rag db create --name "学习资料"
uv run story-rag db list --json
.\scripts\ingest.ps1 -Database "学习资料" -Path 'D:\qa\interview.txt'
uv run story-rag documents list --db "学习资料"
```

导入路径支持单个 TXT 或目录，需替换为自己的文件位置。创建知识库和导入需要 Embedding 服务就绪。

推荐的问答文本格式：

```text
问题：什么是 RAG？
答案：RAG 是检索增强生成，通过检索外部知识为生成模型提供上下文。

问题：What is hybrid retrieval?
答案：Hybrid retrieval combines semantic vector search with keyword-based retrieval.
```

### 5. 预览检索结果

```powershell
uv run story-rag shell
```

进入交互会话后输入：

```text
db use "学习资料"
db current
query "什么是 RAG？" --trace
db release
exit
```

`query` 会执行本地检索与重排，并返回组装后的模型输入；它不会调用远端生成模型。选库仅在当前会话生效。

## 多轮问答

多轮问答需要在上述环境外，另行准备 PostgreSQL 会话库和模型凭据。

### 配置模型凭据

复制配置模板，在本地填入自己的 Key 和模型名：

```powershell
Copy-Item .\api.txt.example .\api.txt
```

`api.txt` 格式如下，占位文字须替换：

```text
deepseek: YOUR_DEEPSEEK_API_KEY
模型: deepseek-flash
```

模型名以账号实际可用模型为准。`api.txt` 已加入 Git 忽略，不要提交真实凭据。已有文件时直接编辑，不重复复制覆盖。

### 配置 PostgreSQL 并显式迁移

先准备已经启动的 PostgreSQL 服务及专用空数据库，再设置连接信息。仓库保留数据库运行程序，但不包含已初始化的集群或原机连接文件。

```powershell
$env:STORY_RAG_POSTGRES_DSN = 'postgresql://USER:URL_ENCODED_PASSWORD@127.0.0.1:55432/story_sessions'
$env:STORY_RAG_POSTGRES_SCHEMA = 'story_rag'
```

将用户名、密码、端口和数据库名换为实际值，密码中的特殊字符需 URL 编码。确认连接目标后执行迁移：

```powershell
uv run python scripts\migrate_conversations.py --apply
```

当前会话 schema 要求版本 5。应用启动只检查版本，不自动迁移；有关会话存储与迁移的说明见 [conversation_ledger.md](docs/conversation_ledger.md)。

### 创建会话并追问

在设置了上述环境变量的 PowerShell 中运行 `uv run story-rag shell`，随后输入：

```text
db use "学习资料"
conversation create "RAG 学习"
ask "根据资料介绍一下 RAG。"
ask "继续解释上一个问题中的检索步骤。"
ask "把第一轮的原回答原样发我。"
conversation list
question list
exit
```

`ask` 可能调用远端模型进行规划、生成或门控，产生 API 费用。即使请求读取历史原文，也可能先调用规划模型；只有 `query` 明确保持零生成预览边界。

## HTTP API

先退出占用同一知识库的 CLI 会话，再启动绑定该库的 API：

```powershell
.\scripts\run_api.ps1 -Database "学习资料"
```

默认监听 `http://127.0.0.1:18082`。主要接口：

| 方法         | 路径                                     | 用途            |
| ---------- | -------------------------------------- | ------------- |
| GET        | `/health`                              | 进程健康检查        |
| GET        | `/ready`                               | 存储就绪检查        |
| POST       | `/v1/qa/ingest`                        | 导入问答 TXT      |
| GET        | `/v1/documents`                        | 搜索和分页列出文档     |
| GET        | `/v1/documents/{document_id}/preview`  | 预览文档正文        |
| DELETE     | `/v1/documents/{document_id}`          | 经身份与版本校验删除文档  |
| POST       | `/v1/chat/query`                       | 单轮问答，会调用生成链路  |
| POST / GET | `/v1/conversations`                    | 创建或列出会话       |
| POST / GET | `/v1/conversations/{session_id}/turns` | 提交一轮或分页读取展示记录 |
| POST       | `/debug/v1/generation/input`           | 预览模型输入，不调用生成  |

完整参数及错误说明见 [interfaces.md](docs/interfaces.md)。API 进程启动后固定绑定一个知识库，不支持逐请求切库；知识库注册管理仍通过 CLI 操作。`/ready` 的存储就绪不代表模型、凭据或 PostgreSQL 会话能力全部可用。

## 常用配置

`.env.example` 仅供参考，应用不会自动加载它。请使用 PowerShell 的 `$env:变量名` 设置配置，并在同一终端启动程序。

| 环境变量                        | 用途                                   |
| --------------------------- | ------------------------------------ |
| `API_FILE`                  | 模型配置文件，默认项目根目录 `api.txt`             |
| `DEEPSEEK_BASE_URL`         | 模型接口地址，默认 `https://api.deepseek.com` |
| `LMSTUDIO_BASE_URL`         | 本地 Embedding 服务地址                    |
| `LMSTUDIO_EMBEDDING_MODEL`  | Embedding 模型标识                       |
| `RERANKER_MODEL_PATH`       | 本地 Reranker 模型目录                     |
| `QDRANT_URL`                | Qdrant HTTP 地址                       |
| `QDRANT_API_KEY`            | 可选的 Qdrant 服务凭据                      |
| `STORY_RAG_POSTGRES_DSN`    | 多轮会话数据库连接串                           |
| `STORY_RAG_POSTGRES_SCHEMA` | 会话 schema，默认 `story_rag`             |
| `APP_HOST` / `APP_PORT`     | API 监听地址与端口                          |
| `DEBUG_ENABLED`             | 是否开放诊断接口                             |

## 项目结构

```text
.
├── src/story_rag/       # CLI、API、工作流与配置
│   ├── adapters/       # 模型、存储与注册表适配
│   ├── pipelines/      # 解析、检索、生成与历史处理
│   ├── migrations/     # PostgreSQL 显式迁移
│   └── static/         # 未托管的前端参考文件
├── scripts/            # 启动、导入、迁移与评估脚本
├── tests/              # 单元、契约与集成测试
├── docs/               # 架构、接口、评估和阶段记录
├── tools/              # Windows 数据库运行组件
├── figma/              # 开发期设计接入资料
├── .env.example        # 环境变量示例
├── api.txt.example     # 模型凭据格式模板
├── pyproject.toml      # 项目配置与直接依赖
├── uv.lock             # 依赖锁文件
└── requirements.txt    # pip 兼容依赖清单
```

运行后生成的 `data/`、`logs/`、`reports/` 与 Python 虚拟环境不随该发布副本提供。模型权重、真实密钥、知识库内容和原机历史会话也不包含在内，需要自行配置或从一致性备份恢复。

## 测试

不依赖真实模型服务的配置、文本解码和问答解析测试：

```powershell
uv run pytest tests/test_config.py tests/test_text_loader.py tests/test_qa_parser.py -q -p no:cacheprovider
```

检查代码风格：

```powershell
uv run ruff check src scripts tests
```

运行较完整的自动化回归：

```powershell
uv run pytest -q -m "not integration" -p no:cacheprovider --basetemp reports\pytest_local_run_01
```

完整回归中的存储用例会启动隔离数据库实例，需要保留相应程序；部分真实会话测试另有显式开关。`--basetemp` 必须选尚不存在的新目录，pytest 会清理该目标目录，不能指向已有业务数据。真实模型集成用例需另外准备模型服务，Mock 测试通过不代表真实回答质量。

## 使用边界

- 当前按本机访问设计，未提供完整的公网多用户鉴权方案。
- 同一物理知识库由受支持的 CLI/API 入口互斥占用，不能同时打开多个写入者；跨存储没有分布式事务。
- Qdrant 与 JSON docstore 必须配套备份和恢复，更换 Embedding 模型后需要重新建库导入。
- 删除和内容替换需核对 UUID、Document ID 与 ETag；恢复命令用于继续未完成操作，不等于撤销删除。
- 前端页面没有挂载到 API，保留的静态文件和 Figma 资料属于参考资源。
- 部分文档记录历史阶段、旧分支和旧实验路径，具体可用接口以当前源码为准；部分基准依赖未随仓库提供的外部语料。
- 仓库体积不代表部署后的磁盘占用。安装依赖、下载模型和积累测试数据库都会增加本地体积。

## 进一步阅读

- [文档导航与当前状态](docs/README.md)
- [架构说明](docs/architecture.md)
- [HTTP 与 Python 接口](docs/interfaces.md)
- [Qdrant 配置与存储边界](docs/qdrant.md)
- [文档管理](docs/file_management.md)
- [会话账本](docs/conversation_ledger.md)
- [逐题证据映射](docs/evidence_mapping.md)
- [历史原文读取与追问](docs/历史原文读取补丁与验收_20261031.md)
- [评估方法](docs/evaluation.md)
