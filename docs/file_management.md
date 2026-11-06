# TXT 文件管理与验证

> 当前实现核对：2026-11-06。当前通过 CLI/业务 API 管理文档，不提供已托管的文件管理网页。历史验证数据不等于本次文档更新重新执行的结果。

## 范围

- API 提供绑定库的 TXT 文件名、来源、节点数和映射状态；新增 CLI 可显式选择命名库管理文件。
- 列表支持搜索和分页；删除必须携带数据库标识和文件版本，允许先调用只读预览核对范围。
- 2026-10-13 按用户要求移除自定义文件管理页面、静态资源及专用浏览器测试，后续优先命令行验证。
- API 不提供逐请求切库；CLI 的建库、选库、改名和替换见[命令行接口](interfaces.md#命令行接口)。不提供批量删除、回收站或任意磁盘文件删除。
- 详细请求/响应和失败恢复约定见 [interfaces.md](interfaces.md)。

## 实现位置

工作流：文件目录读取 → 数据库/版本确认 → 删除恢复记录 → 原生文档删除 → 残留校验 →
原子持久化 → revision 更新 → 下一次检索重建。

| 部位 | 文件 |
|---|---|
| 目录、详情、版本校验、只读预览 | `src/story_rag/pipelines/documents.py` |
| Qdrant/docstore 清理、原子写入、重启恢复、共享锁 | `src/story_rag/adapters/storage.py` |
| 入库/检索协调 | `src/story_rag/pipelines/ingestion.py`、`retrieval.py` |
| HTTP 与本机访问保护 | `src/story_rag/api.py` |
| 生产/诊断端口 | `src/story_rag/ports.py` |

优先核对当前安装的官方源码：`SimpleDocumentStore.get_all_ref_doc_info()`、
`VectorStoreIndex.delete_ref_doc()`、`QdrantVectorStore.delete()`。
Qdrant 原生客户端默认等待删除完成，之后再独立验证没有残留。
重启后从 docstore 恢复内存 IndexStruct 映射，不重新计算 Embedding。

## 隔离验证

Python 测试使用夹具启动的独立 Qdrant 进程、临时 JSON docstore 和随机集合。
不配置 `QDRANT_TEST_URL` 时不会访问主服务；如需指定该变量，必须指向专用测试实例。
测试运行目录使用短路径，避免 Windows Qdrant payload 索引路径长度问题。

```powershell
uv run pytest tests\test_documents.py -q -p no:cacheprovider --basetemp reports\files_unit_new
uv run pytest -q -m "not integration" -p no:cacheprovider --basetemp reports\files_full_new
uv run ruff check src scripts tests
```

运行前重建开发依赖，并为每次 `--basetemp` 选择尚不存在的新目录；pytest 会清理目标目录，不得指向已有数据。

文件管理测试覆盖：一份 TXT 多 Q&A、双语/自定义标题、同名不同来源、重复/替换入库、
搜索排序分页、精准清理及源文件保留、重启后二次删除、预热 BM25 后重建、最后文档删除、
未知/重复删除、库标识不匹配、过期 ETag、管理不调用模型、Qdrant/持久化/恢复记录清理故障、
错误恢复记录拒绝应用、节点范围不匹配、读取与写入协调、API 前置条件和本机同源保护。
故障注入后的日志包含预期错误，不应把这些隔离测试日志当成主服务故障。
测试还检查已移除的页面/静态资源路径返回 404，不再需要 Node、Playwright 或 Edge。

页面撤下前的历史验证结果：新增文件管理自动化用例 37 个；全项目回归 128 passed、
0 failed、0 skipped，耗时 171.95 秒，包含 3 个真实本地模型集成测试。
最终残留检查的小范围优化后，文件管理与 Qdrant 定向回归为 47 passed，耗时 54.00 秒。
Ruff 与 `git diff --check` 通过。全量的 5 个警告均为现有 TestClient、DatasetGenerator
及其数据集类型的弃用提示。本轮没有重新运行历史 128/256 题长基准，也不改变其命中率或性能数据。
报告分别为 `reports/file_management_tests.xml` 和 `reports/file_management_focused.xml`。

页面撤下后的命令行复验：文件管理、HTTP API 与 Qdrant 三个测试模块共 62 passed、
0 failed、0 skipped，耗时 74.75 秒；包含页面/静态资源返回 404 的检查。
使用独立 Qdrant 与替身模型，不访问主库、不调用付费 LLM；未重新跑完整模型或长基准。
报告为 `reports/file_management_no_page_20261013.xml`，Ruff 与 `git diff --check` 通过。
本次有 TestClient 弃用和一次 Qdrant 服务版本探测警告，均未导致测试失败。

## 命令行操作

```powershell
$base = 'http://127.0.0.1:18082'
$page = Invoke-RestMethod "$base/v1/documents?offset=0&limit=50"
$page | Select-Object collection_name, total, offset, limit | Format-List
$page.items | Select-Object filename, source, node_count, document_id | Format-Table -AutoSize -Wrap

# 如需搜索，将搜索文本编码为 URL 参数；这里只读，不执行删除。
$q = [Uri]::EscapeDataString('面试')
Invoke-RestMethod "$base/v1/documents?q=$q&offset=0&limit=50" | ConvertTo-Json -Depth 6
```

详情、只读删除预览及携带 ETag 的单文件删除命令见 [interfaces.md](interfaces.md)。
这些调用通过已运行 API 完成，不会另起一个共写 docstore 的进程。
不应以主库做破坏性测试；自动化删除/失败恢复测试应使用上面的 pytest 隔离夹具。

原 `reports/fui_D4Xwbd/` 等浏览器截图和结果仅保留为历史记录，不再作为当前测试入口；
本次页面撤下不删除数据库、源 TXT、日志或历史报告，也不重启已有主服务。

## 运行边界

未绑定命名库时，配置默认是 `story_chunks` 和项目 `data/docstore/docstore.json`；绑定命名库时使用注册表中的配套集合与 docstore。
发布副本不包含原机知识库或历史 174 TXT 牛客基准数据，不会自动导入或迁移。
文件列表 API 的空结果只能说明当前配置库为空，不能据此认为隔离基准已经丢失。

本机限制不等于多用户权限；不应把无鉴权 API 转发到公网。
进程内共享锁协调单次操作；新增支持的 CLI/API 入口另持有跨进程物理库锁。
每个库仍采用单进程持有，不支持多 worker，也不能保护绕开应用直接操作 Qdrant 的外部工具。
Qdrant 与 JSON docstore 没有 ACID 跨存储事务，恢复记录用于继续已确认的删除，不能撤销删除。
