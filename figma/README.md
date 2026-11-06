# Figma 接入准备

本目录用于 Another RAG 的开发期界面设计，不是 RAG 的生产接口，也不恢复暂缓的业务 MCP 接口。
2026-11-06 文档核对：本目录保留开发期连接配置、脚本和设计参考。旧记录中的 OAuth、云端文件或套餐状态只适用于对应历史阶段；本次没有检查当前账号授权或连接状态。

## 可以做什么

Figma 官方远程 MCP 支持读取设计上下文、截图、组件与变量，也提供创建文件、
生成 FigJam 图表、捕获网页界面和编辑原生画布的工具。
实际可用范围以账号权限、套餐、客户端及连接后暴露的工具为准，不能仅凭配置文件认定全部可用。
当前运行入口为 CLI/业务 API，不托管网页。`src/story_rag/static/` 保留前端参考文件，不代表浏览器工作台已部署；本次文档更新不生成新界面或修改云端设计。

## 文件入口

- `../.codex/config.toml`：Codex 项目级远程 MCP 配置，无凭据。
- `connect.ps1`：默认只查看配置及授权状态，`-Login` 显式启动 OAuth。
- `design-brief.md`：后续 Q&A 管理和检索界面的设计范围与验收目标。
- `../docs/releases/V1.1.md`：Git 版本范围和本地查看方式。

## 首选方式：Codex 的 Figma 插件

按 Figma 官方安装说明，在 Codex 中进入 Plugins，找到 Figma，安装并完成浏览器 OAuth 授权。
账号登录和 Allow access 由用户本人操作，不在聊天中发送密码、Token 或 API Key。
插件可能附带针对设计工作流的技能；需要直接写画布时，优先使用官方插件及其说明。

安装插件与手动配置是两种接入方式，通常选择一种即可。若插件已经提供 Figma 工具，
可将项目配置中的 `enabled` 改为 `false`，避免重复连接；不要擅自删除其他全局 MCP 配置。

## 备选方式：项目级 MCP 配置

以当前克隆/复制后的项目根目录在 Codex 中打开，并确认项目可信。
项目级 `.codex/config.toml` 只有在受信任项目中才会加载；
若仍在其他父目录会话中，子项目配置不会因此自动成为当前会话工具。
配置被加载后，按客户端提示刷新连接或开始项目会话，再确认工具实际出现。

先检查，不发起登录、不修改全局 MCP 配置、不创建 Figma 文件：

```powershell
Set-Location "<你的 AnotherRAG 项目根目录>"
.\figma\connect.ps1
```

`AuthStatus=not_logged_in` 表示还没有登录。仅在你准备授权时运行：

```powershell
.\figma\connect.ps1 -Login
```

脚本使用一次性的 CLI 配置覆盖，便于尚未设置信任时检查和登录；
这不代表项目配置已被当前聊天加载，也不把服务器注册到全局配置。
OAuth 凭据由 Codex 的凭据存储管理，不写入项目或 Git。
即使 CLI 显示已有凭据，仍需在实际会话中用 Figma 工具验证连接。

## 连接验收

1. 当前项目会话能看到 Figma 的实际工具，而不仅是 CLI 配置条目。
2. 调用 `whoami` 确认身份及权限；不要将完整账号信息写入日志或提交。
3. 提供一个你有权访问的 Figma frame/layer 链接，读取设计上下文并查看截图。
4. 如要制作设计，先确认目标文件或所属团队及写入范围，再使用实际提供的创建、画布工具。
5. 云端创建成功后记录返回的真实文件 URL；没有成功结果时不编造链接。

连接诊断暂不请求 Figma 写操作，不上传语料、模型、日志、Prompt 或凭据，
也不调用可能消耗付费额度的生成工具。
项目配置使用 `default_tools_approval_mode="writes"` 请求对非只读工具确认，
但实际仍以客户端权限策略为准，不能把此设置当作硬性安全隔离。

## 在哪里看

本地准备文件位于当前项目的 `figma/`。当前后端能力见[项目 README](../README.md)和[接口文档](../docs/interfaces.md)；设计资料不替代运行契约。
云端设计：授权并成功创建后，在 Figma 的 Recents / Drafts 或目标团队项目中查看；
本次尚无云端设计链接。
Git 记录在本地 Another RAG 仓库，查看命令见 `docs/releases/V1.1.md`；没有配置远程仓库，不会自动上传 GitHub。

## 官方资料

- [Figma 远程 MCP 与 Codex 安装说明](https://developers.figma.com/docs/figma-mcp-server/remote-server-installation/)
- [Figma 工具及能力说明](https://developers.figma.com/docs/figma-mcp-server/tools-and-prompts/)
- [OpenAI 官方 MCP 配置文档](https://developers.openai.com/codex/mcp)
