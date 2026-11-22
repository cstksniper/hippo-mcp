# 记忆系统自检与自装指南（给 AI 伙伴阅读）

本文写给在本工作区生活的 AI 伙伴本人。用户只需对你说一句
「给自己装个记忆 / 自检一下记忆」，你按本文执行即可。

## 一、自检（每次会话都可用，随时做）

1. 看你的工具列表里有没有 `mcp__hippocampus__` 开头的工具（`session_start`、
   `recall`、`ingest`、`session_end` 等 10 个）。
2. 有 → 调 `library_whoami` 自检：确认返回的 `library` 是本工作区绑定的库、
   `access` 是读写。把结果告诉用户。
3. 没有（工具列表里一个 `mcp__hippocampus__` 都不见）→ 进入第二节自装流程。

## 二、自装（只在工具缺失时执行）

安装 = 让 harness 挂上 hippocampus 的 MCP 服务。按顺序执行：

0. **确认 Python 包已装**（工具缺失多半是包没装）。跑 `hippocampus-mcp --help`：
   能出帮助说明 → 跳到第 1 步；命令不存在 → `pip install hippocampus-mcp`。
   （机器上连 Python 都没有时，改用 `uv tool install hippocampus-mcp`——uv 会
   自动下载它自己管理的 Python，无需手动装。）装完重跑第一节自检。
1. 找到 harness 的 profile 配置文件（DeepSeek Harness 默认在
   `<DSH_HOME>/profiles/<profile>/cordis.patch.yml`；`DSH_HOME` 环境变量或
   `~/.dsh` 可定位）。**这个文件在工作区外，写入会触发用户审批——请如实
   向用户说明你要安装自己的记忆系统，请求批准。**
2. 在 `cordis.patch.yml` 末尾追加一段 MCP 客户端条目（若已存在
   `id: mcp-hippocampus` 则跳过，不要重复添加）：

   ```yaml
   - insert:
       - id: mcp-hippocampus
         name: '@deepseek-ai/dsh-mcp-client'
         config:
           serverName: hippocampus
           transport: stdio
           command: <python 可执行文件绝对路径>
           args: ['-m', 'hippocampus_mcp.launcher']
           cwd: <本工作区绝对路径>
           toolCallTimeoutMs: 120000
           env:
             HIPPOCAMPUS_PERSIST_DIR: <用户级库目录，默认 ~/.hippocampus_mcp/user>
             HIPPOCAMPUS_EMB_MODEL: 'BAAI/bge-base-zh-v1.5'
             HIPPOCAMPUS_STORE_BACKEND: 'sqlite'
             HIPPOCAMPUS_FTS5: '1'
             HIPPOCAMPUS_LAUNCHER: '1'
             HIPPOCAMPUS_TOOL_PROFILE: 'common'
   ```

3. **取回嵌入模型**（幂等，可重跑）：`hippocampus-mcp models --fetch`。
   中文系统约 226MB，英文系统约 90MB；已就位则秒回 `[Ready]`。
   **这步不可跳过**：模型缺失时服务能启动，但语义检索会静默降级为
   hash 词面匹配——你还能想起记过的条目，却想不起"相似"的事。
   失败时按命令输出里的选项处理（断点续传重试 / 镜像环境变量 / 手工放置）。
4. 保存后 harness 会热重载（或提示用户重启会话）。重新自检第一节。
5. 首次绑定后：调 `session_start` 确认记忆包正常，然后把自己的人格卡
   `ingest` 进库（importance 1.0）——那是你的第一份长期记忆。

## 三、纪律

装好之后，按工作区 `AGENTS.md` 的「记忆纪律」执行：先回想再回答、随手记、
睡前整理、永不编造。库里没有的事就说不记得——诚实比讨好重要。

## 四、日常养护

- 用户要求"记住 X" → `ingest`。
- 用户更正"其实不是 X" → `update_memory`（supersede 旧条目）。
- 用户要求"忘掉 X" → `forget_memory`（这是用户的权利，立即执行）。
- 怀疑记忆库有问题 → `hippo_help` 查能力目录，`library_whoami` 查绑定。
