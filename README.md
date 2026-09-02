# TraceGraph Coder

TraceGraph Coder 是一个面向编程任务的轻量级 Coding Agent。它不依赖 LangChain、LlamaIndex、AutoGen、CrewAI、OpenAI Agents SDK 等现成 Agent 框架，只使用 OpenAI 兼容的 Chat Completions / Tool Calling 接口让大模型做规划与决策；文件读写、命令执行、仓库索引、上下文压缩、会话恢复、证据记录、验证与终止条件都由本地代码实现。

项目目标是做一个简化但机制完整的 Claude Code / Codex 风格编程智能体：用户给出自然语言任务后，Agent 会自主选择工具，读取相关代码，生成补丁，运行验证，并给出可追溯的最终结果。

## 总体架构

```text
Web 工作台 / CLI
  -> AgentController
  -> OpenAICompatibleLLM
  -> ToolRegistry + 控制工具
  -> ToolEnvironment
  -> RepoGraph / EvidenceLog / WorkingMemory / CodingContext / ContextWindow / Verifier
  -> SessionStore / ExperienceStore
```

核心执行循环：

```text
构建仓库图
 -> 读取项目记忆与历史经验
 -> 进入 PLAN / LOCATE / READ / PATCH / VERIFY / REPORT 阶段
 -> 编译仓库图、证据链、工作记忆和经验提示组成的 Coding Context
 -> 在上下文压力较高时压缩旧对话和旧工具结果
 -> 调用大模型，让模型自行决定下一步工具
 -> 本地执行工具并记录证据
 -> 基于证据生成最小补丁
 -> 修改后自动验证并检查文件指纹
 -> 验证通过后才允许完成任务
 -> 保存报告、会话与可复用经验
```

## 核心能力

- 本地文件工具：支持列文件、读文件、批量读文件、新建文件、补丁修改。
- 本地命令工具：支持安全执行单条命令，禁止 shell 串联和危险命令。
- 仓库图工具：支持查询相关文件、依赖邻域、反向依赖、相关测试。
- 会话记忆工具：支持读取当前工作区的历史对话，继续未完成任务。
- 验证工具：修改后可自动运行测试或显式调用验证工具。
- Web 工作台：提供类似 Codex 的对话区、过程列表、最终结果、证据链、工作记忆、会话树和仓库图视图。
- CLI 模式：保留命令行入口，便于调试和脚本化运行。

## 创新点

### 1. 仓库证据图

Agent 启动时会先构建 RepoGraph，索引文件路径、语言类型、模块角色、符号、导入关系、反向导入关系、函数调用和测试关联。大模型不需要一次性看到所有代码，而是通过 `repo_graph_query` 和 `repo_graph_neighborhood` 按需检索相关文件。

这相当于给 Agent 提供一个轻量级的代码地图：先定位，再精读，再修改，减少无效上下文消耗。

### 2. 可解释检索

仓库图查询不会只返回文件名，还会返回匹配分数、命中的关键词和推荐原因。模型可以知道“为什么这个文件可能相关”，而不是面对一个黑盒检索结果。

### 3. 证据驱动执行

每一次工具调用都会写入 `.tracegraph/evidence.jsonl`。Agent 修改代码前必须已有仓库证据，修改代码后必须验证。最终完成不是靠模型自己说“完成了”，而是由本地 evidence gate 检查：

- 是否发生过修改；
- 修改之后是否运行过验证；
- 验证时的文件指纹是否仍然匹配当前文件。

### 4. 结构化工作记忆

`WorkingMemory` 会记录当前任务阶段、候选文件、目标路径、已读文件、已修改文件、验证结果、历史经验提示和失败信号。模型可以通过 `record_progress` 更新假设和下一步计划，但文件证据、修改记录和验证状态只能由本地工具结果确认。

### 5. 失败经验感知上下文压缩

这是项目最重要的创新点之一。传统上下文压缩容易按时间裁剪，导致失败原因、错误输出、关键文件线索被挤出窗口。TraceGraph Coder 会把重复搜索、验证失败、harness 拦截、完成门控失败等事件结构化为 `FailureEvent`，再编译成高优先级的 `Failure Control Packet`。

下一轮调用模型时，这些失败事件会被显式保留，让模型知道：

- 哪种搜索策略刚刚失败；
- 哪些文件已经足够可疑；
- 哪些错误关键词必须继续保留；
- 下一步应该改为精读、补丁、验证或收敛回答。

这把“失败日志”转化成了“上下文调度信号”，让 Agent 更少重复犯错。

### 6. 自适应执行 Harness

Harness 不写死固定流程，而是观察模型选择的工具是否低收益：

- 重复调用相同工具会被拦截；
- 已经有候选文件后继续泛搜会被拦截；
- 修改代码后又回到大范围搜索会被拦截；
- 连续探索没有新证据时会提示模型收敛。

它的作用不是代替模型规划，而是在模型跑偏时提供轻量级控制。

### 7. 分层上下文管理

系统保留完整会话记录，但每次发给模型的是压缩后的工作视图。上下文由几层组成：

- 系统规则；
- 当前任务；
- 工作记忆；
- 仓库图候选节点；
- 最近证据；
- 历史经验；
- 压缩后的旧工具结果。

当上下文压力变大时，系统会优先保留目标文件、失败关键词、验证输出和最近用户意图，旧的大段工具结果会压缩成摘要。

### 8. 可继续会话

每个工作区拥有独立的会话记录。用户可以像使用 Codex 一样在左侧选择历史对话继续，也可以点击“新对话”开启一个全新的上下文。大工具输出会被外置为 blob 文件，主会话索引保持轻量。

### 9. 本地安全边界

所有路径都会限制在当前工作区内。命令执行拒绝危险命令和 shell 串联，工具参数本地校验，输出会脱敏，API Key 不写入仓库，也不会通过默认接口回传给前端。

## 运行方式

### Web 工作台

Windows 下可直接双击：

```text
TraceGraph Coder Web.pyw
```

它会启动本地服务并打开浏览器工作台。左侧可以选择工作区、填写模型配置、查看历史会话；主区域可以输入任务、继续会话、查看模型执行过程、最终结果、验证结果、证据链、工作记忆、会话树和仓库图。

API Key 可以在界面中临时输入，也可以通过环境变量提供：

```powershell
$env:TRACEGRAPH_API_KEY="..."
$env:TRACEGRAPH_MODEL="your-model"
```

如果使用 DeepSeek 兼容接口：

```powershell
$env:DEEPSEEK_API_KEY="..."
$env:TRACEGRAPH_MODEL="deepseek-chat"
```

### 命令行模式

```powershell
python -m tracegraph_coder --workspace path\to\repo "修复失败的登录测试"
```

只构建仓库图：

```powershell
python -m tracegraph_coder --workspace path\to\repo --graph-only
```

查看最近保存的会话：

```powershell
python -m tracegraph_coder --workspace path\to\repo --show-session
```

继续最近会话：

```powershell
python -m tracegraph_coder --workspace path\to\repo --resume-session "继续修复刚才的验证失败"
```

查看历史经验卡：

```powershell
python -m tracegraph_coder --workspace path\to\repo --show-experiences
```

## 测试

```powershell
python -m unittest discover -s tests
```

当前测试覆盖 controller、repo graph、tools、working memory、context window、session 和 Web app 等核心模块。

## 安全与提交说明

- 项目不包含任何真实 API Key。
- API Key 请通过环境变量或运行时输入提供。
- `.tracegraph/`、`__pycache__/`、本地报告和临时任务文件不会提交到仓库。
- 本项目的 Agent 核心逻辑由本地 Python 实现，没有封装任何现成 Agent 产品或 Agent 框架。
