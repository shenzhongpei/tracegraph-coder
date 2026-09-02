Git 仓库地址：待创建远程仓库后填写

项目名称：TraceGraph Coder

TraceGraph Coder 是一个不依赖 LangChain、LlamaIndex、AutoGen、CrewAI 等 Agent 框架的轻量级编程智能体。模型只负责规划和选择工具，文件读取、补丁写入、命令执行、仓库索引、上下文管理、会话恢复、证据记录、验证与终止条件都由本地代码实现。

运行方式：
1. Windows 下双击 TraceGraph Coder Web.pyw，浏览器会打开本地工作台。
2. 或在命令行进入项目目录，设置 TRACEGRAPH_API_KEY、OPENAI_API_KEY 或 DEEPSEEK_API_KEY 后运行：
   python -m tracegraph_coder --workspace 目标仓库 "你的编程任务"

特色功能：
1. 仓库证据图：启动前索引文件、符号、导入关系、调用关系和测试关联，让模型按需检索，而不是一次性塞入全部代码。
2. 证据驱动执行：每次工具调用都会写入 evidence log，修改前要求已有代码证据，修改后要求验证通过。
3. 工作记忆与会话恢复：保存任务阶段、目标文件、修改记录、验证结果和完整对话，下次可继续同一会话。
4. 失败经验感知上下文压缩：把重复搜索、验证失败、harness 拦截等失败事件结构化为 Failure Control Packet，在后续 prompt 中优先保留相关证据，减少重复犯错。
5. 本地安全边界：路径限制在工作区内，命令拒绝 shell 串联，API key 通过环境变量或本次输入提供，不写入仓库。
