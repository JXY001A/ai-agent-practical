"""本地小模型 Agent 的最小骨架（对应《深入理解 AI Agent》第二章 实验 2-1）。

核心公式：Agent = LLM + 上下文 + 工具。本包用最少的代码把这三件事跑通：

- LLM   —— 本机 Ollama 里的小模型（默认 qwen3:0.6b，约 523 MB）
- 上下文 —— LocalAgent.messages，每轮把助手回复和工具结果追加进去
- 工具   —— ToolRegistry 里注册的 name + description + JSON Schema

与书上完整实验的区别：这里刻意去掉了 vLLM 后端、多后端探测、PDF/代码解释器
等重型依赖，只保留 ollama + requests 两个依赖，方便在 macOS 上快速跑通。

模块：
    tools  —— 工具注册表与两个示例工具
    agent  —— LocalAgent：ReAct 循环与流式事件
    main   —— 命令行入口
"""

from .agent import LocalAgent
from .tools import ToolRegistry, build_default_registry

__all__ = ["LocalAgent", "ToolRegistry", "build_default_registry"]
