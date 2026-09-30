# 参考答案：本地小模型 Agent 最小骨架

> **这是参考答案，不是教程。** 请先按 `docs/tutorial-local-agent.md` 自己一步步写，
> 卡住或写完之后再回来对照。目录名以 `_` 开头就是为了提示这一点。
>
> 对应《深入理解 AI Agent》**第二章 实验 2-1（本地 LLM 服务部署与工具调用）**，只保留最小可运行的部分。

核心公式：**Agent = LLM + 上下文 + 工具**。这个骨架用不到 300 行代码把三件事跑通：

| 要素 | 在本骨架中的实现 |
| --- | --- |
| LLM | 本机 Ollama 里的小模型，默认 `qwen3:0.6b`（约 523 MB） |
| 上下文 | `LocalAgent.messages` —— 每轮把助手回复与工具结果追加进去，是模型唯一的记忆 |
| 工具 | `ToolRegistry` 里的 `name` + `description` + JSON Schema |

与书上完整实验的区别：去掉了 vLLM 后端、多后端自动探测、PDF / 代码解释器等重型依赖，**只依赖 `ollama` 和 `requests`**。

---

## 一、安装

### 1. 机器与前置

本骨架按 **macOS / Apple Silicon** 配置（开发机为 Apple M4 / 24 GB / macOS 27.0）。macOS 没有 CUDA，所以**不需要 `torch`、`vllm`、`transformers`** —— 书上的 `ch2` extra 会拉进 121 个包，这里只需要 17 个。

先确认 Ollama 已安装并在运行：

```bash
brew install ollama
brew services start ollama
curl http://localhost:11434        # 返回 Ollama is running 即正常
```

### 2. 安装 Python 依赖

本项目用 `uv` 管理（`pyproject.toml` 是依赖的唯一来源）：

```bash
cd ai-agent-practical
uv sync
```

不用 uv 时的 pip 兜底：

```bash
python -m pip install -r requirements.txt
```

### 3. 下载模型

```bash
ollama pull qwen3:0.6b             # 约 523 MB
ollama list                        # 确认已下载
```

> 小提示：直接用 `qwen3-vl:*` 等多模态模型也能跑，但它们比 0.6B 大十几倍，加载慢、占用高，而且实验结论（小模型也能可靠调用工具）是围绕 0.6B 设计的。

---

## 二、运行

```bash
# 单次任务
uv run reference-agent --task "东京现在几点？气温是多少？"

# 交互模式（/reset 清空上下文，/exit 退出）
uv run reference-agent

# 换模型 / 不看思考过程
uv run reference-agent --model qwen3-vl:2b --task "东京气温" --no-thinking
```

等价的模块调用方式：

```bash
uv run python -m ai_agent_practical._reference_solution.main --task "东京现在几点？"
```

预期输出（2026-09-28 实测，`--no-thinking` 模式）：

```
🔧 调用工具 get_current_temperature，参数 {'city': 'Tokyo'}
   ✓ Tokyo（Japan）当前气温 20.9°C
东京（Japan）当前气温 20.9°C。
```

一个值得留意的细节：最初工具描述只写了「城市英文名」，0.6B 模型仍然会传中文「东京」导致查询失败。把 `description` 改成明确声明「不支持中文名」之后，模型就直接改用 `Tokyo` 了 —— **只改文字、不改代码**，这就是书上强调「上下文决定 Agent 能力上限」的最直接证据。

---

## 三、代码结构

```
_reference_solution/
├── __init__.py   导出 LocalAgent / ToolRegistry / build_default_registry
├── tools.py      工具注册表 + 两个示例工具（时间、气温）
├── agent.py      LocalAgent：ReAct 循环与流式事件  ← 核心
└── main.py       命令行入口（参数解析 + 事件渲染）
```

建议阅读顺序：`tools.py` → `agent.py` → `main.py`。

---

## 四、四个设计要点

**1. `think` 控制的是「要不要思维链」，不是「思考存到哪个字段」。**
实测（Ollama 0.34 + qwen3:0.6b）：

| 调用方式 | `thinking` 字段 | `content` |
| --- | --- | --- |
| 不传 `think` | 有值 | 干净的最终回答 |
| `think=True` | 有值 | 干净的最终回答 |
| `think=False` | 空 | 直接给答案，没有推理过程 |

三种情况下推理都不会混进 `content` —— 它始终在独立的 `thinking` 字段里。
`LocalAgent` 把 `think` 暴露成构造参数：不传用模型默认（qwen3 默认开），
传 `False` 可以省 token 和延迟。想看模型有没有思维能力：`ollama show <模型>`。

**2. ReAct 循环的出口是「模型不再请求工具」。**
`agent.py` 的主循环没有"任务完成"判断，唯一正常退出条件是某一轮模型没产出 `tool_calls` —— 也就是模型自己认为信息够了。另有 `max_steps=8` 作为防死循环的保险。

**3. 同一批工具调用并行执行。**
它们由模型在没看到任何结果时一次性生成，因此彼此独立。用 `ThreadPoolExecutor.map` 保序执行。如果后一个工具的参数依赖前一个的输出，模型只能在下一轮再调用。

**4. 助手消息要原样保留。**
不只是存文本，还要存 `tool_calls`。这既满足多轮消息契约，也让下一轮请求的消息前缀与这一轮严格一致 —— 这是 KV Cache 能复用的前提（详见书上第二章 KV Cache 一节）。

---

## 五、常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `❌ 连不上 Ollama` | 服务没起：`brew services start ollama` 或 `ollama serve` |
| `❌ 本机没有模型 xxx` | 按提示 `ollama pull xxx`；骨架会拦下未下载的模型，不会静默换模型 |
| 首次回答特别慢 | 模型首次加载进内存，第二次会明显变快（Ollama 默认驻留 5 分钟） |
| 看不到思考内容 | 模型不支持思维链，或被 `think=False` 关掉了。`ollama show <模型>` 看 Capabilities 里有没有 thinking |
| 时区 / 城市理解错 | 0.6B 小模型的正常表现。改进方向是优化工具 `description`，而不是换代码 |
| `ZoneInfoNotFoundError` | 系统缺时区库，装 `tzdata`（`uv add tzdata`） |

---

## 六、下一步可以改什么

- **加工具**：在 `tools.py` 的 `build_default_registry()` 里注册新的 `Tool`，重点打磨 `description`
- **观察上下文**：在 `agent.py` 追加消息处打印 `self.messages`，看上下文如何一轮轮变长
- **验证 KV Cache**：保持系统提示词不变连续问两次，对比首 token 延迟；再改动提示词开头几个字符重试
- **加压缩**：上下文接近模型上限时，把旧工具结果成批替换成摘要（书上实验 2-10 的主题）
