# 从零实现一个本地 Agent：分步教程

> 对应《深入理解 AI Agent》**第二章 实验 2-1（本地 LLM 服务部署与工具调用）**。
> 每一步都给出完整、可直接运行的代码，同时保留「先理解 / 验证 / 检查点」，让你边跑边想。

## 这份教程怎么用

- **每一步都给出完整代码**，落在 `src/ai_agent_practical/local_agent/` 下，可直接运行。
- **但还是建议自己敲一遍。** 照着跑通只是「见过」，自己敲才会暴露真正的理解缺口；卡住了再对照。
- **每步都是一个能独立运行的小脚本**，跑通了再进下一步，不要一次写完。
- 步骤 1–5 是学习用的脚本，第 6 步把它们整理成正式模块。
- 第 6 步的成品在 `src/ai_agent_practical/_reference_solution/`，写完再对照。

## 最终会得到什么

一个能在本机跑起来的小 Agent：你说「东京现在几点？气温多少？」，它会自己决定调用哪两个工具、并行执行、拿到结果后再组织成一句回答。

书里的核心公式贯穿全程：

> **Agent = LLM + 上下文 + 工具**

| 要素 | 你将怎么写 |
| --- | --- |
| LLM | 连本机 Ollama，用 `qwen3:0.6b` |
| 上下文 | 一个 `messages` 列表，每轮自己维护 |
| 工具 | 一段 JSON Schema（给模型看）+ 一个 Python 函数（真正执行） |

## 学习地图

| 步骤 | 你会实现 | 掌握的概念 |
| :--: | --- | --- |
| 0 | 环境自检 | Ollama 客户端怎么用 |
| 1 | 让模型开口说话 | `messages`、上下文、`think` |
| 2 | 把工具「介绍」给模型 | 工具 schema、`tool_calls` |
| 3 | 执行工具并把结果还回去 | 观察结果、消息角色 |
| 4 | 包成 ReAct 循环 | 循环出口、轮数上限 |
| 5 | 流式输出 + 并行工具 | 生成器、chunk 处理 |
| 6 | 整理成模块 | 事件抽象、CLI |

---

## 准备工作

### 1. 环境自检

```bash
cd ai-agent-practical
uv sync                                    # 安装依赖（只有 ollama + requests）
ollama pull qwen3:0.6b                     # 约 523 MB
ollama show qwen3:0.6b | head -20          # 看 Capabilities
```

`ollama show` 的输出里应该能看到：

```
  Capabilities
    completion
    tools          ← 支持工具调用
    thinking       ← 支持思维链
```

**没有 `tools` 这一项，本教程后续全部走不通。**

### 2. 代码位置

六个步骤的完整代码已经放在 `src/ai_agent_practical/local_agent/` 下了：

```bash
ls src/ai_agent_practical/local_agent/
# __init__.py            step3_execute.py
# step0_check.py         step4_react.py
# step1_chat.py          step5_stream.py
# step2_see_tool_calls.py
```

建议**自己照着敲一遍**，再运行；想先看效果也可以直接跑。两种运行方式等价：

```bash
uv run python src/ai_agent_practical/local_agent/step0_check.py
```

---

## 第 0 步：先能连上模型

**目标**：确认 Python 能连到本机 Ollama，并能列出模型。

**API 速查**

```python
import ollama

client = ollama.Client()          # 默认连 http://127.0.0.1:11434
client.list().models              # 已经下载的模型
```

**完整代码**（`src/ai_agent_practical/local_agent/step0_check.py`）

```python
"""第 0 步：确认能连上本机 Ollama，并准备好实验用的小模型。

运行：
    uv run python src/ai_agent_practical/local_agent/step0_check.py
"""

import sys

import ollama

MODEL = "qwen3:0.6b"


def main() -> int:
    client = ollama.Client()  # 默认连 http://127.0.0.1:11434

    try:
        models = [m.model for m in client.list().models]
    except Exception as exc:  # noqa: BLE001 - 服务没起时给出可执行的补救命令
        print(f"连不上 Ollama（{exc}）。请先启动服务：")
        print("    brew services start ollama    # 或：ollama serve")
        return 1

    print(f"本机已下载 {len(models)} 个模型：")
    for name in models:
        print(f"  - {name}")

    if MODEL not in models:
        print(f"\n缺少实验用的小模型，请先执行：ollama pull {MODEL}")
        return 1

    print(f"\n环境就绪：{MODEL} 已就位。")
    print(
        "提示：这个脚本全程没有加载模型——模型是 Ollama 服务端在第一次请求时才读进内存的。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**验证**

```bash
uv run python src/ai_agent_practical/local_agent/step0_check.py
```

期望看到模型名列表，且 `qwen3:0.6b` 在列表里。

**检查点**

- 这个脚本里有没有任何「加载模型」的动作？模型是在什么时候被加载进内存的？

---

## 第 1 步：让模型开口（还不用工具）

**目标**：完成一次最简单的对话，并理解 `messages` 就是全部上下文。

**先理解**

大模型 API 是**无状态**的：它不记得你上一句说过什么。所谓「记忆」，是每次请求时你把历史消息一起发过去。所以 `messages` 列表就是 Agent 的上下文，也就是模型眼里的全部世界。

消息有三种基础角色，你要用到前两种：

| role | 谁写的 |
| --- | --- |
| `system` | 你（开发者），设定身份和规则 |
| `user` | 用户输入 |
| `assistant` | 模型自己的回复（下一轮要带回去） |
| `tool` | 工具执行结果（第 3 步才用） |

**API 速查**

```python
response = client.chat(
    model="qwen3:0.6b",
    messages=[{"role": "user", "content": "你好"}],
    stream=False,
)
response["message"]["content"]     # 正式回答
response["message"]["thinking"]    # 内部思考（推理过程）
response["eval_count"]             # 本次生成的 token 数
```

`response["message"]` 上你会用到的字段：`role`、`content`、`thinking`、`tool_calls`、`tool_name`。

**关于 `think` 参数**（实测结论，请记住）：

| 调用方式 | `thinking` 字段 | `content` |
| --- | --- | --- |
| 不传 `think` | 有值 | 干净的最终回答 |
| `think=True` | 有值 | 干净的最终回答 |
| `think=False` | 空 | 直接给答案，没有推理过程 |

所以 `think` 是**「要不要思维链」的开关**，不是「思考放哪个字段」。qwen3 默认就开。

**完整代码**（`src/ai_agent_practical/local_agent/step1_chat.py`）

```python
"""第 1 步：最简对话，理解 messages 就是模型的全部上下文。

运行：
    uv run python src/ai_agent_practical/local_agent/step1_chat.py

要记住的一句话：大模型 API 是无状态的。它之所以能"记住"上一轮，
是因为你每次请求时把历史消息一起发了过去。
"""

import ollama

MODEL = "qwen3:0.6b"

# system 提示词由开发者编写，定义 Agent 的身份与规则，放在消息列表最前面
messages: list[dict] = [
    {"role": "system", "content": "你是一个简洁的助手，回答不超过两句话。"},
]


def ask(client: ollama.Client, question: str) -> str:
    """发一次请求：打印思考与回复，并把回复追加进历史。"""
    messages.append({"role": "user", "content": question})

    response = client.chat(model=MODEL, messages=messages, stream=False)
    reply = response["message"]

    if reply.get("thinking"):
        print(f"[思考] {reply['thinking']}")
    print(f"[回复] {reply['content']}")
    print(
        f"[统计] 输入 {response.get('prompt_eval_count')} token，"
        f"输出 {response.get('eval_count')} token"
    )

    # 关键的一行：把模型自己的回复也存回历史。
    # 注释掉它，下一轮模型就会"失忆"——因为上下文里没有它说过的话。
    messages.append({"role": "assistant", "content": reply["content"]})
    return reply["content"]


def compare_think() -> None:
    """对比 think 参数的两种取值。

    结论（实测）：think 是「要不要思维链」的开关，不是「思考存到哪个字段」。
    三种情况下推理都不会混进 content，它始终在独立的 thinking 字段里。
    """
    client = ollama.Client()
    question = [{"role": "user", "content": "1+1等于几？只回答数字"}]

    for label, kwargs in [
        ("不传 think", {}),
        ("think=True", {"think": True}),
        ("think=False", {"think": False}),
    ]:
        response = client.chat(model=MODEL, messages=question, stream=False, **kwargs)
        message = response["message"]
        thinking = message.get("thinking") or ""
        print(
            f"{label:12s} thinking 长度={len(thinking):4d}   "
            f"content={message.get('content')!r}"
        )


def main() -> None:
    client = ollama.Client()

    print("=== 第一轮 ===")
    ask(client, "法国的首都是哪里？")

    print("\n=== 第二轮（看它记不记得上一轮）===")
    ask(client, "我刚才问了你什么？")

    print(f"\n=== 此刻上下文共 {len(messages)} 条消息 ===")
    for index, message in enumerate(messages, 1):
        preview = (message["content"] or "")[:32].replace("\n", " ")
        print(f"  {index}. {message['role']:9s} {preview}")

    print(
        "\n=== 实验一：把上面追加 assistant 的那一行注释掉，重跑，看第二轮是否失忆 ==="
    )
    print()
    print("=== 实验二：think 参数到底控制什么 ===")
    compare_think()


if __name__ == "__main__":
    main()
```

**验证**

- 第 2 点：能看到一段推理文本和一段正式回答。
- 第 3 点：模型能正确说出你上一个问题。
- 第 4 点：答不出来或答错 —— 因为历史没带上。
- 第 5 点：`thinking` 为空，回答变短。

**检查点**

- 为什么注释掉「追加 assistant 消息」之后，模型就失忆了？
- `temperature` 调高会有什么变化？（提示：看 `ollama show qwen3:0.6b` 里的默认值）

---

## 第 2 步：把工具介绍给模型（先只看，不执行）

**目标**：让模型知道「有哪些工具可用」，并看懂它返回的 `tool_calls` 长什么样。

**先理解**

这是整个实验最反直觉的一点：**模型看不到你的 Python 函数**。它只看到一段 JSON —— 工具的名字、用途描述、参数结构。它据此判断「该不该调、调哪个、参数填什么」。

所以工具描述写得含糊，模型就会调错或漏调；写清楚，小模型也能调对。这不是代码问题，是**上下文质量问题**。

**工具 schema 的真实格式**（Ollama 和 OpenAI 通用）：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "查询指定时区的当前日期和时间。时区用 IANA 名称，如 Asia/Tokyo。",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA 时区名，不传按 UTC"}
                },
                "required": [],
            },
        },
    }
]
```

**完整代码**（`src/ai_agent_practical/local_agent/step2_see_tool_calls.py`）

```python
"""第 2 步：把工具「介绍」给模型 —— 只给说明书，不给实现。

运行：
    uv run python src/ai_agent_practical/local_agent/step2_see_tool_calls.py

关键认知：模型看不到你的 Python 函数。它只看到 name / description / parameters
这三样东西。所以描述写得含糊，模型就会调错或漏调——这不是代码问题，是上下文问题。
"""

import json

import ollama

MODEL = "qwen3:0.6b"

# 工具说明书（JSON Schema）。注意：这里只有「描述」，没有任何实现代码。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": (
                "查询某个城市的实时气温（摄氏度）。数据来自 Open-Meteo，无需 API Key。"
                "注意：city 必须是英文或拼音地名（如 Tokyo、Shanghai），不支持中文名。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市英文名，例如 Tokyo、Shanghai、Vancouver",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


def main() -> None:
    client = ollama.Client()
    question = "东京现在气温多少？"
    print(f"提问：{question}\n")

    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        tools=TOOLS,
        stream=False,
        think=False,
    )
    message = response["message"]
    calls = message.get("tool_calls") or []

    print("模型返回的 tool_calls：")
    if calls:
        print(
            json.dumps(
                [call.model_dump() for call in calls], ensure_ascii=False, indent=2
            )
        )
    else:
        print(f"（模型没有调用工具，直接回答了：{message.get('content')!r}）")

    print("\n三个值得注意的点：")
    print("1. 模型并没有真的执行任何东西——它只是发起了一个调用请求")
    print("2. arguments 已经是 dict，不是 JSON 字符串（这是 Ollama 原生接口的行为；")
    print("   换成 OpenAI 兼容接口时给的是字符串，需要自己 json.loads）")
    print("3. 如果模型没调用工具，先确认 tools 是否真的传进去了")


if __name__ == "__main__":
    main()
```

**验证**

期望看到类似这样的结构（实测输出）：

```json
[
  {
    "function": {
      "name": "get_current_temperature",
      "arguments": { "city": "Tokyo" }
    }
  }
]
```

注意 `arguments` 已经是 **Python dict**，不是 JSON 字符串 —— 这是 Ollama 原生接口的行为。后面如果你改用 OpenAI 兼容接口，那里给的是字符串，需要自己 `json.loads`。

**检查点**

- 模型在这一步真的执行了工具吗？它为什么不知道天气？
- 把 `description` 改得含糊一点（比如只写「查天气」），行为有变化吗？
- 在 schema 的 `required` 里去掉 `city`，模型还会填吗？

---

## 第 3 步：真正执行工具并把结果还回去

**目标**：完成一个完整的「模型请求 → 你执行 → 结果回填 → 模型作答」闭环。

**先理解**

工具执行结果要以 `role="tool"` 的消息追加进 `messages`，然后**再请求一次**模型。模型看到结果后，才会给出面向用户的自然语言回答。

这一步的关键认知：**工具返回的文本会进入上下文，影响模型下一步动作。** 所以报错信息也要写清楚，让模型有机会自己改参数重试。

**API 速查**

```python
messages.append({
    "role": "tool",
    "tool_name": "get_current_temperature",   # 建议带上，明确是哪个工具的结果
    "content": "Tokyo 当前气温 20.9°C",        # 工具返回值转成字符串
})
```

**完整代码**（`src/ai_agent_practical/local_agent/step3_execute.py`）

```python
"""第 3 步：执行工具并把结果还回去 —— 完成一次完整的闭环。

运行：
    uv run python src/ai_agent_practical/local_agent/step3_execute.py

流程：模型请求工具 → 你执行 → 结果以 role="tool" 追加进历史 → 再请求一次模型。
"""

import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import ollama
import requests

MODEL = "qwen3:0.6b"


# ---------------------------------------------------------------- 工具实现
# 模型看不到下面这些函数，它只看到后面 TOOLS 那段的说明书。


def get_current_time(timezone: str = "UTC") -> str:
    """查询指定时区的当前时间（只用标准库）。"""
    return datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S %Z")


def get_current_temperature(city: str) -> str:
    """查询城市实时气温。Open-Meteo 免费且无需 API Key。

    两步：地名 → 经纬度，再用经纬度查气温。
    """
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    ).json()
    if not geo.get("results"):
        # 错误信息也要「可行动」：告诉模型该怎么改参数重试。
        # 工具返回的文本会进入上下文，直接影响模型下一步动作。
        return (
            f"找不到城市「{city}」。该接口只认英文或拼音地名，"
            f"请改用英文名重新调用，例如 Tokyo、Shanghai、Vancouver。"
        )

    hit = geo["results"][0]
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": hit["latitude"],
            "longitude": hit["longitude"],
            "current": "temperature_2m",
        },
        timeout=10,
    ).json()
    return (
        f"{hit['name']}（{hit.get('country', '')}）当前气温 "
        f"{weather['current']['temperature_2m']}°C"
    )


# 工具名 → 函数。模型给的是名字，靠这张表找到要执行的代码。
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_current_temperature": get_current_temperature,
}

# 工具说明书：给模型看的部分
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "查询指定时区的当前日期和时间。"
                "时区使用 IANA 名称，例如 Asia/Tokyo、Asia/Shanghai、Europe/London。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA 时区名，例如 Asia/Tokyo；不传则按 UTC",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": (
                "查询某个城市的实时气温（摄氏度）。数据来自 Open-Meteo，无需 API Key。"
                "注意：city 必须是英文或拼音地名（如 Tokyo、Shanghai），不支持中文名。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市英文名，例如 Tokyo、Shanghai、Vancouver",
                    }
                },
                "required": ["city"],
            },
        },
    },
]


def main() -> None:
    # 支持从命令行传入问题，方便做实验：
    #     uv run python .../step3_execute.py "东京气温多少？"
    question = sys.argv[1] if len(sys.argv) > 1 else "东京现在气温多少？"

    client = ollama.Client()
    messages: list[dict] = [{"role": "user", "content": question}]

    # ---- 第 1 次请求：模型决定要调用哪个工具 ----
    response = client.chat(
        model=MODEL, messages=messages, tools=TOOLS, stream=False, think=False
    )
    message = response["message"]
    calls = message.get("tool_calls") or []

    if not calls:
        print("模型没有调用工具，直接回答：", message.get("content"))
        return

    # ---- 关键：assistant 消息要带上 tool_calls 一起存进历史 ----
    # 少存了 tool_calls，模型就不知道自己刚才请求过什么。
    messages.append(
        {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": calls,
        }
    )

    # ---- 逐个执行，把结果作为 role="tool" 追加 ----
    for call in calls:
        name = call["function"]["name"]
        arguments = call["function"]["arguments"]
        print(f"[执行] {name}({arguments})")

        result = TOOL_FUNCTIONS[name](**arguments)
        print(f"[结果] {result}")

        messages.append({"role": "tool", "tool_name": name, "content": result})

    # ---- 第 2 次请求：模型看到结果后，给出面向用户的回答 ----
    final = client.chat(
        model=MODEL, messages=messages, tools=TOOLS, stream=False, think=False
    )
    print(f"\n[最终回答] {final['message']['content']}")

    print(f"\n此刻上下文共 {len(messages)} 条消息：")
    for index, item in enumerate(messages, 1):
        print(f"  {index}. {item.get('role'):9s} {str(item.get('content'))[:40]}")

    print("\n=== 实验：把 TOOLS 里「不支持中文名」那句删掉，再用中文城市名提问 ===")
    print(
        '    uv run python src/ai_agent_practical/local_agent/step3_execute.py "东京气温多少？"'
    )
    print("描述变弱后，模型就会传中文「东京」；工具返回「找不到城市…请改用英文名」，")
    print("模型读到这句提示，自己改用 Tokyo 重新调用并成功。")
    print("只改文字、不改代码 —— 这就是「上下文决定能力上限」最直接的证据。")


if __name__ == "__main__":
    main()
```

**额外练习（重要）**

把 `TOOLS` 里 `get_current_temperature` 描述末尾那句「注意：city 必须是英文或拼音地名…不支持中文名」删掉，再用中文城市名提问：

```bash
uv run python src/ai_agent_practical/local_agent/step3_execute.py "东京气温多少？"
```

描述变弱后，模型就会传中文「东京」；工具返回「找不到城市…请改用英文名」，模型读到这句提示后自己改用 `Tokyo` 重新调用并成功。**只改文字、不改代码** —— 这是「上下文决定能力上限」最直接的证据。

> 保留那句提示时，模型会直接传 `Tokyo`，实验就观察不到失败。可以两个版本各跑一次对比。

**验证**

问「东京现在气温多少？」，期望得到一句带真实温度的完整回答，且温度随时间变化（多跑几次）。

**检查点**

- 这个练习印证了书里的哪句话？（提示：上下文决定能力上限）
- 工具执行抛异常时，直接让程序崩掉 vs 把异常信息作为结果回填，哪个更好？为什么？

---

## 第 4 步：包成 ReAct 循环

**目标**：把「请求 → 执行 → 回填 → 再请求」包进一个循环，支持多步任务。

**先理解**

ReAct = Reasoning（推理）+ Acting（行动）交替。一次循环 = 模型思考并决定行动 → 你执行 → 把观察结果还回去。循环直到模型认为信息够了。

**关键问题：循环什么时候结束？**

不是「任务完成」，而是**某一轮模型没有返回任何 `tool_calls`**。也就是模型自己判断「我不需要再查了」，直接给出最终回答。另外一定要加一个**最大轮数上限**兜底 —— 小模型偶尔会反复调同一个工具。

**完整代码**（`src/ai_agent_practical/local_agent/step4_react.py`）

```python
"""第 4 步：把「请求 → 执行 → 回填 → 再请求」包成 ReAct 循环。

运行：
    uv run python src/ai_agent_practical/local_agent/step4_react.py

循环的唯一正常出口：某一轮模型没有返回任何 tool_calls。
也就是说，是「模型自己判断信息够了」，而不是你判断任务完成了。

工具定义直接从 step3 复用，避免重复抄一遍。
"""

import ollama

from ai_agent_practical.local_agent.step3_execute import TOOL_FUNCTIONS, TOOLS

MODEL = "qwen3:0.6b"
MAX_STEPS = 8  # 防死循环的保险：小模型偶尔会反复调同一个工具


def run_tools(calls: list, messages: list[dict]) -> None:
    """执行本轮的每个工具调用，并把结果追加进 messages。"""
    for call in calls:
        name = call["function"]["name"]
        arguments = call["function"]["arguments"]
        print(f"  [执行] {name}({arguments})")

        result = TOOL_FUNCTIONS[name](**arguments)
        print(f"  [结果] {result}")

        messages.append({"role": "tool", "tool_name": name, "content": result})


def main() -> None:
    client = ollama.Client()
    messages: list[dict] = [{"role": "user", "content": "东京现在几点？气温多少？"}]

    for step in range(1, MAX_STEPS + 1):
        print(f"--- 第 {step} 轮 ReAct（上下文已积累 {len(messages)} 条消息）---")

        response = client.chat(
            model=MODEL, messages=messages, tools=TOOLS, stream=False, think=False
        )
        message = response["message"]
        calls = message.get("tool_calls") or []

        # 出口：模型不再请求工具 → 它认为信息够了，给出最终回答
        if not calls:
            content = message.get("content") or ""
            messages.append({"role": "assistant", "content": content})
            print(f"\n[最终回答] {content}")
            print(f"[共 {step} 轮，最终上下文 {len(messages)} 条消息]")
            return

        print(f"  模型请求了 {len(calls)} 个工具")

        # 把助手这一轮的「说了什么 + 要调什么」原样存进历史
        messages.append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": calls,
            }
        )
        run_tools(calls, messages)

    print(f"\n达到最大轮数 {MAX_STEPS}，提前结束（可能是模型陷入了重复调用）")


if __name__ == "__main__":
    main()
```

**验证**

用一个需要两步的提问，例如「东京现在几点？气温多少？」。期望看到**两轮**：

- 第 1 轮：模型一次要求调用两个工具（时间和气温）
- 第 2 轮：模型拿到结果后给出最终回答，不再要工具

**检查点**

- 为什么同一轮里可以有两个工具调用，而不是一个接一个？
- 如果第二个工具的参数必须来自第一个工具的结果，会发生什么？（试试问「现在东京的气温换算成华氏度是多少」，体会区别）
- `max_steps` 设成 1 会怎样？

---

## 第 5 步：流式输出 + 并行执行

**目标**：边生成边显示，看清模型的思考与行动；并让同一批工具并行执行。

**先理解**

流式（`stream=True`）时，返回的不再是一个完整响应，而是一个**逐块的生成器**。实测的块分布很有代表性：一次带工具调用的回答约 100 个 chunk，其中**前 98 个都是 `thinking`**，`tool_calls` 出现在接近末尾的位置。

这意味着两件事：

1. 思考会先哗啦啦流出来，正文和工具调用在后头。
2. **必须先把整个流收完，再执行工具** —— 因为工具调用可能出现在任何一个 chunk 里。

**API 速查**

```python
for chunk in client.chat(model=..., messages=..., tools=..., stream=True):
    msg = chunk.get("message", {})
    msg.get("thinking")      # 可能有
    msg.get("content")       # 可能有
    msg.get("tool_calls")    # 可能有，是 list
```

**完整代码**（`src/ai_agent_practical/local_agent/step5_stream.py`）

```python
"""第 5 步：流式输出 + 并行执行工具。

运行：
    uv run python src/ai_agent_practical/local_agent/step5_stream.py

流式返回的不是一个完整响应，而是一个逐块的生成器。
实测一次带工具调用的回答约 100 块，其中前 90 多块都是 thinking，
tool_calls 出现在接近末尾的位置——所以「先收完整个流，再执行工具」。
"""

from concurrent.futures import ThreadPoolExecutor

import ollama

from ai_agent_practical.local_agent.step3_execute import TOOL_FUNCTIONS, TOOLS

MODEL = "qwen3:0.6b"
MAX_STEPS = 8
GRAY = "\033[90m"  # 终端灰色，用来区分思考与正文
RESET = "\033[0m"


def run_tools(calls: list) -> list[tuple[str, str]]:
    """执行一批工具调用，返回 [(工具名, 结果), ...]。

    同一批里的调用可以并行：它们是模型在没看到任何结果的情况下一次性生成的，
    因此彼此独立。executor.map 保证返回顺序与请求顺序一致。
    如果后一个工具的参数依赖前一个的结果，模型只能等下一轮——那种情况不会
    出现在同一批里。
    """

    def run_one(call: dict) -> tuple[str, str]:
        name = call["function"]["name"]
        arguments = call["function"]["arguments"]
        return name, TOOL_FUNCTIONS[name](**arguments)

    if len(calls) == 1:
        return [run_one(calls[0])]
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return list(pool.map(run_one, calls))


def main() -> None:
    client = ollama.Client()
    messages: list[dict] = [{"role": "user", "content": "东京现在几点？气温多少？"}]

    for step in range(1, MAX_STEPS + 1):
        text_parts: list[str] = []
        pending: list = []

        print(f"\n--- 第 {step} 轮 ---")

        # 不传 think，用模型默认（qwen3 默认开思维链），所以能看到灰色推理文本
        for chunk in client.chat(
            model=MODEL, messages=messages, tools=TOOLS, stream=True
        ):
            message = chunk.get("message", {})

            if message.get("thinking"):
                print(f"{GRAY}{message['thinking']}{RESET}", end="", flush=True)

            if message.get("content"):
                text_parts.append(message["content"])
                print(message["content"], end="", flush=True)

            for call in message.get("tool_calls") or []:
                # 去重：某些服务端会在多个块里重复发送累积的工具列表，
                # 不去重就会重复执行同一个工具
                if call not in pending:
                    pending.append(call)

        print()

        assistant: dict = {"role": "assistant", "content": "".join(text_parts)}
        if not pending:
            messages.append(assistant)
            print(f"\n[结束] 共 {step} 轮")
            return

        assistant["tool_calls"] = pending
        messages.append(assistant)

        # 必须等整个流结束再执行——工具调用可能出现在任意一块里
        for name, result in run_tools(pending):
            print(f"  ✓ {name} → {result}")
            messages.append({"role": "tool", "tool_name": name, "content": result})

    print(f"\n达到最大轮数 {MAX_STEPS}，提前结束")


if __name__ == "__main__":
    main()
```

**验证**

跑一个多工具任务，能看到四类内容按顺序出现，且两个工具是并行执行的（总耗时接近较慢的那个，而不是两者之和）。

**检查点**

- 为什么必须等流结束再执行工具？能不能收到一个就执行一个？
- 并行执行的前提是什么？为什么这些调用一定互相独立？
- 为什么要去重？

---

## 第 6 步：整理成模块

**目标**：把前面 5 个脚本整理成可复用的三个模块。这一步才是「工程化」的开始。

**完整代码**

第 6 步的产物就是一个完整模块，已经放在参考答案里了，**不用再抄一遍**：

```
src/ai_agent_practical/_reference_solution/
├── tools.py      Tool 数据类 + ToolRegistry（schema 与函数绑在一起）
├── agent.py      LocalAgent：流式 ReAct 循环 + 事件生成器
├── main.py       CLI：参数解析 + 事件渲染
└── README.md     使用说明与设计要点
```

**建议先自己写**，写完再逐文件对照。重点对比这三件事：

1. `tools.py` 怎么把「给模型看的 schema」和「真正执行的函数」绑在同一个对象上，并让 `call()` 捕获异常、把错误当成结果返回（而不是抛出去）
2. `agent.py` 怎么用**生成器**把五类事件吐出来，而不是直接 `print`（这样同一套 Agent 逻辑能配终端、Web、日志三种前端）
3. `main.py` 怎么把「终端渲染」从 Agent 逻辑里剥离出去

写完用你自己的模块验证（路径换成你写的）：

```bash
uv run python -m ai_agent_practical.local_agent.main --task "东京现在几点？气温多少？"
```

对照参考答案运行效果：

```bash
uv run reference-agent --task "东京现在几点？气温多少？"
```

**验证**

```bash
uv run python -m ai_agent_practical.local_agent.main --task "东京现在几点？气温多少？"
```

期望与第 5 步行为一致。

**检查点**

- 为什么 `chat()` 返回事件生成器，而不是直接打印？这样做的好处是什么？
- 如果以后要加一个「把工具结果存数据库」的需求，改哪里最合适？

---

## 完成后的自测清单

全部能答上来，说明这个示例你真的掌握了：

- [ ] 能画出 `messages` 在多轮 ReAct 中每一步的变化（谁追加了什么）
- [ ] 能说清循环的唯一正常出口是什么
- [ ] 能解释 `think` 参数到底控制什么（别答成「控制思考放哪个字段」）
- [ ] 能解释为什么「工具描述」比「工具实现」更影响调用成功率
- [ ] 能解释为什么助手消息必须原样保留 `tool_calls`
- [ ] 能解释为什么同一批工具可以并行、而跨批次不行
- [ ] 能自己加第三个工具（比如查汇率），并让它被正确调用

## 常见坑

| 现象 | 原因 |
| --- | --- |
| 报 404 / 模型不存在 | 模型没下载，或名字写错（`ollama list` 核对） |
| 模型从不调用工具 | schema 没放进 `tools` 参数，或模型不支持 `tools`（`ollama show` 确认） |
| 工具调用后模型重复调用同一个工具 | 忘了把 `tool` 消息追加进 `messages`，模型看不到结果 |
| 多轮对话答不上上一轮 | 忘了把 `assistant` 消息追加进历史 |
| 流式时工具被重复执行 | 没做 `tool_calls` 去重 |
| 首次响应特别慢 | 模型首次加载进内存；第二次会明显变快 |
| `ZoneInfoNotFoundError` | 缺时区库：`uv add tzdata` |

## 接下来可以做什么

- **观察 KV Cache**：保持 `system` 提示词不变连续问两次，对比首 token 延迟；再改动提示词开头几个字符重试（书上第二章 KV Cache 一节）
- **上下文压缩**：接近上下文上限时把旧工具结果替换成摘要（书上实验 2-10）
- **加系统提示词**：在 `messages` 开头放一条 `system`，观察对工具选择的影响（书上实验 2-9）
- **换更大的模型**：`--model qwen3-vl:4b` 试试，对比工具调用准确率

## 参考答案

`src/ai_agent_practical/_reference_solution/`，目录名以 `_` 开头提醒它是参考实现。

```bash
uv run reference-agent --task "东京现在几点？气温多少？"
```

建议节奏：**每一步自己写完后**再去对照该部分。直接抄会失去全部学习价值 —— 这个示例里真正值钱的东西，是你调试「模型为什么不按预期调用工具」时积累的手感。

## 下一篇

跑通之后，真正会绊住你的是另外一些问题：怎么调试、上下文为什么越来越贵、工具描述怎么改才有效、Agent 会被人「下套」、以及怎么判断改动变好了。这些在第二篇：

**→ [`advanced-agent-development.md`](advanced-agent-development.md)（进阶篇：从「跑通」到「写好」）**

九个主题，含本机实测数据（工具 schema 的 token 开销、KV Cache 带来的 30 倍差异），都附了可复现的命令。
