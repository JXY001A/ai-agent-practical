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
