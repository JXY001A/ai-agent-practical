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
