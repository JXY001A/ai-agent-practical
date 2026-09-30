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
