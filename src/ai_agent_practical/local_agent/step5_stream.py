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
