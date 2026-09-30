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
