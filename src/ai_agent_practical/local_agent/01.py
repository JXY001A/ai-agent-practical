import ollama
import sys

MODEL = "qwen3:0.6b"

client = ollama.Client()

# first step
# ========================
# def mina() -> int:
#     client = ollama.Client();
#     try:
#         models = [m.model for m in client.list().models]
#     except Exception as exc:
#         print(f"连不上 Ollama（{exc}）。请先启动服务：")
#         return 1

#     print(f'本机下载模型数量：{len(models)}')
#     for name in models:
#         print(f' --{name}')
    
#     return 0
# if __name__ == "__main__":
#         sys.exit(mina())
# ========================

# second step
# ========================

messages: list[dict] = [
    {"role": "system", "content": "你是一个简洁的助手，回答不超过两句话。"},
]


def ask(client:ollama.Client,question: str) -> str:
    messages.append({"role": "user", "content": question})
    response = client.chat(model=MODEL, messages=messages, stream=False)
    reply = response["message"]
    if reply.get("thinking"):
        print(f"[思考]： {reply['thinking']}")

    print(f"[回复]： {reply['content']}")
    print(
        f"[统计]： 输入 {response.get('prompt_eval_count')} token，"
        f"输出 {response.get('eval_count')} token"
    )
    # 注释实验，丢弃上一步的问题，模型将无法会大第二个问题：“上一步问了啥？”
    messages.append({"role": "assistant", "content": reply["content"]})

    return reply["content"]


def compare_think()-> None:
    question = [{"role":"user","content":"1+1 等于几？只能回答数字"}]
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


def main()-> None:
    # 实验 1 
    # ========================
    # print("======第一轮======")
    # ask(client, "法国首都是哪里？")
    # print("\n ======第二轮======"
    # ask(client, "上一步我问的什么")

    # print(f"\n ====此刻上下文共 {len(messages)} 条消息")
    # # 打印上下文,1 表示计数从 1 开始，默认从 0 开始，仅仅是计数而已
    # for index,message in enumerate(messages, 1):
    #     preview = (message["content"]or '')[:40].replace('\n', ' ')
    #     print(f"{index}. {message['role']:10s} {preview}")
    # ========================

    # 实验 2 
    compare_think()



if __name__ == "__main__":
    main()

