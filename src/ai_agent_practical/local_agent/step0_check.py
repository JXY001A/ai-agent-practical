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
