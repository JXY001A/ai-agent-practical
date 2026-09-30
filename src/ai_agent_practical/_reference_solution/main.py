"""命令行入口：单次问答 或 交互对话。

uv run python -m ai_agent_practical.loca_llm_service.main --task "东京现在几点？"
uv run python -m ai_agent_practical.loca_llm_service.main            # 交互模式
uv run local-agent --model qwen3-vl:2b                              # 用已装的其他模型
"""

from __future__ import annotations

import argparse
import sys

from .agent import LocalAgent

RESET = "\033[0m"
GRAY = "\033[90m"


def render(agent: LocalAgent, prompt: str, show_thinking: bool = True) -> None:
    """消费 Agent 的事件流并打印。事件类型见 agent.py 顶部说明。"""
    thinking_printed = False
    for event in agent.chat(prompt):
        kind, content = event["type"], event["content"]

        if kind == "thinking":
            if show_thinking:
                print(f"{GRAY}{content}{RESET}", end="", flush=True)
                thinking_printed = True
        elif kind == "tool_call":
            print(
                f"\n🔧 调用工具 {content.get('name')}，参数 {content.get('arguments')}"
            )
        elif kind == "tool_result":
            print(f"   ✓ {content}")
        elif kind == "content":
            if thinking_printed:
                print()  # 思考之后另起一行，免得推理和正式回复糊在一起
                thinking_printed = False
            print(content, end="", flush=True)
        elif kind == "error":
            print(f"\n❌ {content}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="本地小模型 Agent 最小骨架（Ollama + 工具调用）"
    )
    parser.add_argument(
        "--model", default="qwen3:0.6b", help="Ollama 模型名（默认 qwen3:0.6b）"
    )
    parser.add_argument("--task", help="只执行一条任务后退出；不传则进入交互模式")
    parser.add_argument(
        "--no-thinking", action="store_true", help="不显示模型的思考过程"
    )
    args = parser.parse_args()

    agent = LocalAgent(model=args.model)

    # 启动前先确认模型已下载。书上完整实验在模型缺失时会静默回退到别的模型，
    # 导致「看起来跑通了但结果不可比」；骨架里直接拦下来更清楚。
    try:
        available = agent.available_models()
    except Exception as exc:  # noqa: BLE001 - 服务没起时给出可执行的补救命令
        print(f"❌ 连不上 Ollama（{exc}）。请先启动服务：")
        print("     brew services start ollama     # 或：ollama serve")
        return 1

    if args.model not in available:
        print(f"❌ 本机没有模型 {args.model}。先下载：")
        print(f"     ollama pull {args.model}")
        print(f"   当前可用：{', '.join(available) or '（无）'}")
        return 1

    if args.task:
        render(agent, args.task, show_thinking=not args.no_thinking)
        return 0

    print(f"已连接 Ollama，模型 {args.model}")
    print("命令：/reset 清空上下文　/exit 退出")
    while True:
        try:
            line = input("\n👤 ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue
        if line in {"/exit", "/quit"}:
            return 0
        if line == "/reset":
            agent.reset()
            print("上下文已清空")
            continue
        render(agent, line, show_thinking=not args.no_thinking)


if __name__ == "__main__":
    sys.exit(main())
