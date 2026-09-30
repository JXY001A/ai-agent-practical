"""LocalAgent：Ollama + 工具调用 + ReAct 循环的最小实现。

只保留一条正确路径（流式 ReAct 循环），刻意不做非流式分支 —— 书里
非流式实现只处理一轮工具调用，多步任务会提前收尾，对学习者是个干扰。

事件类型（chat() 逐条产出，交给上层决定怎么显示）：
    thinking     模型的内部思考（仅带思维链的模型会有）
    tool_call    模型请求调用某工具，含解析后的参数
    tool_result  工具执行结果（即 ReAct 里的「观察」）
    content      给用户的正式回复
    error        异常或超过最大步数
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import ollama

from .tools import ToolRegistry, build_default_registry

Event = dict[str, Any]


class LocalAgent:
    """把「模型选择工具 → 执行 → 回填结果 → 再决策」这个循环封装成一个 chat()。"""

    def __init__(
        self,
        model: str = "qwen3:0.6b",
        tools: ToolRegistry | None = None,
        temperature: float = 0.3,
        max_steps: int = 8,
        think: bool | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_steps = max_steps  # ReAct 最多循环几轮，防止小模型反复调同一个工具
        # think 是「要不要思维链」的开关，不是「思考存到哪个字段」：
        # None = 用模型默认（qwen3 默认开）；True / False = 强制开 / 关。
        # 关掉能省 token 和延迟，但会失去最直观的调试线索。
        self.think = think
        self.registry = tools or build_default_registry()
        # client 只是 HTTP 客户端，默认连 http://127.0.0.1:11434；
        # 模型权重由 Ollama 服务端在第一次请求时才加载进内存。
        self.client = ollama.Client()
        self.messages: list[dict[str, Any]] = []  # 上下文：模型唯一的记忆

    # ------------------------------------------------------------------ 公开接口

    def available_models(self) -> list[str]:
        """列出本机已下载的模型，供启动前校验（避免拿到一个看不懂的 404）。"""
        return [m.model for m in self.client.list().models]

    def reset(self) -> None:
        """清空上下文。上下文是模型唯一的记忆，清掉即彻底失忆。"""
        self.messages.clear()

    def chat(self, message: str) -> Iterator[Event]:
        """跑一轮完整对话，逐条产出事件。这是一个生成器，边跑边 yield。"""
        self.messages.append({"role": "user", "content": message})

        for _ in range(self.max_steps):
            text_parts: list[str] = []
            pending: list[dict[str, Any]] = []  # 本轮模型请求的所有工具调用

            # ── 第 1 步：把消息 + 工具定义发给模型，边收边转发 ──
            request: dict[str, Any] = {
                "model": self.model,
                "messages": self.messages,
                "tools": self.registry.schemas(),
                "options": {"temperature": self.temperature},
                "stream": True,
            }
            if self.think is not None:
                request["think"] = self.think

            for chunk in self.client.chat(**request):
                msg = chunk.get("message", {})

                if msg.get("thinking"):
                    yield {"type": "thinking", "content": msg["thinking"]}
                if msg.get("content"):
                    text_parts.append(msg["content"])
                    yield {"type": "content", "content": msg["content"]}

                for call in msg.get("tool_calls") or []:
                    # 某些服务端会在每个 chunk 里重复发送累积的工具列表，需去重
                    if call not in pending:
                        pending.append(call)
                        yield {"type": "tool_call", "content": call["function"]}

            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_parts),
            }

            # ── 第 2 步：模型不再请求工具 = 它认为信息够了，循环结束 ──
            if not pending:
                self.messages.append(assistant)
                return

            # ── 第 3 步：把「助手说了什么 + 要调什么」原样记入上下文，再执行工具 ──
            # 原样保留（而不是只存文本）才能满足多轮消息契约，也让下一轮请求
            # 的消息前缀与这一轮严格一致 —— 这是 KV Cache 能复用的前提。
            assistant["tool_calls"] = pending
            self.messages.append(assistant)

            # ── 第 4 步：执行工具，把观察结果回填，然后回到循环顶部再问模型 ──
            for call, result in zip(pending, self._run_tools(pending)):
                yield {"type": "tool_result", "content": result}
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_name": call.get("function", {}).get("name", "unknown"),
                        "content": result,
                    }
                )

        yield {"type": "error", "content": f"达到最大步数 {self.max_steps}，提前结束"}

    # ------------------------------------------------------------------ 内部实现

    def _run_tools(self, tool_calls: list[dict[str, Any]]) -> list[str]:
        """执行一批工具调用。

        同一批里的调用可以并行：它们是模型在没看到任何结果的情况下一次性生成的，
        因此彼此独立。用 executor.map 保证返回顺序与请求顺序一致。
        若后一个工具的参数依赖前一个的结果，模型只能等下一轮 —— 那种情况不会
        出现在同一批里。
        """

        def run_one(call: dict[str, Any]) -> str:
            function = call.get("function", {})
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):  # 有的服务端把参数序列化成 JSON 字符串
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            return self.registry.call(function.get("name", ""), arguments)

        if len(tool_calls) == 1:
            return [run_one(tool_calls[0])]
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
            return list(pool.map(run_one, tool_calls))
