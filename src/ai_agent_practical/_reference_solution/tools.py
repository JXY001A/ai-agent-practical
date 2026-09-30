"""工具：给模型看的「说明书」+ 真正执行的 Python 函数。

关键认知：模型看不到下面这些函数的实现，它只看到 name / description /
parameters(JSON Schema)。所以 description 写得越清楚具体，小模型选工具
选得越准 —— 这正是「上下文决定 Agent 能力上限」的直接体现。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests


@dataclass
class Tool:
    """一个工具的三要素：模型通过前三个字段决定「调不调、怎么调」。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]


class ToolRegistry:
    """工具注册表：负责把工具转成模型能读的 schema，并按名字执行。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        """转成 Ollama / OpenAI 通用的 tools 字段格式。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """执行工具，把结果转成字符串回填给模型。

        工具自己报错时不要把异常抛出去 —— 把错误信息当成观察结果还给模型，
        模型有机会自己改参数重试。
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"错误：不存在名为 {name} 的工具"
        try:
            return tool.func(**arguments)
        except Exception as exc:  # noqa: BLE001 - 工具报错也要回给模型
            return f"工具执行失败：{exc}"


def get_current_time(timezone: str = "UTC") -> str:
    """查询指定时区的当前时间。只依赖标准库。"""
    now = datetime.now(ZoneInfo(timezone))
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


def get_current_temperature(city: str) -> str:
    """查询城市实时气温。用 Open-Meteo，免费且不需要 API Key。

    两步：地名 → 经纬度（geocoding）→ 当前气温（forecast）。
    这种「后一步依赖前一步结果」的情形，模型只能在下一轮再发起调用。
    """
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    ).json()
    if not geo.get("results"):
        # 报错信息也是「观察结果」，会进入上下文影响模型下一步动作。
        # 所以这里不只报告失败，还明确告诉模型该怎么改参数重试 ——
        # 这正是 ReAct 循环里「工具结果反过来指导行动」的体现。
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


def build_default_registry() -> ToolRegistry:
    """注册骨架默认带的两个工具。想看模型如何选择工具，从这两个的描述改起。"""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="get_current_time",
            description=(
                "查询指定时区的当前日期和时间。"
                "时区使用 IANA 名称，例如 Asia/Shanghai、America/New_York、Europe/London。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA 时区名，例如 Asia/Shanghai；不传则按 UTC",
                    }
                },
                "required": [],
            },
            func=get_current_time,
        )
    )
    registry.register(
        Tool(
            name="get_current_temperature",
            description=(
                "查询某个城市的实时气温（摄氏度）。数据来自 Open-Meteo，无需 API Key。"
                "注意：city 必须是英文或拼音地名（如 Tokyo、Shanghai），不支持中文名。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市英文名，例如 Tokyo、Shanghai、Vancouver",
                    }
                },
                "required": ["city"],
            },
            func=get_current_temperature,
        )
    )
    return registry
