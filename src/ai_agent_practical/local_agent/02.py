import json
import ollama
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
# 模块解释：
from  concurrent.futures import ThreadPoolExecutor



GRAY = "\033[90m"  # 终端灰色，用来区分思考与正文
RESET = "\033[0m"

MODEL = "qwen3:0.6b"
MAX_STEPS = 8  # 防止无限循环

# ========================工具实现
# 获取当前时间
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
# ========================工具实现

# 工具名 → 函数。模型给的是名字，靠这张表找到要执行的代码。
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_current_temperature": get_current_temperature,
}

#   工具列表
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": (
                "查询某个城市的实时气温（摄氏度）。数据来自Open-Meteo，免费且不需要 API Key。"
                "注意：city 必须是英文或拼音地名（如 tokyo、Shanghai）"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市必须是英文名，例如：tokyo、Shanghai"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"get_current_time",
            "description": (
                "查询指定时区的当前时间"
                "市区使用 IANA 时区名称，例如：Asia/Shanghai 、Asia/Tokyo 、Europe/Berlin"
            ),
            "parameters":{
                "type":"object",
                "properties":{
                    "timezone":{
                        "type":"string",
                        "description":"IANA 时区名，例如：Asia/Shanghai；不传则按 UTC"
                    }
                },
                "required":[]
            }
        }
    }
]

def run_tools(calls: list) -> tuple[str, str]:
    """执行一批工具调用，返回 [(工具明，结果),...]。
    
    同一批的调用可以并行执行；它们是模型在没有看到任何结果的情况下一次性生成的，
    因此彼此独立。executor.map 保证返回顺序与请求一致，
    如果后一个工具的参数依赖前一个的结果，模型只能等下一轮，那种情况不会出现在同一批里。
    """

    def run_one(call:dict) -> tuple[str, str]:
        name = call['function']['name']
        arguments = call["function"]["arguments"]

        return name, TOOL_FUNCTIONS[name](**arguments)
    if len(calls)==1:
        return [run_one(calls[0])]
    # 并行执行，利用线程池
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return list(pool.map(run_one, calls))


def main()->None:
    client = ollama.Client()
    messages:list[dict] = [{"role": "user", "content": "上海现在几点？气温多少？"}]
    for step in range(1,MAX_STEPS+1):
        text_parts:list[str] = []
        pending:list = []
        print(f"\n ---- 第 {step} 轮：ReAct 上下文已累计 {len(messages)} 条消息 ----")
        for chunk in client.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            stream=True,
            think = True,
        ):
            message = chunk.get("message",{})
            if message.get("thinking"):
                print(f"{GRAY}{message.get('thinking')}{RESET}", end="",flush=True)
            if message.get("content"):
                text_parts.append(message.get("content"))
                print(message.get("content"), end="", flush=True)
            for call in message.get("tool_calls") or []:
                # 去重
                if call not in pending:
                    pending.append(call)
        
        print()

        assistant :dict = {
            "role":"assistant",
            "content": "".join(text_parts),
        }

        if not pending:
            messages.append(assistant)
            print(f"\n[最终回答] {assistant['content']}")
            print(f"[共 {step} 轮，最终上下文共 {len(messages)} 条消息]")
            return
        
        assistant["tool_calls"] = pending

        for name, result  in run_tools(pending):
            print(f"[工具 {name}] {result}")
            messages.append({
                "role":"tool",
                "tool_name": name,
                "content": result,
            })
        messages.append(assistant)
    print(f"\n 达到最大轮数 {len(messages)}，提前结束可能模型陷入了重复调用")


def main1()->None:
    # question = "上海现在几点？气温多少？"
    question = "现在东京的气温换算成华氏度是多少"
    print(f"提问：{question}\n")
    client = ollama.Client()
    messages = [{"role": "user", "content": question}]
    for step in range(1,MAX_STEPS+1):
        print(f"\n ---- 第 {step} 轮：ReAct 上下文已累计 {len(messages)} 条消息 ----")
        response = client.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            stream=False,
            think = False,
        )
        message = response["message"]
        calls = message.get("tool_calls") or []
        # 没有工具调用：模型认为信息已足够，直接回答
        if not calls:
            print("没有调用工具，直接回答：", message.get("content"))
            content = message.get("content")
            messages.append({"role":"assistant", "content": content})
            print(f"\n[最终回答] {content}")
            print(f"[共 {step} 轮，最终上下文共 {len(messages)} 条消息]")
            return
        print(f"模型请求了 {len(calls)} 个工具")
        messages.append({
            "role":"assistant",
            "content":message.get("content") or "",
            "tool_calls": calls
        })
        run_tools(calls, messages)

    print(f"\n 达到最大轮数 {len(messages)}，提前结束可能模型陷入了重复调用")

if __name__ == "__main__":
    main()