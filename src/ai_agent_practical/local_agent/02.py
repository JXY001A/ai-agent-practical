import json
import ollama
from datetime import datetime
from zoneinfo import ZoneInfo
import requests

from ai_agent_practical.local_agent.step4_react import MAX_STEPS


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

def run_tools(calls: list, messages: list[dict]) -> None:
    """执行本轮的每个工具调用，并把结果追加进 messages。"""
    for call in calls:
        name = call['function']['name'] 
        arguments = call['function']['arguments']
        print(f"[执行] {name}({arguments})")

        result = TOOL_FUNCTIONS[name](**arguments)
        print(f"[结果] {result}")
        messages.append({
            "role": "tool",
            "tool_name": name,
            "content": result
        })


def main()->None:
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
    # # # model_dump_json() 输出完整 JSON 结构（含所有字段），比直接 print 更直观
    # print(f"返回的 response：\n{response.model_dump_json(indent=2, exclude_unset=True)}\n")
    # # print(f"返回的 tool_calls：\n")

    # messages.append({"role": "assistant", "content": message.get("content"),"tool_calls": calls})

    # # if calls:
    # #     # tool_calls 是 pydantic 对象列表，先用 model_dump() 转成 dict 再序列化
    # #     print(json.dumps([c.model_dump() for c in calls], indent=4, ensure_ascii=False))
    # # else:
    # #     print("没有调用工具，直接回答：", message.get("content"))
    # for call in calls:
    #     name = call["function"]["name"]
    #     arguments = call["function"]["arguments"]
    #     print(f"[执行] {name}({arguments})")

    #     result = TOOL_FUNCTIONS[name](**arguments)
    #     print(f"[结果] {result}")
    #     messages.append({
    #         "role": "tool",
    #         "tool_name": name,
    #         "content": result
    #     })
    
    # final_res = client.chat(model=MODEL, messages=messages, stream=False, think=False)
    # print(f"\n [最终回答]: {final_res['message']['content']}")
    # print(f"\n 此刻上下文共 {len(messages)} 条消息：")
    # print(f"返回的 final_res： \n {final_res.model_dump_json(indent=2, exclude_unset=True)} \n")
    # messages.append({
    #     "role": "assistant",
    #     "content": final_res['message']['content']
    # })
    # print(f'messages： \n {json.dumps(messages, indent=2, ensure_ascii=False, default=lambda o: o.model_dump())}')
    # for index,item in enumerate(messages, 1):
    #     print(f"[消息 {index}] {item['role']:9s}: {item['content']}")

if __name__ == "__main__":
    main()
