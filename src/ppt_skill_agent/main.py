import argparse
from pathlib import Path
import subprocess
import sys
from openai import OpenAI, BadRequestError
from pptx import Presentation
import json
from utils import printf_custom



# ---------------------------------------------------------------------------
# 路径与配置
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SKILLS_DIR = ROOT / "skills"
PAPER_PATH = ROOT / "papers" / "sample_paper.md"
OUTPUT_DIR = ROOT / "output"
MODEL = "deepseek-flash"

# ---------------------------------------------------------------------------
# 短语截断
# ---------------------------------------------------------------------------
def _short(v, n=48):
    s = str(v).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"

# ---------------------------------------------------------------------------
# 打印日志
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    # 打印日志
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# 校验：用 python-pptx 重新打开生成的文件，读回页数与每页标题，证明是有效 pptx。
# ---------------------------------------------------------------------------
def verify_pptx(path: Path) -> None:
    log("\n" + "=" * 72)
    log("【校验】用 python-pptx 重新打开生成的文件，读回页数与每页标题：")
    log("-" * 72)
    prs = Presentation(str(path))
    slides = list(prs.slides)
    log(f"文件: {path}")
    log(f"总页数: {len(slides)}")
    for i, slide in enumerate(slides, 1):
        first_text = "(空)"
        for shp in slide.shapes:
            if shp.has_text_frame and shp.text_frame.text.strip():
                first_text = shp.text_frame.text.strip().splitlines()[0]
                break
        log(f"  第 {i:>2} 页标题: {first_text}")
    log("-" * 72)
    log(f"校验通过：这是一个可被 python-pptx / PowerPoint 打开的有效 .pptx（{len(slides)} 页）。")
    log("=" * 72)
    
# ---------------------------------------------------------------------------
# 工具实现：read_skill / read_skill_file / run_skill_script
# 这些是「渐进式披露」的通道——第二、三层内容只有被调用时才进入上下文。
# ---------------------------------------------------------------------------
def tool_read_skill(catalog: dict, name: str) -> str:
    info = catalog.get(name)
    if not info:
        return f"[error] 未找到 Skill: {name}"
    content = (info["dir"] / "SKILL.md").read_text(encoding="utf-8")
    log(f"\n  >>> [渐进式披露·第二层] Agent 调用 read_skill('{name}')，"
        f"加载完整 SKILL.md（{len(content)} 字符）")
    return content

# ---------------------------------------------------------------------------
# 工具实现：read_skill_file
# ---------------------------------------------------------------------------
def tool_read_skill_file(catalog: dict, name: str, rel_path: str) -> str:
    info = catalog.get(name)
    if not info:
        return f"[error] 未找到 Skill: {name}"
    # 解析相对路径
    target = (info["dir"] / rel_path).resolve()
    # 防目录穿越：必须落在该 skill 目录内
    if not str(target).startswith(str(info["dir"].resolve())):
        return f"[error] 非法路径: {rel_path}"
    if not target.exists():
        return f"[error] 文件不存在: {rel_path}"
    content = target.read_text(encoding="utf-8")
    log(f"  >>> [渐进式披露·第三层] Agent 调用 read_skill_file('{name}', '{rel_path}')，"
        f"加载子文档（{len(content)} 字符）")
    return content
# ---------------------------------------------------------------------------
# 工具实现：run_skill_script,  捆绑脚本
# ---------------------------------------------------------------------------
def tool_run_skill_script(catalog: dict, name: str, script: str, payload: str,out_path: Path) -> str:
    info = catalog.get(name)
    if not info:
        return f"[error] 未找到 Skill: {name}"
    scripts_dir = (info["dir"] / "scripts").resolve()
    script_path = (scripts_dir / script).resolve()
    # 防目录穿越：脚本会被直接执行，必须落在该 skill 的 scripts 目录内
    if not script_path.is_relative_to(scripts_dir):
        return f"[error] 非法脚本路径: {script}"
    if not script_path.exists():
        return f"[error] 脚本不存在: {script}"

    # 动态载入捆绑脚本（它就是 Skill 的一部分）
    import importlib.util
    spec = importlib.util.spec_from_file_location("bundled_generator", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except json.JSONDecodeError as e:
        return f"[error] payload 不是合法 JSON: {e}"

    log(f"  >>> [执行捆绑脚本] run_skill_script('{name}', '{script}') "
        f"生成 {out_path.name} ...")
    try:
        result = module.build_presentation(data, str(out_path))
    except Exception as e:
        # 把脚本错误变成模型可见的「可行动」错误，让它自我修正后重试
        return f"[error] 脚本执行失败: {type(e).__name__}: {e}"
    return json.dumps(result, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 工具列表
# 这些工具将被传递给 Agent，用于调用
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": "加载指定 Skill 的完整 SKILL.md（核心流程，渐进式披露第二层）。",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Skill 名称"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill_file",
            "description": "读取某 Skill 目录内的子文档或脚本源码（细则，渐进式披露第三层）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "path": {"type": "string", "description": "相对 skill 目录的路径，如 reference.md 或 scripts/generate_pptx.py"},
                },
                "required": ["name", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_skill_script",
            "description": "执行某 Skill 捆绑的脚本以完成实际产出（如生成 pptx）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "script": {"type": "string", "description": "脚本文件名，如 generate_pptx.py"},
                    "payload": {"type": "string", "description": "传给脚本的 JSON 字符串（大纲）"},
                },
                "required": ["name", "script", "payload"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# 调度工具
# ---------------------------------------------------------------------------
def dispatch(catalog: dict,name:str,args:dict,output:Path) -> str:
    log(f"dispatch: {name}, {args}, {output}")
    required = {
        "read_skill": ["name"],
        "read_skill_file": ["name", "path"],
        "run_skill_script": ["name", "script", "payload"], 
    }
    if name not in required:
        return f"[error] 未知工具: {name}"
    # 检查缺少的参数
    missing = [key for key in required[name] if key not in args]
    if missing:
        return f"[error] 缺少参数: {name}, {missing}"

    if name == "read_skill":
        return tool_read_skill(catalog, args["name"])
    if name == "read_skill_file":
        return tool_read_skill_file(catalog, args["name"], args["path"])
        
    return tool_run_skill_script(catalog, args["name"], args["script"], args["payload"], output)


# ---------------------------------------------------------------------------
# 构建系统提示
# ---------------------------------------------------------------------------
def build_system_prompt(catalog: dict) -> str:
    """构建系统提示，包含所有 Skill 的元数据。"""
    line = [
        "你是一个能使用Agent Skills 的助手。你并不预先知道每个 skill 的详细流程",
        "只在下方看到一份「薄目录」——每个 skill 的 name 和 description。",
        "",
        "当任务需要某个 skill 时，你必须：",
        "1) 先使用 read_skill(name)，加载它的完整 Skill.md (第二层：核心流程)；",
        "2) 如需实现/样式细节,再用 read_skill_file(name, path) 来读取子文档或脚本（第三层）；",
        "3) 按 SKILL.md 的约定，用 run_skill_script 调用捆绑脚本完整任务",
        "不要在没有 read_skill 的情况查下臆测某个 Skill 的调用方式",
        "",
        "== 已安装 Skills (薄目录，仅元数据)==",
    ]

    for name, skill in catalog.items():
        line.append(f"- {name}: {skill['description']}")
    return "\n".join(line)

# ---------------------------------------------------------------------------
# 解析 SKILL.md 的 frontmatter
def parse_frontmatter(skill_md: str) -> dict:
    """从 SKILL.md 顶部的 --- YAML frontmatter --- 中解析 name / description。"""
    meta = {}
    if not skill_md.startswith("---"):
        return meta
    # 找到 --- 的结束位置
    end = skill_md.find("---", 3)
    if end == -1:
        return meta
    for line in skill_md[3:end].splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            meta[key.strip()] = value.strip()
    return meta

# ---------------------------------------------------------------------------
# 扫描 Skill 目录，生成元数据
# ---------------------------------------------------------------------------
def scan_skill_catalog() -> dict:
    """返回 {skill_name: {"description":..., "dir": Path}}，只含元数据。"""
    catalog = {}
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        meta = parse_frontmatter(skill_md.read_text(encoding='utf-8'))
        name = meta.get("name") or skill_md.parent.name
        catalog[name] = {
            "description": meta.get("description"),
            "dir": skill_md.parent,
        }
    return catalog

def chat_with_tool(client,model,messages:list[dict],no_reasoning:bool=False):
    kwargs = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0.2
    }
    if no_reasoning:
        kwargs["reasoning_effort"] = 'none'
    try:
        return client.chat.completions.create(**kwargs), no_reasoning
    except BadRequestError as e:
        if no_reasoning or "reasoning_effort" not in str(e):
            raise
        log('\n[提示] 当前端点不支持「推理 + function tools」并存，本轮起降级为 '
            'reasoning_effort="none"。')
        log("       配置 OPENROUTER_API_KEY 可改走 OpenRouter，保留模型的推理能力。")
        kwargs["reasoning_effort"] = "none"
        return client.chat.completions.create(**kwargs), True

# ---------------------------------------------------------------------------
# 运行 Agent
# ---------------------------------------------------------------------------
def run_agent(paper_path: Path, model: str, output_path: Path,max_turns:int=8) -> Path | None:
    print(f'run_agent: {paper_path}, {model}, {output_path}')
    client = OpenAI(
        api_key="sk-xxxx", 
        # 注意：deepseek-flash 不是 OpenAI 官方模型，base_url 应填
        # 该模型服务商提供的网关地址（一般以 /v1 结尾）
        base_url="https://api.deepseek.com",
        timeout=60,
        max_retries=3,
    )

    catalog = scan_skill_catalog()
    system_prompt = build_system_prompt(catalog)

    log("=" * 72)
    log("【第一层·元数据】Agent 启动时只看到这份薄 Skill 目录（system prompt）：")
    log("-" * 72)
    log(system_prompt)
    log("-" * 72)
    log(f"（薄目录约 {len(system_prompt)} 字符 / 数百 token；各 Skill 的详细流程此刻并不在上下文中）")
    log("=" * 72)

    paper = paper_path.read_text(encoding='utf-8')
    user_task = (
        "请把下面这篇论文做成一份 8-12 页的演示文稿（含标题页、目录页、问题背景、"
        "方法概述、关键结果、局限性、小结页），总页数务必落在 8-12 页。"
        "先判断该用哪个 Skill，再严格按其 SKILL.md 的页序与约束操作。\n\n"
        "=== 论文全文 ===\n" + paper
    )
    messages: list[dict] = [
        {
            "role":"system",
            "content":system_prompt
        },
        {
            "role":"user",
            "content":user_task
        }
    ]
    log("\n【任务下发】要求 Agent 从论文生成演示文稿。观察它如何按需渐进式披露：\n")

    final_result = None
    no_reasoning = False

    for turn in range(1, max_turns + 1):
        log(f"\n──── 第 {turn}/{max_turns} 轮：把当前上下文发给模型决策 ────")
        response, no_reasoning = chat_with_tool(client,model,messages,no_reasoning)
        msg = response.choices[0].message
        # 把模型的回复记入上下文，供下一轮使用
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            log(f"\n【Agent 第 {turn} 轮·结束语】\n{msg.content}")
            break
        for tool_call in msg.tool_calls:  
            log(f"【工具调用】{tool_call.function.name}")
            fn = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments or "{}")  
            except json.JSONDecodeError:
                log(f"  （参数不是合法 JSON，按空参数执行）")
                fn_args = {}
            log(f"\n[Agent 第 {turn} 轮] 调用工具 -> {fn}({', '.join(f'{k}={_short(v)}' for k, v in fn_args.items())})")
            result  = dispatch(catalog, fn, fn_args, output_path)
            if fn == "run_skill_script" and not result.startswith("[error]"):
                final_result = json.loads(result)
                log(f"  >>> 生成结果：{result}")
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
    print('='*50)
    print('messages:\n')
    (OUTPUT_DIR / "messages.json").write_text(json.dumps(messages, ensure_ascii=False, indent=2))
    print('='*50)
    # 循环结束（模型自然收尾或达到最大轮数）后，返回生成的演示文稿路径
    if final_result:
        return Path(final_result["path"])
    return None
        
# ---------------------------------------------------------------------------
# 解析命令行参数
# ---------------------------------------------------------------------------
def parse_args():       
    p = argparse.ArgumentParser(description="实验 2-6：用 Agent Skills 的「渐进式披露」从论文生成演示文稿。"
                    "Agent 启动只看到薄 Skill 目录，按需逐层加载 pptx Skill 的流程与脚本，"
                    "最后用 python-pptx 生成并校验 output/presentation.pptx。",
                    formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--paper', help='输入论文/大纲（markdown）路径，默认 papers/sample_paper.md。', default=str(PAPER_PATH)) 
    p.add_argument('--output','-o', help='输出演示文稿（pptx）路径，默认 output/presentation.pptx。', default=str(OUTPUT_DIR / "presentation.pptx"))
    p.add_argument("--model",default=MODEL,help='使用的模型，默认从环境变量 OPENAI_MODEL 获取。')  
    p.add_argument('--max-turns',type=int,help="agentic loop 最大轮数，默认 8。",default=8)
    # p.add_argument("--offline", action="store_true", help="是否离线模式，默认在线。")
    return p.parse_args()

# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    paper_path = Path(args.paper)
    output_path = Path(args.output)
    # 创建输出目录, 如果不存在则创建
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 运行 agent
    pptx_path = run_agent(paper_path, args.model, output_path, args.max_turns)

    if pptx_path and pptx_path.exists():
        verify_pptx(pptx_path)
    else:
        log("\n未生成 pptx。请检查上面的日志。")
        sys.exit(2)


if __name__ == "__main__":
    main()
