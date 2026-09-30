"""本地 Agent 的分步练习代码（教程见 docs/tutorial-local-agent.md）。

step0_check.py        环境自检
step1_chat.py         最简对话，理解 messages 就是上下文
step2_see_tool_calls  把工具介绍给模型，只看它返回什么
step3_execute.py      执行工具并把结果回填
step4_react.py        包成 ReAct 循环
step5_stream.py       流式输出 + 并行工具

step3 里的 TOOLS / TOOL_FUNCTIONS 被 step4、step5 复用。
第 6 步会把它们整理成正式模块（见 _reference_solution/）。
"""
