
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import logging
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import os
import platform
import sys
import json
import  traceback
from openai import OpenAI 
import subprocess

try:
    from dotenv import load_dotenv
    # 必须显式指向 .env 文件：传目录时 python-dotenv 1.2.x 不会加载任何变量
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 状态枚举
class TodoStatus(Enum):
    """Status of a todo item"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

# 系统配置参数


@dataclass
class SystemHintConfig:
    """Configuration for system hints"""
    enable_timestamps: bool = True
    enable_tool_counter: bool = True
    enable_todo_list: bool = True
    enable_detailed_errors: bool = True
    enable_system_state: bool = True  # Current dir, shell, etc.
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    simulate_time_delay: bool = False  # For demo purposes
    save_trajectory: bool = True  # Save conversation history to file
    trajectory_file: str = "trajectory.json"  # File to save trajectory to


@dataclass
class TodoItem:
    id: int
    content: str
    status: TodoStatus = TodoStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, any]
    result: Optional[any] = None
    error: Optional[str] = None
    call_number: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: Optional[int] = None


class SystemHintAgent:
    def __init__(self, api_key: str=os.getenv("API_KEY"), model: str=os.getenv("MODEL_NAME"), config: SystemHintConfig = None, verbose: bool = False):
        self.config: SystemHintConfig = config
        self.todo_list = []
        self.tool_calls = []

        self.verbose = verbose
        self.tool_call_counts: Dict[str, int] = {}
        self.tool_calls: List[ToolCall] = []
        self.next_todo_id = 1

        # 会话历史
        self.conversation_history = []
        self.simulated_time = datetime.now()

        # 当前项目目录
        self.current_directory = os.getcwd()

        self.last_llm_messages = None

        # 初始化系统提示词
        self.__init_system_prompt();
        
        self.model = model
        self.api_key = api_key
        self.base_url = os.getenv("BASE_URL")

        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def __init_system_prompt(self):
        """Initialize the system prompt."""
        system_content = """You are an intelligent assistant with access to various tools for file operations, code execution, and system commands.
        Your task is to complete the given objectives efficiently using the available tools. Think step by step and use tools as needed.

        ## TODO List Management Rules:
        - For any complex task with 3+ distinct steps, immediately create a TODO list using `rewrite_todo_list`
        - Break down the user's request into specific, actionable TODO items
        - Update TODO items to 'in_progress' when starting work on them using `update_todo_status`
        - Mark items as 'completed' immediately after finishing them
        - Only have ONE item 'in_progress' at a time
        - If you encounter errors or need to change approach, update relevant TODOs to 'cancelled' and add new ones
        - Use the TODO list as your primary planning and tracking mechanism
        - Reference TODO items by their ID when discussing progress

        ## Key Behaviors:
        1. ALWAYS start complex tasks by creating a TODO list
        2. Pay attention to timestamps to understand the timeline of events
        3. Notice tool call numbers (e.g., "Tool call #3") to avoid repetitive loops - if you see high numbers, change strategy
        4. Learn from detailed error messages to fix issues and adapt your approach
        5. Be aware of your current directory and system environment shown in system state
        6. When exploring projects, systematically read key files (README, main.py, agent.py) to understand structure

        ## Error Handling:
        - Read error messages carefully - they contain specific information about what went wrong
        - Use the suggestions provided in error messages to fix issues
        - If a tool fails multiple times (check the call number), try a different approach
        - Common fixes: check file paths, verify current directory, ensure proper permissions

        Important: When you have completed all tasks, clearly state "FINAL ANSWER:" followed by a comprehensive summary of what was accomplished."""
        self.conversation_history.append(
            {"role": "system", "content": system_content}
        )

    # 获取时间戳
    def _get_timestamp(self) -> str:
        if self.config.simulate_time_delay:
            # 计算时间延迟
            return self.simulated_time.strftime(self.config.timestamp_format)
        # 返回当前时间戳
        return datetime.now().strftime(self.config.timestamp_format)

    # 增加模拟时间
    def _advance_simulated_time(self, hours: int = 1, minutes: int = 0, seconds: int = 30):
        if self.config.simulate_time_delay:
            # 增加模拟时间
            self.simulated_time += timedelta(hours=hours, minutes=minutes, seconds=seconds)

    # 保存轨迹
    def _save_trajectory(self,iteration:int,final_answer:Optional[str]=None):
        if not self.config.save_trajectory:
            return
        trajectory_data = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "model": self.model,
            "conversation_history": self.conversation_history,
            "last_llm_messages": self.last_llm_messages,
            "tool_calls": [
                {
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "result": call.result,
                    "error": call.error,
                    "call_number": call.call_number,
                    "timestamp": call.timestamp,
                    "duration_ms": call.duration_ms
                }
                for call in self.tool_calls
            ],
            "todo_list": [
                {
                    "id": item.id,
                    "content": item.content,
                    "status": item.status.value,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at
                }
                for item in self.todo_list
            ],
            "current_directory": self.current_directory,
            "final_answer": final_answer,
            "config": {
                "enable_timestamps": self.config.enable_timestamps,
                "enable_tool_counter": self.config.enable_tool_counter,
                "enable_todo_list": self.config.enable_todo_list,
                "enable_detailed_errors": self.config.enable_detailed_errors,
                "enable_system_state": self.config.enable_system_state,
                "timestamp_format": self.config.timestamp_format,
                "simulate_time_delay": self.config.simulate_time_delay
            }
        }

        try:
            with open(self.config.trajectory_file, "w", encoding='utf-8') as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
            if self.verbose:
                logger.info(f'Trajectory saved to {self.config.trajectory_file} (iteration {iteration})')
        except Exception as e:
            logger.error(f"Error saving trajectory: {e}")

    # 获取系统状态
    def _get_system_state(self, enable_system_state: bool=False)->str:
        """获取当前系统状态信息"""
        if not self.config.enable_system_state and enable_system_state:
            return ''
        
        system  = platform.system();
        
        if system == "Windows":
            shell_type = '"Windows Command Prompt or PowerShell'
        elif system == 'Darwin':
            shell_type = 'macOS Terminal (zsh/bash)'
        else :
            shell_type = f"Linux Shell ({os.environ.get('SHELL', 'bash')})"

        state_info = [
            f"Current Time: {self._get_timestamp()}",
            f"Current Directory: {self.current_directory}",
            f"System Environment: {system}",
            f"Shell Environment: {shell_type}",
            f"Python Version: {platform.python_version()}"
        ]

        return '\n'.join(state_info)
    
    # 格式化待办事项列表
    def _format_todo_list(self) -> str:
        if not self.todo_list:
            return "不存在 Todo List"
        lines = ["TODO List:"]

        #   格式化待办事项列表
        for item in self.todo_list:
            status_symbol = {
                TodoStatus.PENDING: "⏳",
                TodoStatus.IN_PROGRESS: "🔄",
                TodoStatus.COMPLETED: "✅",
                TodoStatus.CANCELLED: "❌"
            }.get(item.status, "❓")

            lines.append(f" [{item.id}] {status_symbol} {item.content} （{item.status.value}）")
        return '\n'.join(lines)

    # 获取系统状态提示
    def _get_system_hint(self) -> Optional[str]:
        # 在没有开启系统状态或待办事项功能时，不生成系统提示
        if not any([self.config.enable_system_state,self.config.enable_todo_list]):
            return None
        hint_parts = []
        
        # 添加系统状态
        if self.config.enable_system_state:
            hint_parts.append("==== SYSTEM STATE====")
            hint_parts.append(self._get_system_state())
            hint_parts.append('')

        # 添加待办事项列表
        if self.config.enable_todo_list and self.todo_list:
            hint_parts.append("==== TODO LIST====")
            hint_parts.append(self._format_todo_list())
            hint_parts.append('')

        if hint_parts:
            return '\n'.join(hint_parts)
        else:
            return None
    
    # 获取工具描述
    def _get_tools_description(self) -> List[Dict[str, Any]]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a text file. Returns error for binary files. Supports partial reading for large files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read (absolute or relative to current directory)"
                            },
                            "begin_line": {
                                "type": "integer",
                                "description": "Optional: Line number to start reading from (1-based indexing). E.g., begin_line=10 starts from line 10."
                            },
                            "number_lines": {
                                "type": "integer",
                                "description": "Optional: Number of lines to read from begin_line. E.g., number_lines=50 reads 50 lines."
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file (creates or overwrites)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "Execute Python code in a restricted environment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute a shell command in the current directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute"
                            },
                            "working_dir": {
                                "type": "string",
                                "description": "Optional working directory for the command (defaults to current directory)"
                            }
                        },
                        "required": ["command"]
                    }
                }
            }
        ]

        if self.config.enable_todo_list:
            tools.extend([
                {
                    "type":"function",
                    "function":{
                        "name":"rewrite_todo_list",
                        "description":"Rewrite the TODO list with new pending items (keeps completed/cancelled items)",
                        "parameters":{
                            "type":"object",
                            "properties":{
                                "items":{
                                    "type":"array",
                                    "items": {
                                        "type": "string"
                                    },
                                    "description":"The new items of the todo list"
                                }
                            },
                            "required":["items"]
                        }
                    },
                },
                {
                    "type":"function",
                    "function":{
                        "name":"update_todo_status",
                        "description":"Update the status of existing TODO items",
                        "parameters":{
                            "type":"object",
                            "properties":{
                                "updates":{
                                    "type":"array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {
                                                "type": "integer",
                                                "description": "TODO item ID"
                                            },
                                            "status": {
                                                "type": "string",
                                                "description": "New status for the item",
                                                "enum": ["pending", "in_progress", "completed", "cancelled"]
                                            }
                                        },
                                        "required": ["id", "status"]
                                    },
                                    "description":"List of TODO items to update with their new status"
                                }
                            },
                            "required":["updates"]
                        }
                    },
                }
            ])
        return tools

    # 获取错误提示
    def _get_error_suggestions(self, exception: Exception, tool_name: str) -> str:
        """Get suggestions for fixing common errors"""
        error_str = str(exception).lower()
        exception_type = type(exception).__name__
        
        suggestions = []
        
        if "permission" in error_str or exception_type == "PermissionError":
            suggestions.append("Check file/directory permissions")
            suggestions.append("Try using a different directory or running with appropriate permissions")
        elif "not found" in error_str or "no such file" in error_str or exception_type == "FileNotFoundError":
            suggestions.append("Verify the file/directory path exists")
            suggestions.append("Check the current working directory")
            suggestions.append("Use absolute paths or create the file/directory first")
        elif "syntax" in error_str or exception_type == "SyntaxError":
            suggestions.append("Check the code syntax")
            suggestions.append("Ensure proper indentation and valid Python syntax")
        elif "timeout" in error_str:
            suggestions.append("The operation took too long")
            suggestions.append("Try with simpler input or break into smaller steps")
        elif "import" in error_str or exception_type == "ImportError":
            suggestions.append("Required module not available in restricted environment")
            suggestions.append("Use only built-in Python modules")
        
        return " | ".join(suggestions) if suggestions else ""
    # Tool implementations
    # 读文件工具
    def _tool_read_file(self, file_path: str, begin_line: Optional[int] = None, number_lines: Optional[int] = None) -> Dict[str, Any]:
        """Read file contents with optional line-based reading"""
        try:
            # 相对路径解析为绝对路径
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.current_directory, file_path)
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件没找到: {file_path}")
            
            # 检查是否为二进制文件
            try:
                with open(file_path, 'rb') as f:
                    # 读取前 1024 字节以检查二进制内容
                    chunk = f.read(1024)
                    # 检查是否存在空字节（在二进制文件中很常见）
                    if b'\x00' in chunk:
                        return {
                            "success": False,
                            "error": "Cannot read binary file. This tool only supports text files.",
                            "file_path": file_path,
                            "is_binary": True
                        }
                    # 也检查是否为有效 UTF-8。以增量方式解码
                    # 最后=False，因此一个多字节字符被 1024 字节读取边界分成两半
                    # (每个 CJK 字符是 3 个字节，因此这很常见)。    
                    try:
                        codecs.getincrementaldecoder('utf-8')().decode(chunk, False)
                    except UnicodeDecodeError:
                        return {
                            "success": False,
                            "error": "File is not a valid text file (encoding error).",
                            "file_path": file_path,
                            "is_binary": True
                        }
            except Exception:
                # If we can't read it as binary, probably permission issue
                raise
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                if begin_line is not None or number_lines is not None:
                    # Line-based reading
                    all_lines = f.readlines()
                    total_lines = len(all_lines)
                    
                    # Calculate line range
                    start_line = (begin_line - 1) if begin_line is not None else 0
                    if start_line < 0:
                        start_line = 0
                    if total_lines == 0 and start_line == 0:
                        return {
                            "success": True,
                            "file_path": file_path,
                            "content": "",
                            "size_bytes": 0,
                            "total_lines": 0,
                            "begin_line": 1,
                            "end_line": 0,
                            "lines_read": 0,
                            "partial_read": True
                        }
                    if start_line >= total_lines:
                        return {
                            "success": False,
                            "error": f"begin_line {begin_line} is beyond file length ({total_lines} lines)",
                            "file_path": file_path,
                            "total_lines": total_lines
                        }
                    
                    if number_lines is not None:
                        end_line = min(start_line + number_lines, total_lines)
                    else:
                        end_line = total_lines
                    
                    # Get the requested lines
                    selected_lines = all_lines[start_line:end_line]
                    content = ''.join(selected_lines)
                    
                    # Get file info
                    stat = os.stat(file_path)
                    
                    return {
                        "success": True,
                        "file_path": file_path,
                        "content": content,
                        "size_bytes": stat.st_size,
                        "total_lines": total_lines,
                        "begin_line": start_line + 1,  # Convert back to 1-based
                        "end_line": end_line,
                        "lines_read": len(selected_lines),
                        "partial_read": True
                    }
                else:
                    # Full file reading
                    content = f.read()
                    
                    # Get file info
                    stat = os.stat(file_path)
                    
                    return {
                        "success": True,
                        "file_path": file_path,
                        "content": content,
                        "size_bytes": stat.st_size,
                        "lines": len(content.splitlines()),
                        "partial_read": False
                    }
        except Exception:
            raise
    # 写文件工具
    def _tool_write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Write content to file"""
        try:
            # 相对路径解析为绝对路径
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.current_directory, file_path)
            
            # 创建目录如果需要
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "file_path": file_path,
                "bytes_written": len(content.encode('utf-8')),
                "lines_written": len(content.splitlines())
            }
        except Exception:
            raise
    
    # 代码解释器工具
    def _tool_code_interpreter(self, code: str) -> Dict[str, Any]:
        """Execute Python code in restricted environment"""
        try:
            # Capture output
            import io
            import contextlib
            
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()
            
            # Run with an explicit namespace: with bare exec(code), top-level
            # assignments land in this method's locals while functions defined
            # in the snippet resolve free variables via module globals, so
            # "x = 5; def f(): return x; f()" raises NameError.
            exec_ns = {}
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
                exec(code, exec_ns)
            
            # Get output
            stdout = output_buffer.getvalue()
            stderr = error_buffer.getvalue()
            
            return {
                "success": True,
                "stdout": stdout,
                "stderr": stderr,
            }
        except Exception:
            raise
    # 执行命令工具
    def _tool_execute_command(self, command: str, working_dir: Optional[str] = None) -> Dict[str, Any]:
        """Execute shell command"""
        try:
            # 使用当前目录如果未指定
            if working_dir is None:
                working_dir = self.current_directory
            elif not os.path.isabs(working_dir):
                # 相对路径解析为绝对路径
                working_dir = os.path.join(self.current_directory, working_dir)
            
            # 更新当前目录如果命令是纯粹的 'cd'
            # Compound commands like `cd proj && make` must fall through to
            # the subprocess below (which runs with cwd=working_dir) —
            # intercepting them here would treat "proj && make" as the
            # directory name and fail with "Directory not found".
            stripped = command.strip()
            # 去除命令前后的空格
            if stripped.startswith('cd ') and not any(t in stripped for t in ('&&', ';', '|')):
                new_dir = stripped[3:].strip()
                if not os.path.isabs(new_dir):
                    new_dir = os.path.join(self.current_directory, new_dir)
                
                if os.path.isdir(new_dir):
                    self.current_directory = os.path.abspath(new_dir)
                    return {
                        "success": True,
                        "command": command,
                        "output": f"Changed directory to: {self.current_directory}",
                        "return_code": 0
                    }
                else:
                    raise FileNotFoundError(f"Directory not found: {new_dir}")
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=working_dir,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "command": command,
                "output": result.stdout,
                "error": result.stderr if result.stderr else None,
                "return_code": result.returncode,
                "working_dir": working_dir
            }
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out after 30 seconds: {command}")
        except Exception:
            raise
    
    # 重写 TODO 列表工具
    def _tool_rewrite_todo_list(self, items: List[str]) -> Dict[str, Any]:
        """Rewrite TODO list with new pending items"""
        # 保留已完成和已取消的项目
        kept_items = [
            item for item in self.todo_list
            if item.status in [TodoStatus.COMPLETED, TodoStatus.CANCELLED]
        ]
        
        # 创建新待办事项
        new_items = []
        for content in items:
            new_items.append(TodoItem(
                id=self.next_todo_id,
                content=content,
                status=TodoStatus.PENDING
            ))
            self.next_todo_id += 1
        
        # 更新 TODO 列表
        self.todo_list = kept_items + new_items
        
        return {
            "success": True,
            "kept_items": len(kept_items),
            "new_items": len(new_items),
            "total_items": len(self.todo_list)
        }
    
    # 更新 TODO 状态工具
    def _tool_update_todo_status(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update status of TODO items"""
        updated_count = 0
        
        for update in updates:
            item_id = update["id"]
            new_status = TodoStatus(update["status"])
            
            for item in self.todo_list:
                if item.id == item_id:
                    item.status = new_status
                    item.updated_at = datetime.now().isoformat()
                    updated_count += 1
                    break
        
        return {
            "success": True,
            "updated_items": updated_count,
            "total_items": len(self.todo_list)
        }
    
    # 获取详细错误信息
    def _get_detailed_error(self, exception: Exception, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Get detailed error information for debugging"""
        error_parts = [
            f"Tool '{tool_name}' failed with {type(exception).__name__}: {str(exception)}",
            f"Arguments: {json.dumps(arguments, indent=2)}",
        ]
        
        # Add traceback for debugging
        if self.verbose:
            tb = traceback.format_exc()
            error_parts.append(f"Traceback:\n{tb}")
        
        # Add suggestions based on error type
        suggestions = self._get_error_suggestions(exception, tool_name)
        if suggestions:
            error_parts.append(f"Suggestions: {suggestions}")
        
        return "\n".join(error_parts)
    
    # 执行工具调用
    def _execute_tool(self, tool_call: str, arguments: Dict[str, Any]) -> Tuple[any,Optional[str],Optional[int]]:
        """执行工具调用"""
        start_time = datetime.now()
        try:
            if tool_call == "read_file":
                result = self._tool_read_file(**arguments)
            elif tool_call == "write_file":
                result = self._tool_write_file(**arguments)
            elif tool_call == "code_interpreter":
                result = self._tool_code_interpreter(**arguments)
            elif tool_call == "execute_command":
                result = self._tool_execute_command(**arguments)
            elif tool_call == "rewrite_todo_list":
                result = self._tool_rewrite_todo_list(**arguments)
            elif tool_call == "update_todo_status":
                result = self._tool_update_todo_status(**arguments)
            else:
                error = f"Unknown tool: {tool_call}"
                return {"error": error},error, None
            duration_ms = int((datetime.now() - start_time).total_seconds()*1000)
            return result, None, duration_ms
        except Exception as exc:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            error_detail = self._get_detailed_error(exc, tool_call, arguments)
            if self.config.enable_detailed_errors:
                return {"error": error_detail}, error_detail, duration_ms
            else:
                return {"error": str(exc)}, str(exc), duration_ms

    # 执行任务
    def execute_task(self, task: str, max_iterations: int = 20) -> Dict[str, Any]:
        """执行一个任务，使用系统可用工具完整地完成任务"""
        if self.config.enable_timestamps:
            timestamp_prefix = f"[{self._get_timestamp()}]"
            task = timestamp_prefix + task

        # 添加用户消息
        self.conversation_history.append(
            {"role": "user", "content": task}
        )

        iteration = 0 
        final_answer = None
        
        # 执行任务
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{max_iterations}")

            # 增加模拟时间
            self._advance_simulated_time(seconds=5)
            
            # 保存轨迹
            self._save_trajectory(iteration);

            try:
                # 为模型准备上下文信息，添加 system hint 作为最新的用户信息
                messages_to_send = self.conversation_history.copy()
                system_hint = self._get_system_hint();
                print(f'$$$$$$$$$$$ System Hint $$$$$$$$$$$: 第 {iteration} 次迭代: {system_hint}')
                
                # 添加系统提示
                if system_hint:
                    messages_to_send.append({"role": "user", "content": system_hint})

                self.last_llm_messages = messages_to_send
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages_to_send,
                    tools=self._get_tools_description(),
                    # tools_choice='auto',
                    temperature=0.4,
                    max_tokens=8192,
                )
                message = response.choices[0].message
                has_tools_call = bool(getattr(message, "tool_calls", None))

                # 无工具调用情况处理
                if not has_tools_call:
                    # 添加模型响应
                    self.conversation_history.append(message.model_dump())
                    content = (message.content or '').strip();
                    if content:
                        final_answer = (content.split("FINAL ANSWER:", 1)[1].strip()
                                        if "FINAL ANSWER:" in content else content)
                        logger.info(f"无工具调用返回结果; final answer: {final_answer[:100]}...")
                    else:
                        logger.info("无工具调用返回结果为空")
                    
                    # 保存轨迹
                    self._save_trajectory(iteration, final_answer)
                    break

                # 有工具调用情况处
                if has_tools_call:
                    self.conversation_history.append(message.model_dump())
                    
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = tool_call.function.arguments or '{}'

                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError as exc:
                            err = (
                                f"Invalid tool arguments (not valid JSON): {exc}. "
                                f"Raw arguments: {tool_args[:500]}"
                            )
                            logger.warning(f"  ❌ {err}")

                            self.tool_calls.append(ToolCall(
                                tool_name=tool_name,
                                arguments={},
                                error=err,
                            ))

                            # 添加工具调用信息
                            self.conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": err
                            })
                            continue
                        
                        if self.config.enable_tool_counter:
                            self.tool_call_counts[tool_name] = self.tool_call_counts.get(tool_name, 0) + 1
                            call_number = self.tool_call_counts[tool_name]
                        else:
                            call_number = 1
                        
                        logger.info(f"执行 tool: {tool_name} (调用数量 #{call_number})")
                        
                        tool_args_str = json.dumps(tool_args, indent=4)
                        if len(tool_args_str) > 200:
                            logger.info(f"  📥 Args: {tool_args_str[:200]}...")
                        else:
                            logger.info(f"  📥 Args: {tool_args_str}")
                        
                        # 调用工具
                        result,error,duration_ms = self._execute_tool(tool_name, tool_args)

                        if error:   
                            error_preview = str(error).replace("\n", " ")[:150]
                            logger.info(f"  ❌ Error: {error_preview}")
                        else:
                            logger.info(f"  ✅ Tool {tool_name} executed successfully")
                            if isinstance(result, dict):
                                if result.get('success'):
                                    # Show key information for successful operations
                                    if 'output' in result and result['output']:
                                        output_preview = str(result['output']).replace('\n', ' ')[:100]
                                        logger.info(f"  ✅ Success: {output_preview}...")
                                    elif 'content' in result:
                                        # Handle read_file results
                                        if result.get('partial_read'):
                                            logger.info(
                                                    f"  ✅ Success: Read lines {result.get('begin_line', 1)}-{result.get('end_line', 0)} "
                                                    f"({result.get('lines_read', 0)} lines) from {result.get('total_lines', 0)} total")
                                        else:
                                            logger.info(f"  ✅ Success: Read {result.get('lines', 0)} lines, {result.get('size_bytes', 0)} bytes")
                                    elif 'file_path' in result:
                                        logger.info(f"  ✅ Success: File operation on {result['file_path']}")
                                    else:
                                        logger.info("  ✅ Success: Operation completed")
                                elif result.get('success') is False:
                                    # Handle explicit failures (like binary file detection)
                                    if result.get('is_binary'):
                                        logger.info(f"  ⚠️ Binary file detected: {result.get('file_path', 'unknown')}")
                                    else:
                                        err_msg = str(result.get('error') or 'Unknown error')
                                        logger.info(f"  ⚠️ Failed: {err_msg[:100]}")
                                else:
                                    logger.info("  ✅ Success: Operation completed")
                            else:
                                result_preview = str(result).replace('\n', ' ')[:150]
                                logger.info(f"  ✅ Result: {result_preview}")

                        # 添加工具调用记录
                        tool_call_record = ToolCall(
                            tool_name=tool_name,
                            arguments=tool_args,
                            result=result if not error else None,
                            error=error,
                            duration_ms=duration_ms,
                            call_number=call_number,
                        )

                        self.tool_calls.append(tool_call_record)

                        # 序列化成字符串作为 tool 消息内容（json.dump 需要文件对象，这里是 dumps 的用法）
                        tool_exec_content = json.dumps(result, ensure_ascii=False, default=str)

                        metadata_parts = []

                        if self.config.enable_timestamps:
                            metadata_parts.append(f"Timestamp: {self._get_timestamp()}")

                        if self.config.enable_tool_counter:
                            metadata_parts.append(f"[Tool call #{call_number} for '{tool_name}']")

                        if metadata_parts:
                            tool_exec_content = " ".join(metadata_parts) + "\n" + tool_exec_content
                        
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_exec_content
                        })
                    
                    if message.content and "FINAL ANSWER:" in message.content:
                        final_answer = message.content.split("FINAL ANSWER:", 1)[1].strip()
                        logger.info(f"Final answer found alongside tool calls: {final_answer[:100]}...")
                        self._save_trajectory(iteration, final_answer)
                        break
                
            except Exception as e:
                logger.error(f"Error during task execution: {str(e)}")
                # 保存会话轨迹即使错误
                self._save_trajectory(iteration)
                return {
                    "error": str(e),
                    "tool_calls": [asdict(call) for call in self.tool_calls],
                    "iterations": iteration,
                    "trajectory_file": self.config.trajectory_file if self.config.save_trajectory else None
                }
        # 保存最后一个会话的轨迹
        self._save_trajectory(iteration, final_answer)

        return {
            "final_answer": final_answer,
            "tool_calls": [asdict(call) for call in self.tool_calls],
            "todo_list": [
                {
                    "id": item.id,
                    "content": item.content,
                    "status": item.status.value
                }
                for item in self.todo_list
            ],
            "iterations": iteration,
            "success": final_answer is not None,
            "trajectory_file": self.config.trajectory_file if self.config.save_trajectory else None
        }

    def reset(self):
        """Reset the agent's state"""
        self.tool_call_counts = {}
        self.tool_calls = []
        self.todo_list = []
        self.next_todo_id = 1
        self.current_directory = os.getcwd()
        self.simulated_time = datetime.now()
        self.last_llm_messages = None
        self._init_system_prompt()
        logger.info("Agent state reset")

def main() -> None:
    # print(f'os.getcwd(): {os.getcwd()}')

    task = """分析并总结 AI Agent Practical 项目在 ai_agent_practical  和 ppt_skill_agent 目录：
        1. 导航到父目录以访问 ai_agent_practical、ppt_skill_agent 文件夹
        2. 对于 ai_agent_practical 目录：
        - 列出所有项目文件
        - 读取项目中的关键文件
        - 识别关键概念
        3. 对于 system_hint 目录：
        - 列出所有项目文件夹      
        - 理解高级功能
        - 识别关键概念
        4. 创建一个综合分析文件
        """
    agent = SystemHintAgent(config= SystemHintConfig())
    result = agent.execute_task(task)
    with open(Path(__file__).parent / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False, sort_keys=True)
if __name__ == "__main__":
    main()
