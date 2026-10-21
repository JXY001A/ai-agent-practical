from dataclasses import dataclass, field
import json
import logging
from typing import List, Dict, Any, Optional

from config import Config, MemoryMode
from user_memory.agent import UserMemoryConfig, UserMemoryAgent
from user_memory.conversation_history import ConversationHistory,ConversationTurn
import threading
from datetime import datetime



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MemoryUpdate:
    """表示每个记忆更新决策"""
    action: str  # 'add', 'update', 'delete', 'none' 表示记忆更新动作
    memory_id: Optional[str] = None
    content: Optional[str] = None
    reason: Optional[str] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class MemoryProcessorConfig:
    """后台记忆进程的配置"""
    # 在多少轮对话后开始处理记忆
    conversation_interval: int = 1  # Process after N conversation rounds (default: every round)
    # 最小轮数
    min_conversation_turns: int = 1  # Minimum turns before processing
    # 最新的窗口大小
    context_window: int = 10  # Number of recent turns to analyze
    # 是否启动自动处理
    enable_auto_processing: bool = True
    temperature: float = 0.3  # Lower temperature for analysis
    # 输出记忆详情
    output_operations: bool = True  # Output detailed memory operations

class BackgroundMemoryProcessor:
    """分析对话并更新记忆的后台处理器,与主对话流分开运行"""
    def  __init__(  self,
            user_id:str,
            api_key:Optional[str] = None,
            model:Optional[str]= None,
            config:Optional[MemoryProcessorConfig] = None,
            mode:Optional[MemoryMode] = None,) -> None:
        
        self.user_id = user_id
        self.api_key = api_key
        self.model = model
        self.config = config or MemoryProcessorConfig()

        agent_config = UserMemoryConfig()
        resolved_mode = mode or agent_config.memory_mode

        # 后台分析 Agent 必须与外部使用同一种记忆模式，否则两种 manager 会互相覆盖同一份记忆文件
        self.analysis_agent = UserMemoryAgent(
            user_id=self.user_id,
            config=UserMemoryConfig(memory_mode=resolved_mode),
        )
        # 与后台 Agent 共用同一个管理器，避免两份内存副本状态不一致
        self.memory_manager = self.analysis_agent.memory_manager
        self.conversation_history = ConversationHistory(user_id=self.user_id)


        # Background processing state
        self.processing_thread = None
        self.stop_processing = False
        self.last_processed_timestamp = None 
        self.processing_lock = threading.Lock()
        self.conversation_count = 0  # Track conversation rounds
        self.last_processed_count = 0  # Track last processed conversation count
        self.processed_turn_ids = set()  # Track which turns have been processed
    
    def analysis_conversation(self, conversation_str: List[Dict[str, Any]])->List[MemoryUpdate]:
        """分析对话并生成记忆更新"""
        if len(conversation_str) < self.config.min_conversation_turns * 2:
            return []

        conversation_str = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_str
        ])

        task = f"""分析最近的对话并更新我的记忆。提取重要事实、偏好、或者应该被记住的信息。最近的会话：
                \n{conversation_str}
                
                请阅读这些对话并且：
                1. 添加任何新的的重要的信息作为记忆
                2. 更新存在的记忆如果存在新的或者需要改变的
                3. 删除任何不再准确的记忆

                专注于提取对未来对话有帮助的信息。
            """
        try:
            result = self.analysis_agent.execute_task(task)
            logger.info(f"记忆更新任务完成: {result.get('success', False)}")
            return []
        except Exception as e:
            logger.error(f"Error during memory analysis: {e}")
            return []
    def increment_conversation_count(self):
        self.conversation_count += 1
    def process_recent_conversations(self)-> Dict[str, Any]:
        with self.processing_lock:
            self.last_processed_count = self.conversation_count
            self.last_processed_timestamp = datetime.now()
            self.conversation_history.load_history()
            
            # 获取最近 context_window 轮对话历史
            recent_turns = self.conversation_history.get_recent_turns(self.config.context_window)

            if not recent_turns:
                return {
                    "message": "近期没有会话历史需要被处理",
                    "operations": [],
                    "summary":{"add":0, "update":0, "delete":0}
                }

            unprocessed_turns: List[ConversationTurn] = []
            for turn in recent_turns:
                turn_id = f"{turn.session_id}_{turn.turn_number}_{turn.timestamp}"
                if turn_id not in self.processed_turn_ids:
                    unprocessed_turns.append(turn)
                    self.processed_turn_ids.add(turn_id)
            if not unprocessed_turns:
                return {
                    "message": "近期没有会话历史需要被处理",
                    "operations": [],
                    "summary":{"add":0, "update":0, "delete":0}
                }
            conversation_context = []

            for turn in unprocessed_turns:
                conversation_context.append({
                    "role": "user",
                    "content": turn.user_message
                })
                conversation_context.append({
                    "role": "assistant",
                    "content": turn.assistant_message
                })
            # 
            self.analysis_conversation(conversation_context)

            tool_calls = getattr(self.analysis_agent, "tool_calls", [])

            operations = []

            summary = {"add":0, "update":0, "delete":0}

            for tool_call in tool_calls:
                if tool_call.tool_name == "add_memory":
                    operations.append({
                        'action': 'add',
                        'result': tool_call.result,
                        'content': tool_call.arguments.get("content"),
                    })
                    if tool_call.result and tool_call.result.get("success"):
                        summary["add"] += 1

                elif tool_call.tool_name == "update_memory":
                    operations.append({
                        'action': 'update',
                        'result': tool_call.result,
                        'content': tool_call.arguments.get("content"),
                        'memory_id': tool_call.arguments.get("memory_id"),
                    })
                    if tool_call.result and tool_call.result.get("success"):
                        summary["update"] += 1
                elif tool_call.tool_name == "delete_memory":
                    operations.append({
                        'action': 'delete',
                        'result': tool_call.result,
                        'memory_id': tool_call.arguments.get("memory_id"),
                    })
                    if tool_call.result and tool_call.result.get("success"):
                        summary["delete"] += 1
                
            # 清空工具调用列表
            self.analysis_agent.tool_calls = []

            return {
                'analyzed_turns': len(unprocessed_turns),
                'operations': operations,
                'summary': summary,
                'details': operations  # Operations are the details
            }

# if __name__ == "__main__":
#     background_memory_processor = BackgroundMemoryProcessor(user_id="jxy001a", mode=MemoryMode.NOTES,model=Config.MODEL_NAME)
#     print(background_memory_processor.user_id)
#     print(background_memory_processor.model)
#     print(background_memory_processor.memory_manager)
