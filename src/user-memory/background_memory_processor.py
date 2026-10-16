from dataclasses import dataclass, field, asdict
import json
import logging
from tokenize import OP
from typing import List, Dict, Any, Optional

from config import Config, MemoryMode
from memory_manager import create_memory_manager



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

        self.memory = create_memory_manager(user_id=self.user_id, mode=mode)
        

if __name__ == "__main__":
    background_memory_processor = BackgroundMemoryProcessor(user_id="jxy001a", mode=MemoryMode.NOTES)
    # print(json.dumps(config.to_dict(), indent=4))
