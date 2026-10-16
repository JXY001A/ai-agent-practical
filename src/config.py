
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
from enum import Enum
from dataclasses import dataclass, asdict


# 必须显式指向 .env 文件：传目录时 python-dotenv 1.2.x 不会加载任何变量
load_dotenv(Path(__file__).parent.parent / ".env")

class MemoryMode(Enum):
    """记忆管理模式"""
    NOTES = "notes"
    ENHANCED_NOTES = "enhanced_notes"
    JSON_CARD = "json_card"
    ENHANCED_JSON_CARD = "enhanced_json_card"



@dataclass
class Config:
    # 模型名称
    MODEL_NAME: str = os.getenv("MODEL_NAME")
    # API 密钥
    API_KEY: str = os.getenv("API_KEY")
    # 基础 URL
    BASE_URL: str = os.getenv("BASE_URL")
    # 模型温度
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.3"))
    # 模型最大令牌数
    MODEL_MAX_TOKENS: int = int(os.getenv("MODEL_MAX_TOKENS", "8192"))
    # Serper 基础 URL
    SERPER_BASE_URL: str = os.getenv("SERPER_BASE_URL")
    # Serper API 密钥
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY")
    # 最大网页长度
    MAX_WEBPAGE_LENGTH: int = int(os.getenv("MAX_WEBPAGE_LENGTH", 50000))
    # 摘要最大令牌数
    SUMMARY_MAX_TOKENS: int = int(os.getenv("SUMMARY_MAX_TOKENS", 500))
    # Open Router 基础 URL
    OPEN_ROUTER_BASE_URL: str = os.getenv("OPEN_ROUTER_BASE_URL")
    # Open Router API 密钥
    OPEN_ROUTER_API_KEY: str = os.getenv("OPEN_ROUTER_API_KEY")
    # 记忆模式（notes / enhanced_notes / json_card / enhanced_json_card）
    MEMORY_MODE: str = os.getenv("MEMORY_MODE", "notes")
    # 内存存储目录
    MEMORY_STORAGE_DIR: str = "data/memories"
    
    # 会话历史目录
    CONVERSATION_HISTORY_DIR: str = "data/conversations"
    # 会话历史文件名
    CONVERSATION_HISTORY_FILE: str = "conversation_history.json"

    def to_dict(self):
        return asdict(self)
