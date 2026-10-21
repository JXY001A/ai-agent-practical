
import json
import logging
from math import log
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single conversation turn"""
    session_id: str
    user_message: str
    assistant_message: str
    timestamp: str
    turn_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary"""
        return cls(**data)


class  ConversationHistory:
    """Manages conversation history with optional vector search"""
    
    def __init__(self, user_id: str):
        """
        Initialize conversation history manager
        
        Args:
            user_id: User identifier
        """
        self.user_id = user_id
        self.history_file = os.path.join(
            Config.CONVERSATION_HISTORY_DIR,
            f"{user_id}_history.json"
        )
        self.conversations: List[ConversationTurn] = []
        self.load_history()
        
        # Initialize Dify client if configured
        self.dify_client = None
        # if Config.ENABLE_HISTORY_SEARCH and Config.DIFY_API_KEY:
        #     self.dify_client = DifySearchClient()
    
    def load_history(self):
        """Load conversation history from storage"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversations = [
                        ConversationTurn.from_dict(turn) 
                        for turn in data.get('conversations', [])
                    ]
                logger.info(f"Loaded {len(self.conversations)} conversation turns for user {self.user_id}")
            except Exception as e:
                logger.error(f"Error loading conversation history: {e}")
                self.conversations = []
        else:
            self.conversations = []
        
        if not self.conversations:
            logger.info(f"没有加载到会话历史对于用户： {self.user_id}")
    
    def save_history(self):
        """Save conversation history to storage"""
        try:
            os.makedirs(os.path.dirname(self.history_file) or ".", exist_ok=True)
            # Write to a temp file then atomically replace: a crash mid-dump
            # must not truncate the only copy of the persisted data.
            tmp_file = self.history_file + '.tmp'
            with open(tmp_file, 'w', encoding='utf-8') as f:
                data = {
                    'user_id': self.user_id,
                    'updated_at': datetime.now().isoformat(),
                    'conversations': [turn.to_dict() for turn in self.conversations]
                }
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_file, self.history_file)
            logger.info(f"Saved {len(self.conversations)} conversation turns")
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")
    
    def add_turn(self, session_id: str, user_message: str, assistant_message: str):
        """
        Add a conversation turn
        
        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_message: Assistant's response
        """
        turn = ConversationTurn(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            timestamp=datetime.now().isoformat(),
            turn_number=len(self.conversations) + 1
        )
        self.conversations.append(turn)
        self.save_history()
        
    
    def get_recent_turns(self, limit: int = 10) -> List[ConversationTurn]:
        """
        Get recent conversation turns
        
        Args:
            limit: Maximum number of turns to return
            
        Returns:
            List of recent conversation turns
        """
        # limit<=0 → []; list[-0:] would return the full list.
        if limit <= 0 or not self.conversations:
            return []
        return self.conversations[-limit:]
    
    def get_session_turns(self, session_id: str) -> List[ConversationTurn]:
        """
        Get all turns from a specific session
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation turns from the session
        """
        return [turn for turn in self.conversations if turn.session_id == session_id]
    
    def search_history(self, query: str, limit: int = 5) -> List[ConversationTurn]:
        """
        Search conversation history
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching conversation turns
        """
        if limit <= 0 or not self.conversations:
            return []

        
        # Fallback to simple text search
        query_lower = query.lower()
        results = []
        
        for turn in self.conversations:
            if (query_lower in turn.user_message.lower() or 
                query_lower in turn.assistant_message.lower()):
                results.append(turn)
                if len(results) >= limit:
                    break
        
        return results

