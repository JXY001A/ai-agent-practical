
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from abc import ABC, abstractmethod
from config import Config, MemoryMode
from pathlib import Path
import os
import uuid
import time
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _normalize_text(text: str) -> str:
    """ 规范文本以进行重复检测：小写和折叠空格。
        由离线整合/数据消除逻辑使用，以便仅记录不同的内容
        在套管或间距上被认为是相同的事实。
    """
    return " ".join(str(text or "").lower().split())

@dataclass 
class MemoryNote:
    note_id: str
    content: str
    session_id: str
    created_at: int
    updated_at: int
    # 每生成一个实例的时候自动调用 list 构造函数生成一个新的空列表，用于存储标签
    tags: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

# 记忆卡片
@dataclass
class MemoryCard:
    category: str
    subCategory: str
    key: str
    value: str
    session_id: str
    created_at: int
    updated_at: int
    def to_dict(self):
        return asdict(self)
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

class BaseMemoryManager(ABC):
    def __init__(self,user_id:str):
        self.user_id = user_id
        self.memory_file = Path(__file__).parent / Config.MEMORY_STORAGE_DIR / f"{self.user_id}_memory.json"
        self.MAX_MEMORY_ITEMS = 100
        self.load_memory()

    def _existing_file_type(self):
        """读取现有记忆文件的 type 字段，用于防止不同模式互相覆盖"""
        if not os.path.exists(self.memory_file):
            return None
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f).get("type")
        except Exception:
            return None

    @abstractmethod
    def load_memory(self):
        pass
    
    @abstractmethod
    def save_memory(self):
        pass
    

    @abstractmethod
    def add_memory(self,content:any,session_id:str,**kwargs):
        pass
    
    @abstractmethod
    def delete_memory(self,memory_id:str):
        pass
    
    
    @abstractmethod
    def update_memory(self,memory_id:str,content:any):
        pass

    @abstractmethod
    def search_memories(self,query:str,session_id:str,**kwargs):
        pass

    @abstractmethod
    def get_context_string(self,memory_id:str):
        pass


class NoteMemoryManager(BaseMemoryManager):
    def __init__(self, user_id: str):
        self.notes: list[MemoryNote] = []
        super().__init__(user_id)

    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    file_type = data.get("type")
                    if file_type and file_type != "notes":
                        logger.warning(
                            f"记忆文件格式为 {file_type}，与 NOTES 模式不匹配，已跳过加载：{self.memory_file}"
                        )
                        self.notes = []
                    else:
                        self.notes = [MemoryNote.from_dict(note) for note in data.get("notes", [])]
                logger.info(f"Loaded {len(self.notes)} notes for user {self.user_id}")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
                self.notes = []
        else:
            self.notes = []
            logger.info(f"对于用户：{self.user_id} 而言，没有找到记忆文件 ")

    def save_memory(self):
        try:
            file_type = self._existing_file_type()
            if file_type and file_type != "notes":
                logger.error(
                    f"拒绝写入：{self.memory_file} 当前为 {file_type} 格式，与 NOTES 模式不匹配"
                )
                return
            # 临时文件的意义
            # 为了让写入过程原子化，避免写崩时把唯一的持久化文件写坏
            #json.dump 是流式写入，先截断原文件再逐块写。若中途进程崩溃/断电，原文件已被清空或只剩半截 JSON——旧数据没了，新数据也不完整，下次 load_memory 直接解析失败。
            # 读者（其他进程或本进程再次 load）可能读到写了一半的文件。
            os.makedirs(self.memory_file.parent, exist_ok=True)
            tmp_file = self.memory_file.with_name(self.memory_file.name + ".tmp")  # 临时文件
            data = {
                "user_id":self.user_id,
                "type":"notes",
                "update_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "notes":[note.to_dict() for note in self.notes]
            }
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
            os.replace(tmp_file, self.memory_file)
            logger.info(f"保存 {len(self.notes)} notes 为用户 {self.user_id}")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def add_memory(self,content: str, session_id: str, tags: List[str] = None):
        note  = MemoryNote(
            note_id=str(uuid.uuid4()),
            content=content,
            session_id=session_id,
            created_at=int(time.time()),
            updated_at=int(time.time()),
            tags=tags or []
        )
        self.notes.append(note)

        if len(self.notes) > self.MAX_MEMORY_ITEMS:
            old_count = len(self.notes)
            # 保留更新时间最晚的 MAX_MEMORY_ITEMS 个记忆
            # 原地降序排序，默认升序 
            self.notes.sort(key=lambda n: n.updated_at, reverse=True)
            self.notes = self.notes[:self.MAX_MEMORY_ITEMS]
            logger.info(f"保留了 {len(self.notes)} 个记忆，删除了 {old_count - len(self.notes)} 个记忆")
        
        self.save_memory()
    
    def update_memory(self, memory_id: str, content: str, session_id: str, tags: List[str] = None):
        """Update an existing note"""
        for note in self.notes:
            if note.note_id == memory_id:
                old_content = note.content
                note.content = content
                note.session_id = session_id
                note.updated_at = datetime.now().isoformat()
                if tags is not None:
                    note.tags = tags
                
                self.save_memory()
                return True
        
        return False
    
    def delete_memory(self, memory_id: str):
        """Delete a note"""
        original_count = len(self.notes)
        deleted_note = None
        for note in self.notes:
            if note.note_id == memory_id:
                deleted_note = note
                break
        
        self.notes = [note for note in self.notes if note.note_id != memory_id]
        
        
        self.save_memory()
    
    def clear_all_memories(self):
        """Clear all memories for this user - useful for testing"""
        self.notes = []
        self.save_memory()
        logger.info(f"Cleared all memories for user {self.user_id}")
        print(f"  🧹 Cleared all memories for user {self.user_id}")
    
    def get_context_string(self) -> str:
        """Get notes as formatted string for LLM context"""
        if not self.notes:
            return "No previous memory notes available."
        
        context = "User Memory Notes:\n\n"
        for i, note in enumerate(self.notes, 1):
            context += f"Note {i} (ID: {note.note_id}, Session: {note.session_id}):\n"
            context += f"  Content: {note.content}\n"
            if note.tags:
                context += f"  Tags: {', '.join(note.tags)}\n"
            context += f"  Updated: {note.updated_at}\n\n"
        
        return context
    
    def search_memories(self, query: str) -> List[MemoryNote]:
        """Search notes by query (simple text search)"""
        query_lower = query.lower()
        results = []
        for note in self.notes:
            if query_lower in note.content.lower() or any(query_lower in tag.lower() for tag in note.tags):
                results.append(note)
        return results
    

class JSONCardMemoryManager(BaseMemoryManager):
    def __init__(self, user_id: str):
        self.memory_cards: Dict[str, Dict[str, Dict[str, Any]]] = {}

        super().__init__(user_id)

    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    file_type = data.get("type")
                    if file_type and file_type != "json_cards":
                        logger.warning(
                            f"记忆文件格式为 {file_type}，与 JSON_CARD 模式不匹配，已跳过加载：{self.memory_file}"
                        )
                        self.memory_cards = {}
                    else:
                        self.memory_cards = data.get('memory_cards', {})
                logger.info(f"Loaded memory cards for user {self.user_id}")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
                self.memory_cards = {}
        else:
            self.memory_cards = {}
            logger.info(f"对于用户：{self.user_id} 而言，没有找到记忆文件 ")
    
    def save_memory(self):
        try:
            file_type = self._existing_file_type()
            if file_type and file_type != "json_cards":
                logger.error(
                    f"拒绝写入：{self.memory_file} 当前为 {file_type} 格式，与 JSON_CARD 模式不匹配"
                )
                return
            os.makedirs(self.memory_file.parent, exist_ok=True)
            tmp_file = self.memory_file.with_name(self.memory_file.name + ".tmp")  # 临时文件
            data = {
                "user_id":self.user_id,
                "type":"json_cards",
                "update_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "memory_cards":self.memory_cards
            }
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            os.replace(tmp_file, self.memory_file)
            logger.info(f"保存记忆卡片为用户 {self.user_id}")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def add_memory(self, content: Dict[str, Any], session_id: str, **kwargs):
        category = content.get("category", "general")
        sub_category = content.get("subcategory", "info")
        key = content.get("key", str(uuid.uuid4()))
        value = content.get("value") or ''

        if category not in self.memory_cards:
            self.memory_cards[category] = {}

        if sub_category not in self.memory_cards[category]:
            self.memory_cards[category][sub_category] = {}
            
        self.memory_cards[category][sub_category][key] = {
            'value': value,
            'source': session_id,
            'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.save_memory()

        return f"{category}.{sub_category}.{key}"


    def update_memory(self, memory_id: str, content: Dict[str, Any], session_id: str, **kwargs):
        """Update an existing memory card"""
        parts = memory_id.split('.')
        if len(parts) != 3:
            return False
        
        category, subcategory, key = parts
        
        if (category in self.memory_cards and 
            subcategory in self.memory_cards[category] and 
            key in self.memory_cards[category][subcategory]):
            
            old_value = self.memory_cards[category][subcategory][key]['value']
            value = content.get('value')
            self.memory_cards[category][subcategory][key] = {
                'value': value,
                'source': session_id,
                'updated_at': datetime.now().isoformat()
            }
        
            
            self.save_memory()
            return True

        return False

    def delete_memory(self, memory_id: str):
        """Delete a memory card"""
        parts = memory_id.split('.')
        if len(parts) != 3:
            return
        
        category, subcategory, key = parts
        
        if (category in self.memory_cards and 
            subcategory in self.memory_cards[category] and 
            key in self.memory_cards[category][subcategory]):
            
            deleted_value = self.memory_cards[category][subcategory][key]['value']
            del self.memory_cards[category][subcategory][key]
            
            
            # Clean up empty subcategories and categories
            if not self.memory_cards[category][subcategory]:
                del self.memory_cards[category][subcategory]
            if not self.memory_cards[category]:
                del self.memory_cards[category]
            
            self.save_memory()

    def clear_all_memories(self):
        """Clear all memories for this user - useful for testing"""
        self.memory_cards = {}
        self.save_memory()
        logger.info(f"Cleared all memories for user {self.user_id}")
        print(f"  🧹 Cleared all memories for user {self.user_id}")
    
    def get_context_string(self) -> str:
        """Get memory cards as formatted string for LLM context"""
        if not self.memory_cards:
            return "No previous memory cards available."
        
        context = "User Memory Cards (Hierarchical JSON):\n\n"
        context += json.dumps(self.memory_cards, indent=2, ensure_ascii=False)
        return context
    
    def search_memories(self, query: str) -> List[Tuple[str, Any]]:
        """Search memory cards by query"""
        query_lower = query.lower()
        results = []
        
        for category, subcategories in self.memory_cards.items():
            for subcategory, items in subcategories.items():
                for key, data in items.items():
                    memory_path = f"{category}.{subcategory}.{key}"
                    value_str = str(data.get('value', '')).lower()
                    
                    if (query_lower in category.lower() or 
                        query_lower in subcategory.lower() or 
                        query_lower in key.lower() or 
                        query_lower in value_str):
                        
                        results.append((memory_path, data))
        
        return results

class AdvancedJSONMemoryManager(BaseMemoryManager):
    """
    Advanced JSON memory manager with complete memory card objects
    Structure: categories -> memory_card_key -> memory card (arbitrary JSON)
    """
    
    def __init__(self, user_id: str, verbose: bool = False):
        self.categories: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.verbose = verbose
        super().__init__(user_id)
    
    def load_memory(self):
        """Load advanced JSON memory cards from storage"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.categories = data.get('categories', {})
                logger.info(f"Loaded advanced memory cards for user {self.user_id}")
            except Exception as e:
                logger.error(f"Error loading advanced memory cards: {e}")
                self.categories = {}
        else:
            self.categories = {}
            logger.info(f"No existing memory file for user {self.user_id}")
    
    def save_memory(self):
        """Save advanced JSON memory cards to storage"""
        try:
            os.makedirs(os.path.dirname(self.memory_file) or ".", exist_ok=True)
            # Write to a temp file then atomically replace: a crash mid-dump
            # must not truncate the only copy of the persisted data.
            tmp_file = self.memory_file.with_name(self.memory_file.name + ".tmp")  # 临时文件
            with open(tmp_file, 'w', encoding='utf-8') as f:
                data = {
                    'user_id': self.user_id,
                    'type': 'advanced_json_cards',
                    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'categories': self.categories
                }
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_file, self.memory_file)
            logger.info(f"Saved advanced memory cards for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error saving advanced memory cards: {e}")
    
    def add_memory(self, content: Dict[str, Any], session_id: str, **kwargs):
        """
        Add a new memory card
        
        Args:
            content: Dictionary with 'category', 'card_key', and 'card' (complete memory card object)
            session_id: Session identifier
        
        Returns:
            Memory ID in format: category.card_key
        """
        category = content.get('category', 'general')
        card_key = content.get('card_key')
        card = content.get('card', {})
        
        if not card_key:
            card_key = str(uuid.uuid4())
        
        if category not in self.categories:
            self.categories[category] = {}
        
        # Add metadata to the card
        card['_metadata'] = {
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'source': session_id
        }
        
        # Ensure required fields
        if 'backstory' not in card:
            card['backstory'] = kwargs.get('backstory', '')
        if 'date_created' not in card:
            card['date_created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if 'person' not in card:
            card['person'] = kwargs.get('person', 'Unknown')
        if 'relationship' not in card:
            card['relationship'] = kwargs.get('relationship', 'primary account holder')
        
        self.categories[category][card_key] = card
        self.save_memory()
        
        return f"{category}.{card_key}"
    
    def update_memory(self, memory_id: str, content: Dict[str, Any], session_id: str, **kwargs):
        """
        Update an existing memory card
        
        Args:
            memory_id: Memory ID in format category.card_key
            content: Complete new memory card or partial updates
            session_id: Session identifier
        
        Returns:
            True if successful, False otherwise
        """
        parts = memory_id.split('.', 1)
        if len(parts) != 2:
            return False
        
        category, card_key = parts
        
        if category not in self.categories or card_key not in self.categories[category]:
            return False
        
        card = content.get('card', content)
        
        # Preserve existing metadata
        if '_metadata' in self.categories[category][card_key]:
            old_metadata = self.categories[category][card_key]['_metadata']
            card['_metadata'] = {
                'created_at': old_metadata.get('created_at', datetime.now().isoformat()),
                'updated_at': datetime.now().isoformat(),
                'source': session_id
            }
        else:
            card['_metadata'] = {
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'source': session_id
            }
        
        # Update the card
        self.categories[category][card_key] = card
        self.save_memory()
        
        return True
    
    def delete_memory(self, memory_id: str):
        """Delete a memory card"""
        parts = memory_id.split('.', 1)
        if len(parts) != 2:
            return
        
        category, card_key = parts
        
        if category in self.categories and card_key in self.categories[category]:
            del self.categories[category][card_key]
            
            # Clean up empty categories
            if not self.categories[category]:
                del self.categories[category]
            
            self.save_memory()
    
    def clear_all_memories(self):
        """Clear all memories for this user"""
        self.categories = {}
        self.save_memory()
        logger.info(f"Cleared all memories for user {self.user_id}")
        print(f"  🧹 Cleared all memories for user {self.user_id}")
    
    def get_context_string(self) -> str:
        """Get memory cards as formatted string for LLM context"""
        if not self.categories:
            return "No previous memory cards available."
        
        context = "User Memory Cards (Advanced JSON Structure):\n\n"
        for category, cards in self.categories.items():
            context += f"Category: {category}\n"
            for card_key, card in cards.items():
                # Remove internal metadata from display
                display_card = {k: v for k, v in card.items() if k != '_metadata'}
                context += f"  Card '{card_key}':\n"
                context += f"    {json.dumps(display_card, indent=4, ensure_ascii=False)}\n"
        
        return context
    
    def search_memories(self, query: str) -> List[Tuple[str, Any]]:
        """Search memory cards by query"""
        query_lower = query.lower()
        results = []
        
        for category, cards in self.categories.items():
            for card_key, card in cards.items():
                memory_id = f"{category}.{card_key}"
                
                # Search in category, card_key, and all card fields
                card_str = json.dumps(card, ensure_ascii=False).lower()
                
                if (query_lower in category.lower() or 
                    query_lower in card_key.lower() or 
                    query_lower in card_str):
                    
                    results.append((memory_id, card))
        
        return results

class NoMemoryManager(BaseMemoryManager):
    """NONE 模式：不保存、不读取任何记忆，所有接口为空实现"""

    def load_memory(self):
        self.memories = []

    def save_memory(self):
        return None

    def add_memory(self, content: any, session_id: str, **kwargs):
        return None

    def delete_memory(self, memory_id: str):
        return False

    def update_memory(self, memory_id: str, content: any):
        return False

    def search_memories(self, query: str, session_id: str = "", **kwargs):
        return []

    def get_context_string(self) -> str:
        return ""


def create_memory_manager(user_id: str, mode: MemoryMode | None = None) -> BaseMemoryManager:
    """
    Factory function to create appropriate memory manager
    
    Args:
        user_id: User identifier
        mode: Memory mode (defaults to config setting)
    
    Returns:
        Memory manager instance
    """
    mode = mode or MemoryMode(Config.MEMORY_MODE)
    
    if mode == MemoryMode.NONE:
        # 无记忆模式：返回空实现，调用方无需额外判断
        return NoMemoryManager(user_id)
    elif mode == MemoryMode.NOTES or mode == MemoryMode.ENHANCED_NOTES:
        # Both basic and enhanced notes use the same manager
        # The difference is in the prompts used by the agent
        return NoteMemoryManager(user_id)
    elif mode == MemoryMode.JSON_CARD:
        return JSONCardMemoryManager(user_id)
    elif mode == MemoryMode.ENHANCED_JSON_CARD:
        return AdvancedJSONMemoryManager(user_id)
    else:
        raise ValueError(f"Unknown memory mode: {mode}")


def ensure_memory_cleared(memory_manager: BaseMemoryManager, description: str = "memory") -> bool:
    """
    Ensures that all memory is cleared for a given memory manager.
    Used primarily for testing and evaluation to ensure clean state before each test case.
    
    Args:
        memory_manager: The memory manager to clear
        description: Description for logging (e.g., "agent memory", "processor memory")
        
    Returns:
        True if memory was successfully cleared, False otherwise
    """
    if not memory_manager:
        logger.warning(f"No memory manager provided for {description}")
        return False
    
    try:
        # Clear all memories
        if hasattr(memory_manager, 'clear_all_memories'):
            memory_manager.clear_all_memories()
        else:
            logger.warning(f"Memory manager for {description} doesn't support clear_all_memories()")
            return False
        
        # Verify memory is cleared by checking the context string
        context = memory_manager.get_context_string()
        is_cleared = "No previous memory" in context
        
        if is_cleared:
            logger.info(f"✅ {description} cleared successfully")
        else:
            logger.warning(f"⚠️ {description} may not be fully cleared. Context: {context[:100]}...")
        
        return is_cleared
        
    except Exception as e:
        logger.error(f"Error clearing {description}: {e}")
        return False


if __name__ == "__main__":
    # memory = MemoryNote("1", "This is a test note.", "session1", 1672531200, 1672531200)
    # print(json.dumps(memory.to_dict(), indent=4, ensure_ascii=False))
    # print(Config.API_KEY)
    print(111)
    noteMemory  = NoteMemoryManager("jxy001a")
    # noteMemory.add_memory("This is a test note.", "session1")

    for note in noteMemory.notes:
        print(json.dumps(note.to_dict(), indent=4, ensure_ascii=False))
    # print(f"\n noteMemory.memory_file: \n\n {noteMemory.} \n")
