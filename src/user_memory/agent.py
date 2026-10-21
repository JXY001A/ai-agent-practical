from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Dict, Any, List, Optional, Tuple
from config import Config, MemoryMode
from openai import OpenAI
from user_memory.memory_manager import create_memory_manager
from user_memory.conversation_history import ConversationHistory, ConversationTurn
import uuid
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ToolCall:
    """Represents a single tool call with tracking"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UserMemoryConfig:
    """Configuration for the user memory agent"""
    enable_memory_updates: bool = True
    enable_conversation_history: bool = True
    enable_memory_search: bool = True
    memory_mode: MemoryMode = MemoryMode.NOTES
    max_memory_context: int = 10  # Max memory items to include in context
    save_trajectory: bool = True
    trajectory_file: str = "memory_trajectory.json"



class UserMemoryAgent:
    def __init__(self, 
        api_key: str= Config.API_KEY,
        user_id: str= "jxy001a",
        model: str= Config.MODEL_NAME ,
        config: Optional[UserMemoryConfig] = None):


        self.api_key = api_key
        self.user_id = user_id
        self.model = model
        self.tool_calls = []

        self.config = config or UserMemoryConfig()
        self.memory_manager = create_memory_manager(user_id=self.user_id, mode=self.config.memory_mode)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=Config.BASE_URL,
        )
        self.conversation_history = ConversationHistory(user_id=self.user_id)
        
        self.tools: List[ToolCall] = []
        self.tool_call_counts: Dict[str, int] = {}

        self.conversation = []
        self.session_id = self._start_session()
        
        self._init_system_prompt()
        

    def _init_system_prompt(self):
        """Initialize the system prompt."""
        base_prompt = """You are an intelligent assistant with persistent memory across conversations. 
            You have access to various tools to manage user memories and search conversation history. If you want to add, update, or delete multiple memories, you should use multiple tool calls at once. After you have finished updating memories, you should output STOP without any other text.

            The full history of the latest conversation is automatically loaded in the context below.

            ## Key Behaviors:
            1. All user memories are automatically loaded and shown in the "USER MEMORIES" section below
            2. Proactively update memories when learning new information about the user
            3. Reference relevant memories when responding
            4. Maintain consistency with previously stored information
            5. Be personalized based on what you know about the user

            """
        # Add mode-specific memory instructions
        if self.config.memory_mode == MemoryMode.NOTES:
            memory_instructions = """## Memory Management:
            - All user memories are pre-loaded in the context below
            - Use `add_memory` to store new important information about the user
            - Use `update_memory` to modify existing memories
            - Use `delete_memory` to remove outdated or incorrect memories

            Keep memories as simple facts or preferences."""
        
        elif self.config.memory_mode == MemoryMode.ENHANCED_NOTES:
            memory_instructions = """## Memory Management:
            - All user memories are pre-loaded in the context below
            - Use `add_memory` to store new important information about the user
            - Use `update_memory` to modify existing memories
            - Use `delete_memory` to remove outdated or incorrect memories

            IMPORTANT: Each note should contain all important factual information and user preferences in a complete, contextual manner.
            Notes can be full paragraphs that capture the complete context, not just simple key-value pairs.

            Example of good enhanced notes:
            - "User works at TechCorp as a senior software engineer, specializing in machine learning. They've been there for 3 years and enjoy the collaborative culture."
            - "User's email is john.doe@techcorp.com for work and johndoe.personal@gmail.com for personal matters. They prefer work emails during business hours only."
            - "User has two children: Sarah (8 years old, loves soccer) and Michael (5 years old, interested in dinosaurs). Both attend Oakwood Elementary School."

            Extract all factual information from the conversation that may be useful for future interactions."""

        elif self.config.memory_mode == MemoryMode.JSON_CARD:
            memory_instructions = """## Memory Management:
                - All user memories are pre-loaded in the context below
                - Use `add_memory` to store new memory cards with structured data
                - Use `update_memory` to modify existing memory cards
                - Use `delete_memory` to remove outdated memory cards

                Memory cards use a hierarchical structure: category -> subcategory -> key -> value

                Example operations:
                1. Adding a memory card:
                content: {"category": "personal", "subcategory": "contact", "key": "email", "value": "user@example.com"}

                2. Updating a memory card:
                memory_id: "personal.contact.email"
                content: {"value": "newemail@example.com"}

                3. Structure examples:
                - personal.preferences.coding_style -> "prefers functional programming"
                - work.projects.current -> "developing AI chatbot"
                - family.children.sarah -> {"age": 8, "interests": ["soccer", "reading"]}"""

        elif self.config.memory_mode == MemoryMode.ENHANCED_JSON_CARD:
            memory_instructions = """## Memory Management:
            - All user memory cards are pre-loaded in the context below
            - Use `add_memory` to store complete memory card objects
            - Use `update_memory` to modify existing memory cards
            - Use `delete_memory` to remove memory cards

            Memory cards are complete JSON objects within categories. Each card MUST include:
            - backstory: Context about when/why this information was learned (1-2 sentences)
            - date_created: Current timestamp (YYYY-MM-DD HH:MM:SS)
            - person: Who this relates to (e.g., "John Smith (primary)", "Sarah Smith (daughter)")
            - relationship: Role/relationship (e.g., "primary account holder", "family member")
            - Additional relevant fields based on the information type

            Example memory card operations:

            1. Adding a complete memory card:
            content: {
                "category": "financial",
                "card_key": "bank_account_primary",
                "card": {
                    "backstory": "User shared their banking details while setting up automatic bill payments",
                    "date_created": "2024-01-15 10:30:00",
                    "person": "John Smith (primary)",
                    "relationship": "primary account holder",
                    "bank_name": "Chase Bank",
                    "account_type": "checking",
                    "account_ending": "4567",
                    "routing_number": "021000021",
                    "purpose": "primary checking for bills"
                }
            }

            2. Adding a medical memory card:
            content: {
                "category": "medical",
                "card_key": "doctor_dermatologist_sarah",
                "card": {
                    "backstory": "User needed to schedule a dermatology appointment for their daughter's skin condition",
                    "date_created": "2024-01-16 14:00:00",
                    "person": "Sarah Smith (daughter)",
                    "relationship": "family member",
                    "doctor_name": "Dr. Emily Johnson",
                    "specialty": "Pediatric Dermatology",
                    "clinic": "Children's Health Center",
                    "phone": "555-0123",
                    "condition_treated": "eczema"
                }
            }

            CRITICAL: The backstory and person fields prevent confusion. For example, without proper person identification, 
            a dermatologist for a child might be mistakenly suggested for an elderly parent's Alzheimer's care."""

        else:
            memory_instructions = """## Memory Management:
                - All user memories are pre-loaded in the context below
                - Use `add_memory` to store new important information about the user
                - Use `update_memory` to modify existing memories
                - Use `delete_memory` to remove outdated or incorrect memories"""
            
        system_content = base_prompt + memory_instructions + """

            Current Memory Context will be provided with each message."""
        
        self.conversation = [{
            "role":"system",
            "content": system_content
        }]

    def _start_session(self) -> str:
        """Start a new session"""
        return f"session-{uuid.uuid4().hex[:8]}"

    def _get_memory_context(self) -> str:
        """Get the memory context for the current session"""
        context_parts = []
        context_parts.append("=== USER MEMORIES  ===")
        context_parts.append(self.memory_manager.get_context_string())
        context_parts.append("")

        if self.conversation_history:
            all_conversations: List[ConversationTurn] = self.conversation_history.conversations if hasattr(self.conversation_history, "conversations") else []
            if all_conversations:
                context_parts.append("=== CONVERSATION HISTORY ===")
                context_parts.append(f"Total conversations: {len(all_conversations)}")
                context_parts.append('')
                
                for turn in all_conversations:
                    context_parts.append(f"[Conversation ID: {turn.session_id}, Turn: {turn.turn_number}]")
                    context_parts.append(f"User: {turn.user_message}")
                    context_parts.append(f"Assistant: {turn.assistant_message}")
                    context_parts.append('')

        return '\n'.join(context_parts)

    def _get_tools_description(self) -> List[Dict[str, Any]]:
        """Get tool descriptions for the model"""
        tools = []
        
        tools.extend([
            {
                "type": "function",
                "function": {
                    "name": "add_memory",
                    "description": "Add a new memory about the user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The memory content to store"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional tags for categorizing the memory"
                            }
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_memory",
                    "description": "Update an existing memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "ID of the memory to update"
                            },
                            "content": {
                                "type": "string",
                                "description": "New content for the memory"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional new tags"
                            }
                        },
                        "required": ["memory_id", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_memory",
                    "description": "Delete a memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "ID of the memory to delete"
                            }
                        },
                        "required": ["memory_id"]
                    }
                }
            }
        ])
        
        return tools
    
    # 添加记忆
    def _tool_add_memory(self, content: Any, tags: List[str] = None) -> Dict[str, Any]:
        """Add a new memory"""
        if self.config.memory_mode in [MemoryMode.NOTES, MemoryMode.ENHANCED_NOTES]:
            # Both basic and enhanced notes use the same storage, just different prompts
            if isinstance(content, dict):
                # If content is a dict, extract string representation
                content_str = str(content)
            else:
                content_str = content
            
            memory_id = self.memory_manager.add_memory(
                content=content_str,
                session_id=self.session_id,
                tags=tags or []
            )
            
        elif self.config.memory_mode == MemoryMode.JSON_CARD:
            # Basic JSON cards mode
            if isinstance(content, dict):
                # Content should already have the structure
                memory_content = content
            else:
                try:
                    parsed_content = json.loads(content)
                except (TypeError, json.JSONDecodeError):
                    parsed_content = None

                if isinstance(parsed_content, dict):
                    memory_content = parsed_content
                else:
                    # Fallback for legacy "key: value" style content.
                    parts = str(content).split(':')
                    if len(parts) >= 2:
                        category = "personal"
                        subcategory = "info"
                        key = parts[0].strip().replace(' ', '_').lower()
                        value = ':'.join(parts[1:]).strip()
                    else:
                        category = "general"
                        subcategory = "notes"
                        key = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        value = content

                    memory_content = {
                        'category': category,
                        'subcategory': subcategory,
                        'key': key,
                        'value': value
                    }
            
            memory_id = self.memory_manager.add_memory(
                content=memory_content,
                session_id=self.session_id
            )
        # Advanced JSON cards mode
        elif self.config.memory_mode == MemoryMode.ENHANCED_JSON_CARD:
            # Advanced JSON cards mode
            if not isinstance(content, dict):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    return {
                        "success": False,
                        "message": "Advanced JSON cards mode requires properly structured JSON content"
                    }
            
            memory_id = self.memory_manager.add_memory(
                content=content,
                session_id=self.session_id
            )
        else:
            return {
                "success": False,
                "message": f"Unknown memory mode: {self.config.memory_mode}"
            }
        
        return {
            "success": True,
            "memory_id": memory_id,
            "message": f"Memory added successfully"
        }
    
    def _tool_update_memory(self, memory_id: str, content: Any, tags: List[str] = None) -> Dict[str, Any]:
        """Update an existing memory"""
        if self.config.memory_mode in [MemoryMode.NOTES, MemoryMode.ENHANCED_NOTES]:
            # Both basic and enhanced notes use the same storage
            if isinstance(content, dict):
                content_str = str(content)
            else:
                content_str = content
            
            success = self.memory_manager.update_memory(
                memory_id=memory_id,
                content=content_str,
                session_id=self.session_id,
                tags=tags
            )
            
        elif self.config.memory_mode == MemoryMode.JSON_CARD:
            # Basic JSON cards mode
            if isinstance(content, dict):
                memory_content = content
            else:
                # For JSON cards, parse the memory_id and content
                parts = memory_id.split('.')
                if len(parts) == 3:
                    memory_content = {'value': content}
                else:
                    return {
                        "success": False,
                        "message": "Invalid memory_id format for JSON cards"
                    }
            
            success = self.memory_manager.update_memory(
                memory_id=memory_id,
                content=memory_content,
                session_id=self.session_id
            )
            
        elif self.config.memory_mode == MemoryMode.ENHANCED_JSON_CARD:
            # Advanced JSON cards mode
            if not isinstance(content, dict):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    return {
                        "success": False,
                        "message": "Advanced JSON cards mode requires properly structured JSON content"
                    }
            
            success = self.memory_manager.update_memory(
                memory_id=memory_id,
                content=content,
                session_id=self.session_id
            )
        else:
            success = False
        
        return {
            "success": success,
            "message": "Memory updated successfully" if success else "Memory not found"
        }
    
    def _tool_delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory"""
        self.memory_manager.delete_memory(memory_id)
        return {
            "success": True,
            "message": f"Memory {memory_id} deleted"
        }
    # 执行工具
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """
        Execute a tool and return the result
        
        Returns:
            Tuple of (result, error_message)
        """
        try:
            if tool_name == "add_memory":
                result = self._tool_add_memory(**arguments)
            elif tool_name == "update_memory":
                result = self._tool_update_memory(**arguments)
            elif tool_name == "delete_memory":
                result = self._tool_delete_memory(**arguments)
            else:
                error = f"Unknown tool: {tool_name}"
                return {"error": error}, error
            
            return result, None
            
        except Exception as e:
            error_msg = f"Tool '{tool_name}' failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}, error_msg

    def _save_trajectory(self, iteration: int, final_answer: Optional[str] = None):
        """Save current trajectory to file for debugging"""
        if not self.config.save_trajectory:
            return
        
        trajectory_data = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "model": self.model,
            "conversation": self.conversation,
            "tool_calls": [
                {
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "result": call.result,
                    "error": call.error,
                    "timestamp": call.timestamp
                }
                for call in self.tool_calls
            ],
            "memory_state": self.memory_manager.get_context_string(),
            "final_answer": final_answer
        }
        
        try:
            with open(self.config.trajectory_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Trajectory saved to {self.config.trajectory_file}")

        except Exception as e:
            logger.warning(f"Failed to save trajectory: {e}")

    def execute_task(self, task: str, max_iterations: int = 10) -> str:
        memory_context = self._get_memory_context()
        full_message = f"{task} \n\n {memory_context}"
        # logger.info(f"用户请求: {task}")

        self.conversation.append({"role": "user", "content": full_message})

        iteration = 0 
        final_answer = None

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}")
            self._save_trajectory(iteration)
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages= self.conversation,
                max_tokens=Config.MODEL_MAX_TOKENS,
                temperature=Config.MODEL_TEMPERATURE,
                tools=self._get_tools_description(),
                tool_choice="auto",
                stream=True
            )

            collected_content = []
            current_tool_calls = []

            collected_reasoning = []

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # DeepSeek 思考模式的推理字段名是 reasoning_content
                reasoning_delta = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning_delta:
                    collected_reasoning.append(reasoning_delta)

                if delta.content:
                    collected_content.append(delta.content)

                # tool_calls 分片常在 content 为空时到达，必须独立判断
                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        # 列表靠 index 定位累积流式分片，但 index 可能跳着来、交错来，所以每次写入前先把列表垫到该 index 长度，防止越界。
                        if tool_call_delta.index is not None:
                            while len(current_tool_calls) <= tool_call_delta.index:
                                current_tool_calls.append({
                                    "id":"",
                                    "type":"function",
                                    "function": {
                                        "name":"",
                                        "arguments":""
                                    }
                                })
                        if tool_call_delta.id:
                            current_tool_calls[tool_call_delta.index]["id"] = tool_call_delta.id

                        if tool_call_delta.function:
                            if tool_call_delta.function.name:
                                current_tool_calls[tool_call_delta.index]["function"]["name"] = tool_call_delta.function.name
                            if tool_call_delta.function.arguments:
                                # arguments 是逐片下发的 JSON 片段，必须累加而不是覆盖
                                current_tool_calls[tool_call_delta.index]["function"]["arguments"] += tool_call_delta.function.arguments

                            
            complete_message = {
                "role":"assistant",
            }

            if collected_reasoning:
                reasoning_text = ''.join(collected_reasoning)
                # 思考模式下必须把推理内容原样回传给 API，否则带 tool_calls 的下一轮会 400
                complete_message["reasoning_content"] = reasoning_text

            if collected_content:
                content_text = ''.join(collected_content)
                complete_message["content"] = content_text

            if current_tool_calls:
                complete_message["tool_calls"] = current_tool_calls


            if complete_message.get("reasoning_content") or complete_message.get("content") or current_tool_calls:
                self.conversation.append(complete_message)

            
            if current_tool_calls:
                for tool_call in current_tool_calls:
                    tool_name  = tool_call["function"]["name"]
                    try:
                        tool_call_arguments = json.loads(tool_call["function"]["arguments"] or "{}")
                    except json.JSONDecodeError as exc:
                        error_msg = f"Invalid tool arguments (not valid JSON): {exc}"
                        print(f"\n  ❌ Tool Error: {error_msg}")

                        self.conversation.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps({"error": error_msg})
                        })
                        continue
                        
                    self.tool_call_counts[tool_name] = self.tool_call_counts.get(tool_name, 0) + 1

                    result, error = self._execute_tool(tool_name, tool_call_arguments)
                    if error:
                        print(f"\n  ❌ Tool Error: {error}")
                    else:
                        print(f"\n  ✅ Tool Result: {json.dumps(result)[:200]}")

                    tool_call_record = ToolCall(
                        tool_name=tool_name,
                        arguments= tool_call_arguments,
                        result= result,
                        error=error,
                    )

                    self.tool_calls.append(tool_call_record)

                    self.conversation.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result)
                    })
            elif complete_message.get("content")or complete_message.get("reasoning_content"):
                final_answer = complete_message.get("content") or complete_message.get("reasoning_content")

                if complete_message.get("reasoning_content") and complete_message.get("content"):
                    logger.info(f"Response complete with reasoning and content")
                else:
                    logger.info(f"Response complete (no more tool calls): {final_answer}")

                if self.conversation_history:
                        self.conversation_history.add_turn(
                            session_id=self.session_id,
                            user_message=task,
                            assistant_message=final_answer
                        )
                    
                # Save final trajectory
                self._save_trajectory(iteration, final_answer)
                break
        
        # Save final trajectory
        self._save_trajectory(iteration, final_answer)
        result = {
            "final_answer": final_answer,
            "tool_calls": self.tool_calls,
            "iterations": iteration,
            "success": final_answer is not None,
            "memory_state": self.memory_manager.get_context_string(),
            "trajectory_file": self.config.trajectory_file if self.config.save_trajectory else None
        }
        
        if self.conversation and isinstance(self.conversation[-1], dict):
            last_message = self.conversation[-1]
            if last_message.get("role") == "assistant" and last_message.get("reasoning_content"):
                result["reasoning"] = last_message["reasoning_content"]
        
        return result
        
        

    def reset(self):
        """Reset the agent's state for a new conversation"""
        self.tool_calls = []
        self.tool_call_counts = {}
        self.session_id = self._start_session()
        self._init_system_prompt()
        logger.info("Agent state reset")


if __name__ == "__main__":
    agent = UserMemoryAgent()
