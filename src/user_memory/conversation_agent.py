import logging
from config import Config, MemoryMode
from openai import OpenAI
from user_memory.memory_manager import create_memory_manager
from user_memory.conversation_history import ConversationHistory, ConversationTurn
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConversationConfig:
    """Configuration for the conversational agent"""
    enable_memory_context: bool = True  # Include memory in context but don't update
    enable_conversation_history: bool = True
    max_memory_context: int = 10
    temperature: float = 0.7
    max_tokens: int = Config.MODEL_MAX_TOKENS



class ConversationAgent:
    def __init__(self, 
        user_id: str,
        api_key: str=Config.API_KEY,
        model: str=Config.MODEL_NAME,
        config: ConversationConfig = None,
        memory_mode: MemoryMode=MemoryMode.NONE):
        
        self.user_id = user_id
        self.api_key = api_key
        self.model = model
        self.config = config or ConversationConfig()
        self.memory_mode = memory_mode

        self.client = OpenAI(api_key=api_key, base_url=Config.BASE_URL)

        self.memory_manager = create_memory_manager(user_id, memory_mode)

        self.conversation_history = ConversationHistory(user_id)

        self.session_id = self._generate_session_id()
        self.conversation = []
        self._init_system_prompt()

    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return f"session-{uuid4().hex[:8]}"

    def _init_system_prompt(self):
        """Initialize the system prompt"""
        system_content = """
            You are a helpful and personalized assistant. You have access to information about the user from previous conversations, which helps you provide personalized and contextual responses.
            You MUST analyze the context, user's questions and memories in detail, and provide a comprehensive and detailed response.
        """

        self.conversation = [
            {
                "role": "system",
                "content": system_content
            }
        ]
    def _get_memory_context(self) -> str:
        """Get the memory context"""
        self.memory_manager.load_memory()
        
        memory_str = self.memory_manager.get_context_string()
        context_parts = []
        if memory_str:
            context_parts.append("=== BEGIN USER MEMORIES ===")
            context_parts.append(memory_str)
            context_parts.append("")
        
        if self.conversation_history:
            session_turns = self.conversation_history.get_session_turns(self.session_id)
            if session_turns:
                context_parts.append("=== CURRENT CONVERSATION HISTORY ===")
                context_parts.append(f"Total turns: {len(session_turns)}")
                for turn in session_turns:
                    context_parts.append(f"[Session: {turn.session_id}, Turn: {turn.turn_number},Time:{turn.timestamp}]")
                    context_parts.append(f"User: {turn.user_message}")
                    context_parts.append(f"Assistant: {turn.assistant_message}")
                    context_parts.append("")

        return "\n".join(context_parts)

    def chat(self, message: str) -> str:
        """Generate a response to the user message"""
        context_str = self.memory_manager.get_context_string()
        full_message = message
        if context_str:
            full_message = f"{message}\n{context_str}"

        self.conversation.append({
            "role": "user",
            "content": message,
        })
        api_message  =  self.conversation[:-1] + [{"role": "user", "content": full_message}]
        assistant_message = ''
        
        try:
            stream = self.client.chat.completions.create(
            model=self.model,
            messages=api_message,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True
        )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    assistant_message += delta
                    # print(delta, end='', flush=True)
        except Exception as e:
            logger.error(f"Error during API call: {e}")
            return f"Error during API call: {e}";

        print()

        self.conversation.append({
            "role": "assistant",
            "content": assistant_message,
        })

        if self.conversation_history:
            self.conversation_history.add_turn(self.session_id, message, assistant_message)
        
        return assistant_message

    def reset_session(self):
        """Start a new conversation session"""
        self.session_id = self._generate_session_id()
        self._init_system_prompt()
        logger.info(f"Started new session: {self.session_id}")
    
    def get_session_id(self) -> str:
        """Get the current session ID"""
        return self.session_id