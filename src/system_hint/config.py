from json import load
import os 
from typing import Optional
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path 

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# print(f'MODEL_NAME: {os.getenv("MODEL_NAME")}')
# print(f'VLLM_HOST: {os.getenv("VLLM_HOST")}')

@dataclass
class AgentConfig:
    """Agent configuration."""
    api_key: Optional[str] = None
    model: Optional[str] = None

    # system hint feature

    enable_timestamps:bool = True
    enable_tool_counter:bool = True
    enable_todo_list:bool = True
    enable_detailed_errors:bool = True
    enable_system_state:bool = True

    timestamp_format:str = "%Y-%m-%d %H:%M:%S"
    simulate_time_state:bool = False

    # Execution Options
    max_iterations: int = 20
    verbose: bool = False
    timeout: int = 30  # seconds for command execution

    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.getenv("API_KEY"),
            model=os.getenv("LLM_MODEL"),
            enable_timestamps=os.getenv("ENABLE_TIMESTAMPS", "true").lower() == "true",
            enable_tool_counter=os.getenv("ENABLE_TOOL_COUNTER", "true").lower() == "true",
            enable_todo_list=os.getenv("ENABLE_TODO_LIST", "true").lower() == "true",
            enable_detailed_errors=os.getenv("ENABLE_DETAILED_ERRORS", "true").lower() == "true",
            enable_system_state=os.getenv("ENABLE_SYSTEM_STATE", "true").lower() == "true",
            timestamp_format=os.getenv("TIMESTAMP_FORMAT", "%Y-%m-%d %H:%M:%S"),
            simulate_time_delay=os.getenv("SIMULATE_TIME_DELAY", "false").lower() == "true",
            max_iterations=int(os.getenv("MAX_ITERATIONS", "20")),
            verbose=os.getenv("VERBOSE", "false").lower() == "true",
            timeout=int(os.getenv("COMMAND_TIMEOUT", "30"))
        )

PRESETS = {
    "full": AgentConfig(
        enable_timestamps=True,
        enable_tool_counter=True,
        enable_todo_list=True,
        enable_detailed_errors=True,
        enable_system_state=True
    ),
    "minimal": AgentConfig(
        enable_timestamps=False,
        enable_tool_counter=False,
        enable_todo_list=False,
        enable_detailed_errors=False,
        enable_system_state=False
    ),
    "debug": AgentConfig(
        enable_timestamps=True,
        enable_tool_counter=True,
        enable_todo_list=True,
        enable_detailed_errors=True,
        enable_system_state=True,
        verbose=True
    ),
    "demo": AgentConfig(
        enable_timestamps=True,
        enable_tool_counter=True,
        enable_todo_list=True,
        enable_detailed_errors=True,
        enable_system_state=True,
        simulate_time_delay=True
    )
}

def get_config(preset: Optional[str] = None) -> AgentConfig:
    if preset and preset in PRESETS:
        return PRESETS[preset]
    else:
       return AgentConfig.from_env()
