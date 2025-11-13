import agentscope
from agentscope.agent import ReActAgent
from agentscope.message import TextBlock, ToolUseBlock
from agentscope.tool import ToolResponse, Toolkit, execute_python_code
from agentscope.model import DashScopeChatModel
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeChatFormatter

from tools.ToolFileList import load_file_tree
import os
from typing import Optional

class ModelDeployAgent:
    def __init__(self):
        self.agent_model = None
    
    @staticmethod
    def build_toolkit() -> Toolkit:
        toolkit = Toolkit()
        toolkit.register_tool_function(load_file_tree)
        return toolkit

    def init_agent_model(self, agent_name: str = "Jarvis", model_name: str = "qwen-max", api_key: Optional[str] = None, **kwargs):
        try:
            agent_model = ReActAgent(
                name=agent_name,
                sys_prompt=f"You are {agent_name}, an AI model deployment assistant.",
                model=DashScopeChatModel(
                    model_name=model_name,
                    api_key=api_key if api_key is not None else os.environ["DASHSCOPE_API_KEY"],
                ),
                memory=InMemoryMemory(),
                formatter=DashScopeChatFormatter(),
                toolkit=self.build_toolkit(),
                parallel_tool_calls=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize agent model: {e}")
        self.agent_model = agent_model

    def get_agent_model(self) -> ReActAgent:
        if self.agent_model is None:
            raise ValueError("Agent model is not initialized. Please call init_agent_model() first.")
        return self.agent_model
        



