from abc import ABC, abstractmethod
from typing import List, Optional
from config.config import Config
import asyncio
import json
import os

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit, ToolResponse, execute_python_code, execute_shell_command

class AgentBase(ABC):
    def __init__(self, configs: Config):
        self.configs = configs
        self.toolkit = Toolkit()

    @abstractmethod
    def _build_toolkit(self, **kwargs) -> Toolkit:
        pass

    @abstractmethod
    def create_agent(self, **kwargs) -> ReActAgent:
        pass

    @abstractmethod
    def example_msg(self, **kwargs) -> Msg:
        pass
