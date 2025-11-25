import agentscope
from agentscope.agent import ReActAgent
from agentscope.message import TextBlock, ToolUseBlock
from agentscope.tool import ToolResponse, Toolkit, execute_python_code
from agentscope.model import DashScopeChatModel
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeChatFormatter, DashScopeMultiAgentFormatter

import os
from typing import Optional

# TODO: use uv for dependency management
def create_dependency_agent(agent_model_name: str) -> ReActAgent:
    return ReActAgent(
        name="DevOpsSpecialist",
        sys_prompt=
        """
        You are a DevOps specialist with expertise in creating deployment files for AI model function-services on Kubernetes.
        """,
        model=DashScopeChatModel(
            model_name=agent_model_name,
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=True,
        ),
        # Use multiagent formatter because the multiple entities will
        # occur in the prompt of the LLM API call
        formatter=DashScopeMultiAgentFormatter(),
    )

def create_build_agent(agent_model_name: str) -> ReActAgent:
    return ReActAgent(
        name="DeploymentExpert",
        sys_prompt=
        """
        You are the deployment expert responsible for building and deploying AI model function-services on Kubernetes.
        """,
        model=DashScopeChatModel(
            model_name=agent_model_name,
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=True,
        ),
        # Use multiagent formatter because the multiple entities will
        # occur in the prompt of the LLM API call
        formatter=DashScopeMultiAgentFormatter(),
    )

def dependency_check_agent(agent_model_name: str, toolkit: Toolkit) -> ReActAgent:
    # TODO: maybe existing tools can be reused (e.g., ruff)
    """
    Something we can use:
    `ruff check app.py --select F821`
    `ruff check . --fix`
    Use ruff to check for missing imports and fix them.
    """
    return

def create_issue_solver_agent(agent_model_name: str, toolkit: Toolkit) -> ReActAgent:
    return ReActAgent(
        name="IssueSolver",
        sys_prompt=
        """
        You are an expert software engineer specialized in solving issues during building and deploying AI model function-services on Kubernetes.
        """,
        model=DashScopeChatModel(
            model_name=agent_model_name,
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=True,
        ),
        memory=InMemoryMemory(),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
    )

