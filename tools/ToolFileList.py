import asyncio
import inspect
import json
from typing import Any, AsyncGenerator

from pydantic import BaseModel, Field

import agentscope
from agentscope.message import TextBlock, ToolUseBlock
from agentscope.tool import ToolResponse, Toolkit, execute_python_code

import os

def list_files_in_directory(directory: str) -> list[str]:
    """List all files in the given directory."""
    try:
        files = os.listdir(directory)
        return [f for f in files if os.path.isfile(os.path.join(directory, f))]
    except Exception as e:
        return f"Error: {str(e)}"