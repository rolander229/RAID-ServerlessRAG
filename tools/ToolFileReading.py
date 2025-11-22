from typing import Dict, List, Union, Optional
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
import argparse
import os

def read_file(file_path: str) -> ToolResponse:
    """ Read a file and return its contents
    Args:
        file_path (`str`):
            absolute path to the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return ToolResponse(
            content=[
                TextBlock(
                    type = "text",
                    text = content
                ),
            ],
        )
    except Exception as e:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=f"Error reading file '{file_path}': {str(e)}"
                ),
            ]
        )