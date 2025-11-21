import json
from typing import Dict, List, Union, Optional
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
import argparse
import os

def model_refactor_validate(output_path: str) -> ToolResponse:
    """
    Validate that the refactored model folder at `output_path` contains:
    - Dockerfile
    - requirements.txt
    - at least one .py file
    Returns a ToolResponse indicating success and a descriptive message.
    """
    if not os.path.exists(output_path) or not os.path.isdir(output_path):
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=f"Error: The specified output path '{output_path}' does not exist or is not a directory."
                ),
            ]
        )

    entries = os.listdir(output_path)
    lower_entries = {name.lower() for name in entries}

    missing = []
    if "dockerfile" not in lower_entries:
        missing.append("Dockerfile")
    if "requirements.txt" not in lower_entries:
        missing.append("requirements.txt")
    if not any(name.lower().endswith(".py") for name in entries):
        missing.append("*.py")

    if missing:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=f"Validation failed for `{output_path}`: missing required files: {', '.join(missing)}."
                ),
            ]
        )

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=f"Validation succeeded for `{output_path}`: all required files are present."
            ),
        ]
    )
