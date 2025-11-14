# TODO: Static Graph Analysis Tool for Python Deep Learning Projects
## Python Function Call Graph Tools: pyan3
## Module Dependency Analysis Tools: pydeps
## Unused Code Tools(Dead Code Detection): vulture
## Custom Analysis: AST

"""
Call Graph Analysis Tool for Python Projects
GPT:
https://chatgpt.com/share/69158c6b-9098-8002-b482-57a8d8237108

Claude:
https://claude.ai/share/7c95a9fe-d9c8-4f0b-b86f-acac240ebc06
"""
import os
import code2flow
from typing import Dict, List, Union, Optional
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
LANG="py"


def analyze_code_graph(input_path: str, output_file: Optional[str]) -> ToolResponse:
    """Analyze the function call graph of a Python file and save it as a DOT file.

    Args:
        input_file: Path to the input Python file.
        output_file: Path to save the output DOT file.
    """
    # input path can be path or a file, check the legality
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")
    if not (os.path.isfile(input_path) or os.path.isdir(input_path)):
        raise ValueError(f"Input path '{input_path}' must be a file or directory.")

    # check output_file must be one of the 'png', 'svg', 'dot', 'gv', 'json'
    if output_file is not None:
        if not output_file.endswith(('.png', '.svg', '.dot', '.gv', '.json')):
            raise ValueError("output_file must have one of the following extensions: .png, .svg, .dot, .gv, .json")

    output_json = code2flow.code2flow(input_path, output_file, language=LANG)

    # TODO: fix this
    try:
        return ToolResponse(
            message=TextBlock(f"Code graph analysis completed. Output saved to '{output_file}'."),
            data={"output_file": output_file, "output_json": output_json}
        )
    except Exception as e:
        return ToolResponse(
            message=TextBlock(f"Code graph analysis failed: {str(e)}"),
            data={}
        )
