import json
from typing import Dict, List, Union, Optional
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
import argparse
import os
from tools.ToolFileList import list_files_hierarchy
import subprocess

def py_ruff_fix(root_path: str) -> ToolResponse:
    """ Use ruff to check and try to fix the lint, dependency issues in the given root path
    Args:
        root_path (`str`):
            The root path of the project to check and fix.
    """
    try:
        # Run ruff fix command
        result = subprocess.run(
            ["ruff", "check", root_path, "--fix"],
            capture_output=True,
            text=True
        )
        print(result.stdout)

        if result.returncode == 0:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=f"Ruff fix completed successfully for project at '{root_path}'."
                    ),
                ],
            )
        else:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=f"Ruff fix encountered issues:\n{result.stdout}\n{result.stderr}"
                    ),
                ],
            )
    except Exception as e:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=f"An error occurred while running ruff fix: {e}"
                ),
            ],
        )



def dependency_check_fix_per_file(root_path: str, max_depth: Optional[int] = None, respect_gitignore: bool = True) -> ToolResponse:
    """
    Use ruff to check for missing imports and fix them.
    """
    file_hierarchy = list_files_hierarchy(
        root_path=root_path,
        max_depth=max_depth,
        respect_gitignore=respect_gitignore,
    )
    # filtering only .py files with hierarchy, should also save them with file path and corresponding names

    def extract_py_files(hierarchy: Dict[str, Union[Dict, str]], current_path: str = "") -> Dict[str, str]:
        py_files_list = []
        for name, content in hierarchy.items():
            if name == "_files":
                for f in content:
                    if f.endswith(".py"):
                        # use abs path as key, filename as value
                        full_path = os.path.join(current_path, f)
                        py_files_list.append(full_path)
            elif isinstance(content, dict):
                # Recurse into subdirectory
                sub_path = os.path.join(current_path, name)
                sub_py_files = extract_py_files(content, sub_path)
                py_files_list.extend(sub_py_files)
            else:
                # It's a file, check if it's a .py file
                if name.endswith(".py"):
                     full_path = os.path.join(current_path, name)
                     py_files_list.append(full_path)

        return py_files_list    

    py_path_files_list = extract_py_files(file_hierarchy)
    py_files = [os.path.join(root_path, f) for f in py_path_files_list]

    # use ruff to check and fix the files
    py_ruff_check = {}
    for py_file in py_files:
        try:
            # First, check for issues
            check_result = subprocess.run(
                ["ruff", "check", py_file, "--select", "F821"],
                capture_output=True,
                text=True
            )
            # save the check result
            if check_result.returncode != 0:
                py_ruff_check[py_file] = check_result.stdout
        except Exception as e:
            raise RuntimeError(f"Failed to run ruff check on {py_file}: {e}")
    
        
    issue_text = ""
    segment_line = "=" * 120
    for idx, (file, issues) in enumerate(py_ruff_check.items()):
        issue_text += f"Issue {idx + 1}: {issues}"
        issue_text += segment_line
    return ToolResponse(
            content=[
                TextBlock(
                    type = "text",
                    text = issue_text
                ),
            ],
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dependency Check and Fix Tool")
    parser.add_argument("--root_path", type=str, required=False, default=None, help="Root path to check for dependencies")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth to traverse")
    parser.add_argument("--respect_gitignore", type=bool, default=False, help="Respect .gitignore patterns")

    args = parser.parse_args()
    args.root_path = "/.tmp_cache/.pipes/saliency_service_bk_1122"


    # dependency_check_fix_per_file(
    #     root_path=args.root_path,
    #     max_depth=args.max_depth,
    #     respect_gitignore=args.respect_gitignore
    # )

    args.root_path = "/Users/dinghaoxue/Projects/hello-agentscope/.ml_pipes/saliency_model"
    py_ruff_fix(args.root_path)

    # print("Success:", result.success)
    # print("Message:", result.message.content)