import json
from typing import Dict, List, Union, Optional
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
import argparse
import os

def _load_gitignore_patterns(root_path: str) -> List[str]:
    """Load patterns from .gitignore file if it exists."""
    gitignore_path = os.path.join(root_path, '.gitignore')
    patterns = ['.git']  # Always exclude .git directory
    if os.path.isfile(gitignore_path):
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        patterns.append(line)
        except Exception:
            pass
    return patterns


def _should_ignore(name: str, is_dir: bool, patterns: List[str]) -> bool:
    """Check if a file/directory should be ignored based on gitignore patterns.
    
    This is a simplified gitignore matcher. For full gitignore spec compliance,
    consider using the 'pathspec' library.
    """
    for pattern in patterns:
        # Handle directory-only patterns (ending with /)
        if pattern.endswith('/'):
            pattern = pattern[:-1]  # Remove trailing slash
            if is_dir and (name == pattern or name.startswith(pattern + '/')):
                return True
        # Handle exact matches and wildcard patterns
        elif pattern.startswith('*'):
            # Simple wildcard: *.pyc matches any .pyc file
            suffix = pattern[1:]
            if name.endswith(suffix):
                return True
        elif pattern.startswith('/'):
            # Root-relative pattern
            if name == pattern[1:]:
                return True
        else:
            # Match directory name or file name exactly
            if name == pattern:
                return True
            # For directories, also match if name starts with pattern/
            if is_dir and name.startswith(pattern + '/'):
                return True
    return False

def list_files_hierarchy(
    root_path: str,
    max_depth: Optional[int] = None,
    respect_gitignore: bool = True,
    _current_depth: int = 0,
    _gitignore_patterns: Optional[List[str]] = None,
) -> Union[Dict[str, "list_files_hierarchy"], List[str]]:
    """
    Recursively list files and directories starting at `root_path`, preserving hierarchy.

    Returns a nested dict where directory names map to their contents. Files in a
    directory are stored in a list under the special key "_files".

    Args:
        root_path: Path to start listing from.
        max_depth: Optional maximum recursion depth (None = unlimited). The root
            directory is at depth 0.
        respect_gitignore: If True, excludes files/directories matching .gitignore patterns.

    Example return shape:
    {
        "subdir": {
            "nested": {"_files": ["a.txt"]},
            "_files": ["b.py"]
        },
        "_files": ["rootfile.md"]
    }
    """
    # Load gitignore patterns on first call
    if _current_depth == 0 and respect_gitignore and _gitignore_patterns is None:
        _gitignore_patterns = _load_gitignore_patterns(root_path)
    
    result: Dict[str, Union[Dict, List[str]]] = {}
    try:
        entries = sorted(os.listdir(root_path))
    except Exception as e:
        return {"_error": str(e)}

    files: List[str] = []

    for entry in entries:
        full = os.path.join(root_path, entry)
        is_dir = os.path.isdir(full)
        
        # Check if this entry should be ignored
        if respect_gitignore and _gitignore_patterns:
            if _should_ignore(entry, is_dir, _gitignore_patterns):
                continue
        
        if is_dir:
            # Directory: recurse if allowed
            if max_depth is None or _current_depth < (max_depth - 1 if max_depth is not None else _current_depth):
                result[entry] = list_files_hierarchy(
                    full, max_depth, respect_gitignore, _current_depth + 1, _gitignore_patterns
                )
            else:
                # Reached max depth: include directory name with empty contents
                result[entry] = {"_files": []}
        else:
            files.append(entry)

    if files:
        result["_files"] = files

    return result


def print_file_tree(root_path: str, max_depth: Optional[int] = None, respect_gitignore: bool = True) -> None:
    """Print a tree-like view of files/directories starting at root_path.

    This is a convenience wrapper around `list_files_hierarchy`.
    
    Args:
        root_path: Path to start listing from.
        max_depth: Optional maximum recursion depth.
        respect_gitignore: If True, excludes files/directories matching .gitignore patterns.
    """
    def _print_node(node: Union[Dict, List[str]], prefix: str = ""):
        if isinstance(node, dict):
            # print files first under this node
            files = node.get("_files", [])
            for i, f in enumerate(files):
                print(f"{prefix}{f}")
            for name, child in sorted(node.items()):
                if name == "_files":
                    continue
                print(f"{prefix}{name}/")
                _print_node(child, prefix + "    ")
        elif isinstance(node, list):
            for f in node:
                print(f"{prefix}{f}")

    tree = list_files_hierarchy(root_path, max_depth, respect_gitignore)
    # save the tree to a json file
    if isinstance(tree, dict) and tree.get("_error"):
        print(f"Error listing '{root_path}': {tree['__error'] if '__error' in tree else tree.get('_error')}")
        return
    print(f"{root_path}:")
    _print_node(tree, prefix="    ")

def load_file_tree(root_path: str, max_depth: Optional[int] = None, respect_gitignore: bool = True) -> ToolResponse:
    """Load the file tree as a nested dictionary."""
    tree = list_files_hierarchy(root_path, max_depth, respect_gitignore)
    try:
        return ToolResponse(
            content=[
                TextBlock(
                    type = "text",
                    text = f"File tree for '{root_path}':\n{json.dumps(tree, indent=2)}"
                ),
            ],
        )
    except Exception as e:
        return ToolResponse(
            content=[
                TextBlock(
                    type = "text",
                    text = f"Error loading file tree for '{root_path}': {str(e)}"
                ),
            ],
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print a tree-like view of files/directories starting at root_path.")
    parser.add_argument("path", type=str, nargs="?", default=".", help="Root path to list files from.")
    parser.add_argument("--max-depth", type=int, help="Maximum recursion depth for listing files.")
    parser.add_argument("--no-gitignore", action="store_true", help="Don't respect .gitignore patterns.")
    args = parser.parse_args()
    path = args.path
    max_depth = args.max_depth
    respect_gitignore = not args.no_gitignore

    print_file_tree(path, max_depth, respect_gitignore)