import ast
import json
import os
from pathlib import Path
from typing import Optional, List, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

try:
    import yaml as _yaml

    class _CFNLoader(_yaml.SafeLoader):
        """SafeLoader extended with CloudFormation/SAM intrinsic function tags."""
        pass

    _CFN_TAGS = [
        "!Ref", "!Sub", "!GetAtt", "!Join", "!Select", "!Split",
        "!ImportValue", "!FindInMap", "!GetAZs", "!Cidr",
        "!If", "!Not", "!Equals", "!And", "!Or", "!Condition",
        "!Transform", "!Base64",
    ]
    _multi_ctor = lambda loader, tag_suffix, node: node.value if isinstance(node, _yaml.ScalarNode) else loader.construct_sequence(node) if isinstance(node, _yaml.SequenceNode) else loader.construct_mapping(node)
    for _tag in _CFN_TAGS:
        _CFNLoader.add_constructor(_tag, lambda loader, node, t=_tag: node.value if isinstance(node, _yaml.ScalarNode) else loader.construct_sequence(node) if isinstance(node, _yaml.SequenceNode) else loader.construct_mapping(node))
    _CFNLoader.add_multi_constructor("!", _multi_ctor)

except ImportError:
    _yaml = None
    _CFNLoader = None

_SYNTAX_CHECK_SUFFIXES = {".py", ".json", ".yaml", ".yml"}

# --- Input Schemas ---

class ReadFileInput(BaseModel):
    """Input schema for ReadFileTool."""
    path: str = Field(..., description="Relative path to the file to read (e.g., 'src/main.py').")
    start_line: Optional[int] = Field(default=None, description="1-based start line number. If omitted, reads from the beginning.")
    end_line: Optional[int] = Field(default=None, description="1-based end line number (inclusive). If omitted, reads to the end.")

class WriteFileInput(BaseModel):
    """Input schema for WriteFileTool."""
    path: str = Field(..., description="Relative path to the file to write.")
    content: str = Field(..., description="The content to write to the file.")
    append: bool = Field(default=False, description="If True, append to existing file instead of overwriting.")
    json_key: Optional[str] = Field(default=None, description=(
        "For .json files only. When set, merges content into the JSON file at this top-level key. "
        "content must be a valid JSON value string. "
        "If the key already exists and BOTH the existing value and content are arrays, "
        "the arrays are concatenated (append). Otherwise the key is overwritten. "
        "The file is created with {} if it does not exist. "
        "Use this to build large JSON files section by section (keep each content under ~4000 chars)."
    ))

class FileListInput(BaseModel):
    """Input schema for FileListTool."""
    path: str = Field(default=".", description="Relative path to list files from (defaults to root).")
    recursive: bool = Field(default=False, description="If True, list files recursively.")

# --- Tools ---

_LARGE_FILE_LINE_THRESHOLD = 500

class ReadFileTool(BaseTool):
    name: str = "ReadFileTool"
    description: str = (
        "Read content from a file. Returns the full file by default. "
        "For large files (>500 lines), only the first 500 lines are returned "
        "with a warning — use start_line/end_line to page through the rest. "
        "(1-based, inclusive). The output always includes the total line count."
    )
    args_schema: Type[BaseModel] = ReadFileInput
    root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])

    def __init__(self, root: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        if root:
            self.root = root

    def _run(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        try:
            target = (self.root / path).resolve()
            if not target.exists():
                return f"Error: File not found: {path}"
            with target.open("r", encoding="utf-8", errors="ignore") as f:
                all_lines = f.readlines()
            total = len(all_lines)

            if start_line is not None or end_line is not None:
                s = max((start_line or 1) - 1, 0)
                e = min(end_line or total, total)
                selected = all_lines[s:e]
                header = f"[Lines {s+1}-{e} of {total} total lines in {path}]\n"
                return header + "".join(selected)

            if total > _LARGE_FILE_LINE_THRESHOLD:
                selected = all_lines[:_LARGE_FILE_LINE_THRESHOLD]
                return (
                    f"[Lines 1-{_LARGE_FILE_LINE_THRESHOLD} of {total} total lines in {path}]\n"
                    + "".join(selected)
                    + f"\n\n⚠️  FILE TRUNCATED — only the first {_LARGE_FILE_LINE_THRESHOLD} of "
                    f"{total} lines are shown. Use start_line/end_line to read the rest "
                    f"(e.g., start_line={_LARGE_FILE_LINE_THRESHOLD + 1}, end_line={min(total, _LARGE_FILE_LINE_THRESHOLD * 2)})."
                )

            return f"[{total} total lines in {path}]\n" + "".join(all_lines)
        except Exception as e:
            return f"Error reading file: {str(e)}"

class WriteFileTool(BaseTool):
    name: str = "WriteFileTool"
    description: str = (
        "Write content to a file at the given path. Creates directories if needed. "
        "Set append=True to append to an existing file instead of overwriting. "
        "For large JSON files: use json_key to merge content into a specific top-level key "
        "(avoids truncation by writing one section at a time; auto-appends if both are arrays)."
    )
    args_schema: Type[BaseModel] = WriteFileInput
    root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])

    def __init__(self, root: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        if root:
            self.root = root

    @staticmethod
    def _validate_content(suffix: str, content: str) -> str | None:
        """Return an error message if *content* is invalid for the given suffix, else None."""
        if suffix == ".json":
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                return f"Invalid JSON: {e}"
        elif suffix == ".py":
            try:
                ast.parse(content)
            except SyntaxError as e:
                return f"Invalid Python syntax: {e}"
        elif suffix in {".yaml", ".yml"} and _yaml is not None:
            try:
                _yaml.load(content, Loader=_CFNLoader)
            except _yaml.YAMLError as e:
                return f"Invalid YAML syntax: {e}"
        return None

    @staticmethod
    def _detect_overlap(existing: str, new_content: str, max_check_lines: int = 80) -> int:
        """Detect duplicate overlap between end of existing file and start of new content.

        Returns the number of characters to skip from the beginning of *new_content*
        to avoid duplication.  Only triggers when >= 2 matching lines are found so
        that trivial single-line matches (e.g. a lone ``}``) don't cause false
        positives.
        """
        if not existing or not new_content:
            return 0

        existing_lines = existing.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        if len(existing_lines) < 2 or len(new_lines) < 2:
            return 0

        max_overlap = min(len(existing_lines), len(new_lines), max_check_lines)

        for overlap_size in range(max_overlap, 1, -1):  # minimum 2 lines
            if existing_lines[-overlap_size:] == new_lines[:overlap_size]:
                return sum(len(line) for line in new_lines[:overlap_size])

        return 0

    def _run_json_merge(self, target: Path, path: str, content: str, json_key: str) -> str:
        """Merge content into a JSON file at a specific top-level key."""
        try:
            parsed_value = json.loads(content)
        except json.JSONDecodeError as e:
            return f"Error: content is not valid JSON: {e}"

        if target.exists():
            raw = target.read_text(encoding="utf-8", errors="ignore")
            try:
                data = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

        if not isinstance(data, dict):
            data = {}

        if json_key in data and isinstance(data[json_key], list) and isinstance(parsed_value, list):
            data[json_key].extend(parsed_value)
            action = "appended to"
        else:
            data[json_key] = parsed_value
            action = "set"

        output = json.dumps(data, ensure_ascii=False)
        with target.open("w", encoding="utf-8") as f:
            f.write(output)
            f.flush()
            os.fsync(f.fileno())

        written = target.read_text(encoding="utf-8")
        try:
            json.loads(written)
        except json.JSONDecodeError as e:
            return f"CRITICAL: File failed JSON validation after merge: {e}"

        summary = self._json_summary(data)
        return f"OK — {action} key '{json_key}'. File: {len(written)} chars. Keys: {summary}"

    @staticmethod
    def _json_summary(data: dict) -> str:
        parts = []
        for k, v in data.items():
            if isinstance(v, list):
                parts.append(f"{k}([{len(v)} items])")
            elif isinstance(v, dict):
                parts.append(f"{k}({{...}})")
            else:
                parts.append(f"{k}={v!r}")
        return ", ".join(parts) if parts else "(empty)"

    def _run(self, path: str, content: str, append: bool = False, json_key: Optional[str] = None) -> str:
        try:
            target = (self.root / path).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)

            if json_key is not None:
                return self._run_json_merge(target, path, content, json_key)

            dedup_info = ""
            if append and target.exists():
                existing = target.read_text(encoding="utf-8", errors="ignore")
                skip_chars = self._detect_overlap(existing, content)
                if skip_chars > 0:
                    original_len = len(content)
                    content = content[skip_chars:]
                    skipped_lines = original_len - len(content)
                    dedup_info = (
                        f" [DEDUP: removed {skip_chars} overlapping chars "
                        f"from the beginning of your append content]"
                    )
                    if not content.strip():
                        return (
                            f"SKIPPED — all {original_len} chars you tried to append "
                            f"are duplicates of content already at the end of {path}. "
                            f"No changes made. Use ReadFileTool to verify the file."
                        )

            mode = "a" if append else "w"
            with target.open(mode, encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())

            written = target.read_text(encoding="utf-8")
            if not append and len(written) != len(content):
                return (
                    f"WRITE LENGTH MISMATCH — expected {len(content)} chars "
                    f"but on-disk file has {len(written)} chars. "
                    f"The file may be corrupted. You MUST re-write it."
                )

            if target.suffix in _SYNTAX_CHECK_SUFFIXES:
                err = self._validate_content(target.suffix, written)
                if err:
                    if append:
                        return (
                            f"WARNING — content was appended but the resulting file "
                            f"has syntax errors: {err}\n"
                            f"Total file: {len(written)} chars. "
                            f"This often means duplicate or malformed content was appended. "
                            f"Use ReadFileTool to inspect the FULL file, then re-write "
                            f"the complete corrected file with append=False."
                        )
                    return (
                        f"WARNING — file was written but may be incomplete: {err}\n"
                        f"Written length: {len(written)} chars. "
                        f"Use ReadFileTool to verify the current file content, then "
                        f"re-write the complete file with append=False if needed."
                    )

            action = "appended to" if append else "wrote to"
            line_count = written.count("\n") + (0 if written.endswith("\n") else 1)
            return f"Successfully {action} {path} ({line_count} total lines, {len(written)} chars){dedup_info}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

class FileListTool(BaseTool):
    name: str = "FileListTool"
    description: str = "List files and directories at the given path. Useful for exploring file structure."
    args_schema: Type[BaseModel] = FileListInput
    root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])

    def __init__(self, root: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        if root:
            self.root = root

    def _run(self, path: str = ".", recursive: bool = False) -> List[str]:
        target = (self.root / path).resolve()
        
        if not target.exists():
            return [f"Error: Path does not exist: {path}"]
        
        if not target.is_dir():
            return [f"Error: Path is not a directory: {path}"]
        
        try:
            if recursive:
                files = []
                for p in target.rglob('*'):
                    if p.is_file():
                        rel_path = p.relative_to(target)
                        files.append(str(rel_path))
                return sorted(files)
            else:
                items = []
                for p in target.iterdir():
                    name = p.name
                    if p.is_dir():
                        name += "/"
                    items.append(name)
                return sorted(items)
        except Exception as e:
            return [f"Error listing directory: {str(e)}"]
