import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import ast


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MONOLITH_DIR = ROOT_DIR / "input_monolith"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "storage" / "analysis_report.json"


PY_ENTRY_METHOD_DECORATORS = {"get", "post", "put", "delete", "patch"}
# Directories to skip during project-structure traversal and code analysis.
IGNORE_DIRS = {
    'venv',
    'env',
    '.venv',
    '__pycache__',
    'node_modules',
    '.git',
    '.idea',
    '.vscode',
    'dist',
    'build',
    '.pytest_cache',
    '.mypy_cache',
    'htmlcov',
    '.tox',
    'eggs',
    '.eggs',
    'coverage'
}


def build_project_structure(root: Path) -> str:
    lines: List[str] = []

    def _walk(current: Path, prefix: str = "") -> None:
        # Filter hidden entries and ignored directories.
        entries = sorted([
            p for p in current.iterdir() 
            if not p.name.startswith(".") and p.name not in IGNORE_DIRS
        ])
        for idx, entry in enumerate(entries):
            connector = "└── " if idx == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if idx == len(entries) - 1 else "│   "
                _walk(entry, prefix + extension)

    lines.append(root.name + "/")
    if root.exists():
        _walk(root)
    return "\n".join(lines)



def tag_file(source: str, file_path: str) -> List[str]:
    tags: Set[str] = set()
    lower = source.lower()
    # Backend / general feature detection.

    # AWS SDK detection.
    if "boto3" in lower:
        tags.add("AWS_SDK")

    # DynamoDB usage detection.
    if any(keyword in lower for keyword in [
        "dynamodb", "dynamo", "putitem", "getitem", "updateitem", "deleteitem", 
        ".table(", "batchwriteitem"
    ]):
        tags.add("DynamoDB")
    
    # Authentication detection.
    if "jwt" in lower or "jsonwebtoken" in lower:
        tags.add("Auth")

    # File-upload detection.
    file_upload_keywords = [
        # Python Flask/FastAPI patterns
        "request.files", "fileupload", "file.save(", "werkzeug.datastructures.filestorage",
        "uploadfile", "fastapi.uploadfile", "form.file", "multipart/form-data","input_file","output_file"
        
        # Express.js/Node.js patterns
        "multer", "formidable", "busboy", "multiparty", "express-fileupload",
        "req.files", "req.file", "upload.single", "upload.array", "upload.fields",
        
        # Generic file upload patterns
        "file upload", "upload file", "image upload", "avatar upload", "document upload",
        "file_upload", "upload_file", "image_upload", "avatar_upload",
        
        # File storage/processing
        "send_file", "send_from_directory", "static/uploads", "/uploads/",
        "fs.writefile", "fs.createwritestream", "pipe(fs.createwritestream",
        
        # HTTP methods commonly used for file upload
        "enctype=\"multipart/form-data\"", "enctype='multipart/form-data'",
        
        # S3 upload patterns
        "s3.upload", "s3.putobject", "upload_to_s3", "uploadtos3",
        "boto3.client('s3')", "boto3.resource('s3')",
        
        # Common file extensions in upload context
        ".save(", ".upload(", "allowed_extensions", "allowed_files",
        "file_allowed", "check_file_extension"
    ]
    
    if any(keyword in lower for keyword in file_upload_keywords):
        tags.add("FileUpload")
    
    return sorted(tags)


def module_name_from_path(root: Path, file_path: Path, app_name: str = None) -> str:
    """
    Derive a dotted module name from a file path.

    Args:
        root: Application root directory.
        file_path: Absolute path to the file.
        app_name: Optional app name prepended as a namespace prefix for global uniqueness.

    Returns:
        Module name string, e.g. "shopping-cart.app" or "app".
    """
    rel = file_path.relative_to(root)
    parts = rel.with_suffix("").parts

    # Prepend app_name to ensure globally unique symbol IDs.
    if app_name:
        return ".".join([app_name] + list(parts))
    else:
        return ".".join(parts)


def analyze_python_file(root: Path, file_path: Path, app_name: str = None) -> Dict[str, Any]:
    """
    Analyze a Python source file and extract entry points, symbols, and call edges.

    Args:
        root: Application root directory.
        file_path: Absolute path to the file.
        app_name: Optional app name used as a namespace prefix in the symbol table.
    """
    rel_path = str(file_path.relative_to(root).as_posix())

    # Prefix with app_name for global uniqueness.
    if app_name:
        rel_path = f"{app_name}/{rel_path}"
    
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    tags = tag_file(source, rel_path)
    dependency_targets: Set[str] = set()
    entry_points: List[Dict[str, Any]] = []
    symbol_table: List[Dict[str, Any]] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {
            "rel_path": rel_path,
            "tags": tags,
            "dependencies": [],
            "entry_points": [],
            "symbols": [],
        }

    module_name = module_name_from_path(root, file_path, app_name)

    # Extract Flask Blueprint url_prefix (e.g. bp = Blueprint('name', __name__, url_prefix='/api/orders'))
    blueprint_prefix = ""
    blueprint_var_names: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Call) and
                isinstance(node.value.func, ast.Name) and
                node.value.func.id == "Blueprint"):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                blueprint_var_names.add(target.id)
        for kw in node.value.keywords:
            if kw.arg == "url_prefix" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                blueprint_prefix = kw.value.value
                break
        if not blueprint_prefix and len(node.value.args) >= 3:
            arg2 = node.value.args[2]
            if isinstance(arg2, ast.Constant) and isinstance(arg2.value, str):
                blueprint_prefix = arg2.value

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:  # type: ignore[override]
            for alias in node.names:
                dependency_targets.add(alias.name)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # type: ignore[override]
            if node.module:
                dependency_targets.add(node.module)

        def _record_function(self, func: ast.AST, parent_class: str | None = None) -> None:
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return
            start = getattr(func, "lineno", None)
            end = getattr(func, "end_lineno", start)
            if parent_class:
                symbol_id = f"{module_name}.{parent_class}.{func.name}"
            else:
                symbol_id = f"{module_name}.{func.name}"
            symbol_table.append(
                {
                    "id": symbol_id,
                    "file_path": rel_path,
                    "start_line": start,
                    "end_line": end,
                }
            )

            # Flask / FastAPI style decorators
            for dec in getattr(func, "decorator_list", []):
                if not isinstance(dec, ast.Call):
                    continue
                # app.get("/path") / app.post("/path") / app.route("/path", methods=[...])
                if isinstance(dec.func, ast.Attribute) and isinstance(dec.func.value, ast.Name):
                    attr = dec.func.attr.lower()
                    if attr == "route":
                        http_method = "GET"
                        path = None
                        if dec.args:
                            arg0 = dec.args[0]
                            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                                path = arg0.value
                        for kw in dec.keywords:
                            if kw.arg == "methods" and isinstance(kw.value, (ast.List, ast.Tuple)):
                                for elt in kw.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        http_method = elt.value.upper()
                        if path is not None:
                            full_path = blueprint_prefix.rstrip("/") + "/" + path.lstrip("/") if path else blueprint_prefix or "/"
                            entry_points.append(
                                {
                                    "file": rel_path,
                                    "method": http_method,
                                    "path": full_path,
                                    "handler_function": func.name,
                                }
                            )
                    elif attr in PY_ENTRY_METHOD_DECORATORS:
                        # FastAPI style: @app.get("/items")
                        path = None
                        if dec.args:
                            arg0 = dec.args[0]
                            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                                path = arg0.value
                        if path is not None:
                            full_path = blueprint_prefix.rstrip("/") + "/" + path.lstrip("/") if path else blueprint_prefix or "/"
                            entry_points.append(
                                {
                                    "file": rel_path,
                                    "method": attr.upper(),
                                    "path": full_path,
                                    "handler_function": func.name,
                                }
                            )

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
            self._record_function(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
            self._record_function(node)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[override]
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", start)
            class_id = f"{module_name}.{node.name}"
            symbol_table.append(
                {
                    "id": class_id,
                    "file_path": rel_path,
                    "start_line": start,
                    "end_line": end,
                }
            )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._record_function(child, parent_class=node.name)
            self.generic_visit(node)

    Visitor().visit(tree)

    return {
        "rel_path": rel_path,
        "tags": tags,
        "dependencies": sorted(dependency_targets),
        "entry_points": entry_points,
        "symbols": symbol_table,
    }


# ============================================================
# Cross-file Call Graph Analysis
# ============================================================

def extract_python_call_graph(
    root: Path, file_path: Path, source: str, app_name: str = None
) -> List[Dict[str, Any]]:
    """
    Extract cross-file function call edges from a Python file via AST.

    Strategy:
      1. Build import map: local_name -> module_path (from import / from-import)
      2. Filter to project-local modules (file exists on disk)
      3. Walk each FunctionDef body and record ast.Call nodes that reference
         an imported local name (obj.method() or func())

    Returns list of edges, each:
      { caller_file, caller_function, caller_line, callee_module, callee_symbol }
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    # Annotate parent references for return-value usage analysis
    for _node in ast.walk(tree):
        for _child in ast.iter_child_nodes(_node):
            _child._parent = _node  # type: ignore[attr-defined]

    rel_path = str(file_path.relative_to(root).as_posix())
    if app_name:
        rel_path = f"{app_name}/{rel_path}"

    # Pass 1 — import map
    import_map: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                import_map[alias.asname or alias.name] = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                import_map[alias.asname or alias.name] = alias.name

    # Keep only project-local modules (resolvable to a file on disk)
    local_import_map: Dict[str, str] = {}
    for local_name, mod_path in import_map.items():
        rel_parts = mod_path.replace(".", os.sep)
        candidates = [
            root / (rel_parts + ".py"),
            root / rel_parts / "__init__.py",
        ]
        if any(c.exists() for c in candidates):
            local_import_map[local_name] = mod_path

    if not local_import_map:
        return []

    # Pass 2 — walk functions, collect cross-file call edges
    # Dict-based dedup: same (caller, module, symbol) merges with OR on boolean fields
    edge_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        caller_func = node.name

        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue

            callee_module: Optional[str] = None
            callee_symbol: Optional[str] = None

            # obj.method()  e.g. CatalogService.reserve_flight_seat()
            if (isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)):
                obj = child.func.value.id
                if obj in local_import_map:
                    callee_module = local_import_map[obj]
                    callee_symbol = f"{obj}.{child.func.attr}"

            # func()  e.g. some_imported_function()
            elif isinstance(child.func, ast.Name):
                name = child.func.id
                if name in local_import_map:
                    callee_module = local_import_map[name]
                    callee_symbol = name

            if callee_module and callee_symbol:
                # Determine return-value usage from AST parent chain
                is_awaited = False
                parent = getattr(child, '_parent', None)
                if isinstance(parent, ast.Await):
                    is_awaited = True
                    parent = getattr(parent, '_parent', None)
                # ast.Expr = expression used as statement → return value discarded
                return_value_used = not isinstance(parent, ast.Expr)

                key = (caller_func, callee_module, callee_symbol)
                if key not in edge_map:
                    edge_map[key] = {
                        "caller_file": rel_path,
                        "caller_function": caller_func,
                        "caller_line": getattr(child, "lineno", node.lineno),
                        "callee_module": callee_module,
                        "callee_symbol": callee_symbol,
                        "return_value_used": return_value_used,
                        "is_awaited": is_awaited,
                    }
                else:
                    if return_value_used:
                        edge_map[key]["return_value_used"] = True
                    if is_awaited:
                        edge_map[key]["is_awaited"] = True

    return list(edge_map.values())


def _find_function_end_js(lines: List[str], start_idx: int) -> int:
    """
    Find the closing brace of a JS/TS function via brace counting, starting from start_idx.

    Args:
        lines: Source code split into lines.
        start_idx: 0-based index of the function's opening line.

    Returns:
        1-based line number of the closing brace.
    """
    brace_count = 0
    started = False

    for idx in range(start_idx, len(lines)):
        line = lines[idx]

        # Strip string literals to avoid counting braces inside strings.
        line_cleaned = re.sub(r"'[^']*'", "", line)
        line_cleaned = re.sub(r'"[^"]*"', "", line_cleaned)
        line_cleaned = re.sub(r"`[^`]*`", "", line_cleaned)

        for char in line_cleaned:
            if char == '{':
                brace_count += 1
                started = True
            elif char == '}':
                brace_count -= 1
                if started and brace_count == 0:
                    return idx + 1  # return 1-based line number

        # Guard against infinite loops: scan at most 1000 lines.
        if idx - start_idx > 1000:
            break

    # Fallback: if no closing brace found, return start + 1.
    return start_idx + 2


def analyze_js_like_file(root: Path, file_path: Path, app_name: str = None) -> Dict[str, Any]:
    """
    Analyze a JavaScript/TypeScript source file and extract entry points, symbols, and call edges.

    Args:
        root: Application root directory.
        file_path: Absolute path to the file.
        app_name: Optional app name used as a namespace prefix in the symbol table.
    """
    rel_path = str(file_path.relative_to(root).as_posix())

    # Prefix with app_name for global uniqueness.
    if app_name:
        rel_path = f"{app_name}/{rel_path}"
    
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    tags = tag_file(source, rel_path)
    dependency_targets: Set[str] = set()
    entry_points: List[Dict[str, Any]] = []
    symbol_table: List[Dict[str, Any]] = []
    
    # Extract dependencies via require()/import.
    dep_patterns = [
        re.compile(r"require\(['\"]([^'\"]+)['\"]\)"),
        re.compile(r"from\s+['\"]([^'\"]+)['\"]"),
        re.compile(r"import\s+[^'\"]+['\"]([^'\"]+)['\"]"),
    ]
    for pat in dep_patterns:
        for m in pat.finditer(source):
            dependency_targets.add(m.group(1))

    # Entry points: Express.js app.get/app.post/router.get/...
    route_pattern = re.compile(
        r"\b(app|router)\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*([A-Za-z0-9_$.]+)",
        re.IGNORECASE,
    )
    for m in route_pattern.finditer(source):
        method = m.group(2).upper()
        path = m.group(3)
        handler_name = f"{method}_{path.replace('/', '_').replace(':', '_')}"
        entry_points.append(
            {
                "file": rel_path,
                "method": method,
                "path": path,
                "handler_function": handler_name,
            }
        )

    # Symbol table - enhanced JavaScript function recognition.
    module_name = module_name_from_path(root, file_path, app_name)
    lines = source.splitlines()

    # Extended set of function recognition patterns.
    patterns = {
        # function name() {}
        "function_decl": re.compile(r"^\s*(?:async\s+)?(?:export\s+)?function\s+([A-Za-z0-9_$]+)\s*\("),
        # class Name {}
        "class_decl": re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z0-9_$]+)\b"),
        # const/let/var name = function() {}  or  const name = () => {}
        "const_func": re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>)"
        ),
        # router.get('/path', async (req, res) => {}) - Express route handler
        "router_func": re.compile(
            r"^\s*(?:router|app)\.(get|post|put|delete|patch|use)\s*\(\s*['\"]([^'\"]+)['\"]"
        ),
        # exports.name = function() {}  or  module.exports.name =
        "exports_func": re.compile(
            r"^\s*(?:module\.)?exports\.([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>)"
        ),
        # Object method: methodName() {}  or  async methodName() {}
        "object_method": re.compile(
            r"^\s*(?:async\s+)?([A-Za-z0-9_$]+)\s*\([^)]*\)\s*\{"
        ),
        # WebSocket/EventEmitter listener: .on('event', callback)
        # Matches: wss.on('connection', ...)  ws.on('message', ...)  process.on('SIGINT', ...)
        "event_listener": re.compile(
            r"^\s*([A-Za-z0-9_$]+)\.on\s*\(\s*['\"]([^'\"]+)['\"]"
        ),
    }

    for idx, line in enumerate(lines, start=1):
        name = None
        kind = "function"
        
        # Try each pattern in priority order.
        if m := patterns["function_decl"].search(line):
            name = m.group(1)
        elif m := patterns["class_decl"].search(line):
            name = m.group(1)
            kind = "class"
        elif m := patterns["const_func"].search(line):
            name = m.group(1)
        elif m := patterns["router_func"].search(line):
            # Express route: use method_path as the identifier.
            method = m.group(1).upper()
            path = m.group(2)
            name = f"{method}_{path.replace('/', '_').replace(':', '_')}"
            kind = "route_handler"
        elif m := patterns["event_listener"].search(line):
            # WebSocket/EventEmitter listener, e.g.:
            #   wss.on('connection', ...) -> wss_on_connection
            #   ws.on('message', ...) -> ws_on_message
            obj_name = m.group(1)
            event_name = m.group(2)
            name = f"{obj_name}_on_{event_name}"
            kind = "event_handler"
        elif m := patterns["exports_func"].search(line):
            name = m.group(1)
        elif m := patterns["object_method"].search(line):
            # Object method, but exclude control-flow keywords.
            potential_name = m.group(1)
            if potential_name not in {'if', 'for', 'while', 'switch', 'catch', 'with'}:
                name = potential_name

        if name:
            # Locate the function's closing brace via brace matching.
            end_line = _find_function_end_js(lines, idx - 1)
            
            symbol_id = f"{module_name}.{name}"
            symbol_table.append(
                {
                    "id": symbol_id,
                    "file_path": rel_path,
                    "start_line": idx,
                    "end_line": end_line,
                    "kind": kind,
                }
            )

    return {
        "rel_path": rel_path,
        "tags": tags,
        "dependencies": sorted(dependency_targets),
        "entry_points": entry_points,
        "symbols": symbol_table,
    }


def _find_enclosing_js_function(
    file_symbols: List[Dict[str, Any]], line_no: int
) -> str:
    """Find the tightest enclosing function for *line_no* using symbol table ranges."""
    best_name: Optional[str] = None
    best_span = float("inf")

    for sym in file_symbols:
        start = sym.get("start_line", 0)
        end = sym.get("end_line", 0)
        if start <= line_no <= end:
            span = end - start
            if span < best_span:
                best_span = span
                best_name = sym["id"].rsplit(".", 1)[-1]

    return best_name or "(module-level)"


def extract_js_call_graph(
    root: Path,
    file_path: Path,
    source: str,
    file_symbols: List[Dict[str, Any]],
    app_name: str = None,
) -> List[Dict[str, Any]]:
    """
    Extract cross-file function call edges from a JS/TS file via regex.

    Only tracks calls to locally-imported modules (relative paths starting with '.').
    Uses *file_symbols* (the symbol table entries for this file) to attribute each
    call to its enclosing function / route handler.

    Returns list of edges with the same schema as the Python variant.
    """
    rel_path = str(file_path.relative_to(root).as_posix())
    if app_name:
        rel_path = f"{app_name}/{rel_path}"

    # ---- Build import map (relative paths only → local modules) ----
    import_map: Dict[str, str] = {}

    # const { a, b } = require('./module')
    for m in re.finditer(
        r"(?:const|let|var)\s*\{([^}]+)\}\s*=\s*require\(\s*['\"](\.[^'\"]+)['\"]\s*\)",
        source,
    ):
        names = [
            n.strip().split(":")[-1].strip().split(" ")[-1].strip()
            for n in m.group(1).split(",")
        ]
        module = m.group(2)
        for name in names:
            if name and re.match(r"^[A-Za-z_$]", name):
                import_map[name] = module

    # const name = require('./module')
    for m in re.finditer(
        r"(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*require\(\s*['\"](\.[^'\"]+)['\"]\s*\)",
        source,
    ):
        import_map.setdefault(m.group(1), m.group(2))

    # import { a, b } from './module'
    for m in re.finditer(
        r"import\s*\{([^}]+)\}\s*from\s*['\"](\.[^'\"]+)['\"]", source
    ):
        names = [n.strip().split(" as ")[-1].strip() for n in m.group(1).split(",")]
        module = m.group(2)
        for name in names:
            if name and re.match(r"^[A-Za-z_$]", name):
                import_map.setdefault(name, module)

    # import name from './module'
    for m in re.finditer(
        r"import\s+([A-Za-z_$][A-Za-z0-9_$]*)\s+from\s*['\"](\.[^'\"]+)['\"]",
        source,
    ):
        import_map.setdefault(m.group(1), m.group(2))

    if not import_map:
        return []

    # ---- Scan source for calls to imported local names ----
    lines = source.splitlines()
    # Dict-based dedup: same (caller, module, symbol) merges with OR on boolean fields
    edge_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    # Detect assignment / return patterns in the text preceding a call
    _assign_re = re.compile(
        r'(?:const|let|var)\s+.+=(?!=)\s*(?:await\s*)?$|'
        r'[A-Za-z_$][\w$.]*\s*(?<!=)=(?!=)\s*(?:await\s*)?$|'
        r'\breturn\s*(?:await\s*)?$'
    )

    def _analyze_call_context(line: str, match_start: int) -> Tuple[bool, bool]:
        """Return (is_awaited, return_value_used) by inspecting line prefix."""
        prefix = line[:match_start].rstrip()
        is_awaited = bool(re.search(r'\bawait\b', prefix))
        return_value_used = bool(_assign_re.search(prefix))
        return is_awaited, return_value_used

    for local_name, module_path in import_map.items():
        escaped = re.escape(local_name)

        # name.method(...)
        method_re = re.compile(rf"\b{escaped}\.([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
        # name(...)  — negative lookbehind avoids double-counting obj.name()
        direct_re = re.compile(rf"(?<![.\w]){escaped}\s*\(")

        for line_idx, line in enumerate(lines):
            line_no = line_idx + 1

            for m in method_re.finditer(line):
                callee = f"{local_name}.{m.group(1)}"
                caller = _find_enclosing_js_function(file_symbols, line_no)
                is_awaited, return_value_used = _analyze_call_context(line, m.start())

                key = (caller, module_path, callee)
                if key not in edge_map:
                    edge_map[key] = {
                        "caller_file": rel_path,
                        "caller_function": caller,
                        "caller_line": line_no,
                        "callee_module": module_path,
                        "callee_symbol": callee,
                        "return_value_used": return_value_used,
                        "is_awaited": is_awaited,
                    }
                else:
                    if return_value_used:
                        edge_map[key]["return_value_used"] = True
                    if is_awaited:
                        edge_map[key]["is_awaited"] = True

            for m in direct_re.finditer(line):
                caller = _find_enclosing_js_function(file_symbols, line_no)
                is_awaited, return_value_used = _analyze_call_context(line, m.start())

                key = (caller, module_path, local_name)
                if key not in edge_map:
                    edge_map[key] = {
                        "caller_file": rel_path,
                        "caller_function": caller,
                        "caller_line": line_no,
                        "callee_module": module_path,
                        "callee_symbol": local_name,
                        "return_value_used": return_value_used,
                        "is_awaited": is_awaited,
                    }
                else:
                    if return_value_used:
                        edge_map[key]["return_value_used"] = True
                    if is_awaited:
                        edge_map[key]["is_awaited"] = True

    return list(edge_map.values())


def analyze_project_config(monolith_root: Path) -> Dict[str, Any]:
    """Parse project dependency manifests (requirements.txt and package.json)."""
    config_info = {}

    # Parse Python requirements.txt.
    requirements_file = monolith_root / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                # Skip blank lines and comments.
                if line and not line.startswith('#'):
                    # Parse specifier: Flask==3.0.0 -> {name: Flask, version: 3.0.0}
                    if '==' in line:
                        name, version = line.split('==', 1)
                        requirements.append({"name": name.strip(), "version": version.strip()})
                    elif '>=' in line:
                        name, version = line.split('>=', 1)
                        requirements.append({"name": name.strip(), "version": f">={version.strip()}"})
                    else:
                        requirements.append({"name": line, "version": None})
            if requirements:
                config_info['python_dependencies'] = requirements
    
    # Parse JavaScript/Node.js package.json.
    package_json = monolith_root / "package.json"
    if package_json.exists():
        try:
            with open(package_json, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
                nodejs_info = {}

                # Production dependencies.
                if 'dependencies' in package_data and package_data['dependencies']:
                    nodejs_info['dependencies'] = [
                        {"name": name, "version": version}
                        for name, version in package_data['dependencies'].items()
                    ]

                # Development dependencies.
                if 'devDependencies' in package_data and package_data['devDependencies']:
                    nodejs_info['devDependencies'] = [
                        {"name": name, "version": version}
                        for name, version in package_data['devDependencies'].items()
                    ]

                if nodejs_info:
                    config_info['nodejs_dependencies'] = nodejs_info
        except (json.JSONDecodeError, KeyError):
            pass  # Ignore parse errors.
    
    return config_info


def _prioritize_schema_files(files: List[str]) -> List[str]:
    """
    Filter and rank files by their likelihood of containing DynamoDB table schema definitions.

    Priority tiers:
    - P1 (high): Table-initialization files (init-db, setup-db, create-tables, etc.)
    - P2 (medium): Database config and CRUD files (db.js/py, database.js/py, models.py, etc.)
    - P3 (low): Business logic files (routes/, frontend/, middleware/, etc.) — ignored

    Returns at most the top 3 highest-priority files.
    """
    from pathlib import PurePath

    def get_priority(file_path: str) -> int:
        """Return a priority score; lower number = higher priority."""
        path = PurePath(file_path)
        filename = path.name.lower()
        parts = [p.lower() for p in path.parts]

        # P3 - low priority (ignored): return 99
        ignore_dirs = {'frontend', 'client', 'public', 'routes', 'views', 'controllers',
                       'middleware', 'components', 'pages', 'src/components', 'src/pages'}
        if any(ignore_dir in parts for ignore_dir in ignore_dirs):
            return 99

        # P1 - high priority: table schema definition files
        p1_patterns = [
            'init-db', 'init_db', 'init_dynamodb', 'initdb',
            'setup-db', 'setup_db', 'setup_dynamodb',
            'create-tables', 'create_tables', 'createtables',
            '_tables.py'  # e.g. init_dynamodb_tables.py
        ]
        if any(pattern in filename for pattern in p1_patterns):
            return 1

        # P2 - medium priority: database config and CRUD files
        p2_filenames = {'db.js', 'db.py', 'database.js', 'database.py',
                        'dynamodb.js', 'dynamodb.py', 'models.py'}
        p2_allowed_dirs = {'config', 'database', 'utils', 'services', 'app/models', ''}  # '' = root

        if filename in p2_filenames:
            # Accept files in root or explicitly allowed directories.
            parent_dir = parts[-2] if len(parts) > 1 else ''
            if parent_dir in p2_allowed_dirs or len(parts) == 1:
                return 2
            # Also accept files inside a 'database/' directory.
            if 'database' in parts:
                return 2
        
        # Default: lower priority (still included).
        return 50

    # Compute priority for every file.
    file_priorities = [(f, get_priority(f)) for f in files]

    # Drop ignored files (priority == 99).
    file_priorities = [(f, p) for f, p in file_priorities if p < 99]

    # Sort by priority first, then filename for stable ordering.
    file_priorities.sort(key=lambda x: (x[1], x[0]))

    # Return at most the top 3 files.
    top_files = [f for f, _ in file_priorities[:3]]

    return top_files


def extract_dynamodb_info(monolith_root: Path, file_tags: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Extract lightweight DynamoDB metadata from the monolith:
    1. Whether DynamoDB is used at all.
    2. Probable table names (derived from env-var defaults and hard-coded strings).
    3. Schema-definition files (top 3 by priority, for the SAM Engineer to read at runtime).

    Full KeySchema / GSI structures are intentionally not extracted here; the SAM
    Engineer agent reads the schema files directly at runtime.
    """
    info = {
        "used": False,
        "probable_tables": [],
        "schema_files": []
    }

    # Find files tagged with DynamoDB.
    db_files = [f for f, tags in file_tags.items() if "DynamoDB" in tags]
    if not db_files:
        return info

    info["used"] = True
    # Keep only the top 3 most likely schema-definition files.
    info["schema_files"] = _prioritize_schema_files(db_files)

    # Extract probable table names from env-var defaults and hard-coded strings.
    for rel_file_path in db_files:
        file_path = monolith_root / rel_file_path
        if not file_path.exists():
            continue

        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                source = f.read()

            # Python pattern 1: os.environ.get('XXX_TABLE_XXX', 'default-table')
            # Only match env vars containing TABLE to avoid false positives like AWS_REGION.
            pattern1 = re.findall(
                r"environ\.get\s*\(\s*['\"]([A-Z_]*TABLE[A-Z_]*)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)",
                source
            )
            info["probable_tables"].extend([m[1] for m in pattern1])

            # Python pattern 2: os.getenv('XXX_TABLE_XXX', 'default-table')
            pattern2 = re.findall(
                r"getenv\s*\(\s*['\"]([A-Z_]*TABLE[A-Z_]*)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)",
                source
            )
            info["probable_tables"].extend([m[1] for m in pattern2])

            # JavaScript pattern 1: process.env.XXX_TABLE_XXX || 'default-table'
            pattern3 = re.findall(
                r"process\.env\.([A-Z_]*TABLE[A-Z_]*)\s*\|\|\s*['\"]([^'\"]+)['\"]",
                source
            )
            info["probable_tables"].extend([m[1] for m in pattern3])

            # Generic pattern: TableName='hardcoded'  or  TableName: 'hardcoded'
            pattern4 = re.findall(
                r"[Tt]ableName\s*[=:]\s*['\"]([^'\"]+)['\"]",
                source
            )
            info["probable_tables"].extend(pattern4)

            # JavaScript pattern 2: config object with well-known table keys
            # e.g. const tables = { TODO_TABLE: 'todo-table', ... }
            pattern5 = re.findall(
                r"(?:TODO|USER|ORDER|PRODUCT|CART|BOOKING|FLIGHT|LOYALTY)_TABLE\s*:\s*['\"]([^'\"]+)['\"]",
                source,
                re.IGNORECASE
            )
            info["probable_tables"].extend(pattern5)

        except Exception as e:
            print(f"Warning: Failed to extract table names from {rel_file_path}: {e}")
            continue

    # Deduplicate and sort.
    info["probable_tables"] = sorted(list(set(info["probable_tables"])))

    return info




def build_entry_point_call_map(
    entry_points: List[Dict[str, Any]],
    call_edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Aggregate raw call-graph edges per HTTP entry_point.

    For each entry_point that has a ``handler_function`` field, collects every
    cross-file call edge whose (caller_file, caller_function) matches, producing
    a compact dependency map that Architect can consume directly.

    Returns:
        [
          {
            "entry_point": {"method": "POST", "path": "/bookings", "file": "..."},
            "handler_function": "create_booking",
            "cross_file_calls": [
              {"callee_module": "services.catalog",
               "callee_symbol": "CatalogService.reserve_flight_seat"},
              ...
            ]
          }
        ]
    """
    if not call_edges:
        return []

    # Index edges by (caller_file, caller_function) for O(1) lookup
    edges_index: Dict[Tuple[str, str], List[Dict]] = {}
    for edge in call_edges:
        key = (edge["caller_file"], edge["caller_function"])
        edges_index.setdefault(key, []).append(edge)

    result: List[Dict[str, Any]] = []
    for ep in entry_points:
        handler = ep.get("handler_function")
        if not handler:
            continue

        key = (ep["file"], handler)
        matching = edges_index.get(key, [])
        if not matching:
            continue

        calls_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for e in matching:
            ck = (e["callee_module"], e["callee_symbol"])
            if ck not in calls_map:
                calls_map[ck] = {
                    "callee_module": e["callee_module"],
                    "callee_symbol": e["callee_symbol"],
                    "return_value_used": e.get("return_value_used", False),
                    "is_awaited": e.get("is_awaited", False),
                }
            else:
                if e.get("return_value_used"):
                    calls_map[ck]["return_value_used"] = True
                if e.get("is_awaited"):
                    calls_map[ck]["is_awaited"] = True
        calls = list(calls_map.values())

        result.append({
            "entry_point": {
                "method": ep["method"],
                "path": ep["path"],
                "file": ep["file"],
            },
            "handler_function": handler,
            "cross_file_calls": calls,
        })

    return result


def run_static_analysis(monolith_root: Path) -> Dict[str, Any]:
    project_structure = build_project_structure(monolith_root)

    # Use the root directory name as the application identifier.
    app_name = monolith_root.name

    # Parse project dependency manifests.
    config_info = analyze_project_config(monolith_root)

    dependency_graph: Dict[str, List[str]] = {}
    file_tags: Dict[str, List[str]] = {}
    entry_points: List[Dict[str, Any]] = []
    symbol_table: List[Dict[str, Any]] = []

    # Cache (file_path, ext, source) for the call-graph second pass
    source_cache: List[Tuple[Path, str, str]] = []

    # ---- Pass 1: per-file structure analysis (existing logic) ----
    for dirpath, dirnames, filenames in os.walk(monolith_root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith('.')]

        for name in filenames:
            if name.startswith("."):
                continue

            file_path = Path(dirpath) / name
            ext = file_path.suffix.lower()

            if ext not in {".py", ".js", ".ts", ".jsx", ".tsx", ".vue"}:
                continue

            if ext == ".py":
                result = analyze_python_file(monolith_root, file_path, app_name)
            else:
                result = analyze_js_like_file(monolith_root, file_path, app_name)

            rel_path = result["rel_path"]
            dependency_graph[rel_path] = result["dependencies"]
            if result["tags"]:
                file_tags[rel_path] = result["tags"]
            entry_points.extend(result["entry_points"])
            symbol_table.extend(result["symbols"])

            # Read source once for the call-graph pass
            try:
                with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                    source_cache.append((file_path, ext, f.read()))
            except Exception:
                pass

    # ---- Pass 2: cross-file call graph extraction ----
    all_call_edges: List[Dict[str, Any]] = []

    # Pre-index symbol_table by file for JS enclosing-function lookup
    symbols_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for sym in symbol_table:
        fp = sym.get("file_path", "")
        symbols_by_file.setdefault(fp, []).append(sym)

    for file_path, ext, source in source_cache:
        if ext == ".py":
            edges = extract_python_call_graph(
                monolith_root, file_path, source, app_name
            )
        else:
            rel_path = str(file_path.relative_to(monolith_root).as_posix())
            if app_name:
                rel_path = f"{app_name}/{rel_path}"
            file_syms = symbols_by_file.get(rel_path, [])
            edges = extract_js_call_graph(
                monolith_root, file_path, source, file_syms, app_name
            )
        all_call_edges.extend(edges)

    # ---- Aggregate: per-entry-point dependency map ----
    entry_point_deps = build_entry_point_call_map(entry_points, all_call_edges)

    # ---- DynamoDB info ----
    dynamodb_info = extract_dynamodb_info(monolith_root, file_tags)

    # ---- Assemble report ----
    report: Dict[str, Any] = {
        "project_structure": project_structure,
        "entry_points": entry_points,
        "dependency_graph": dependency_graph,
        "file_tags": file_tags,
        "symbol_table": symbol_table,
    }

    if all_call_edges:
        report["call_graph"] = all_call_edges
    if entry_point_deps:
        report["entry_point_dependencies"] = entry_point_deps

    if config_info:
        report["config_info"] = config_info

    if dynamodb_info["used"]:
        report["dynamodb_info"] = dynamodb_info

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="MAG static analyzer")
    parser.add_argument(
        "--monolith-root",
        type=str,
        default=str(DEFAULT_MONOLITH_DIR),
        help="Path to the monolith source root (default: input_monolith)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for analysis_report.json",
    )

    args = parser.parse_args()

    monolith_root = Path(args.monolith_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not monolith_root.exists():
        raise SystemExit(f"Monolith root not found: {monolith_root}")
    
    report = run_static_analysis(monolith_root)

    symbol_table = report.pop("symbol_table", [])
    symbol_table_path = output_path.parent / "symbol_table.json"
    with symbol_table_path.open("w", encoding="utf-8") as f:
        json.dump(symbol_table, f, indent=2, ensure_ascii=False)
    print(f"Symbol table ({len(symbol_table)} symbols) written to {symbol_table_path}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Static analysis report written to {output_path}")


if __name__ == "__main__":
    main()
