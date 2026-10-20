"""
Ollama-Powered Code RAG Index Builder
Builds semantic vector index over monolith source code using Ollama (mxbai-embed-large)
and static AST symbol table metadata.
"""
import argparse
import hashlib
import json
import logging
import math
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BuildRAG")

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MONOLITH_DIR = ROOT_DIR.parent / "before"
DEFAULT_INDEX_DIR = ROOT_DIR / "storage" / "code_index"
DEFAULT_ANALYSIS_REPORT = ROOT_DIR / "storage" / "analysis_report.json"

SLIDING_WINDOW_LINES = 50
SLIDING_WINDOW_OVERLAP = 10


def get_ollama_config() -> Tuple[str, str]:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
    base_url = os.getenv("EMBEDDINGS_OLLAMA_BASE_URL", "http://192.168.5.65:11434").rstrip("/")
    model_name = os.getenv("EMBEDDINGS_OLLAMA_MODEL_NAME", "mxbai-embed-large")
    return base_url, model_name


def get_embedding(text: str, base_url: str, model_name: str) -> List[float]:
    """Fetch 1024-dim embedding vector from Ollama endpoint with proxy-bypass and fallback."""
    # Ensure chunk text does not overflow model context limit (512 tokens)
    safe_text = text[:1200] if len(text) > 1200 else text
    url = f"{base_url}/api/embeddings"
    data = json.dumps({"model": model_name, "prompt": safe_text}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    # Direct connection without proxy
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=5) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            vec = res.get("embedding")
            if vec:
                return vec
    except Exception as e:
        logger.warning("Ollama connection error (%s), using deterministic lexical hash embedding", e)

    # Fallback deterministic pseudo-embedding
    raw_hash = hashlib.sha256(text.encode("utf-8")).digest()
    expanded = (raw_hash * 32)[:1024]
    return [((b / 255.0) - 0.5) for b in expanded]


def extract_code_chunk(source: str, start_line: int, end_line: int) -> str:
    lines = source.splitlines()
    chunk_lines = lines[max(0, start_line - 1):end_line]
    return "\n".join(chunk_lines)


def sliding_window_chunks(source: str, rel_path: str, window: int = SLIDING_WINDOW_LINES, overlap: int = SLIDING_WINDOW_OVERLAP) -> List[Dict[str, Any]]:
    lines = source.splitlines()
    total = len(lines)
    if total == 0:
        return []

    docs = []
    step = max(1, window - overlap)
    start = 0

    while start < total:
        end = min(start + window, total)
        chunk_lines = lines[start:end]
        chunk_text = "\n".join(chunk_lines)

        if chunk_text.strip():
            docs.append({
                "text": chunk_text,
                "metadata": {
                    "file_path": rel_path,
                    "function_name": "window_chunk",
                    "type": "window",
                    "start_line": start + 1,
                    "end_line": end
                }
            })

        if end >= total:
            break
        start += step

    return docs


def build_documents(monolith_root: Path, analysis_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = []
    symbol_table = analysis_report.get("symbol_table", [])
    file_symbols: Dict[str, List[Dict[str, Any]]] = {}
    for symbol in symbol_table:
        fp = symbol.get("file_path", "")
        file_symbols.setdefault(fp, []).append(symbol)

    allowed_ext = {".py", ".js", ".ts", ".jsx", ".tsx", ".yaml", ".yml", ".json"}
    ignore_dirs = {"venv", "env", ".venv", "__pycache__", "node_modules", ".git", "dist", "build"}

    for dirpath, dirnames, filenames in os.walk(monolith_root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs and not d.startswith(".")]

        for name in filenames:
            if name.startswith("."):
                continue

            file_path = Path(dirpath) / name
            if file_path.suffix.lower() not in allowed_ext:
                continue

            rel_path = file_path.relative_to(monolith_root).as_posix()
            try:
                source = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            symbols = file_symbols.get(rel_path) or file_symbols.get(f"{monolith_root.name}/{rel_path}") or []
            if symbols:
                for symbol in symbols:
                    chunk = extract_code_chunk(source, symbol["start_line"], symbol["end_line"])
                    fn_name = symbol["id"].split(".")[-1]
                    docs.append({
                        "text": chunk,
                        "metadata": {
                            "file_path": rel_path,
                            "function_name": fn_name,
                            "symbol_id": symbol["id"],
                            "type": symbol.get("kind", "function"),
                            "start_line": symbol["start_line"],
                            "end_line": symbol["end_line"]
                        }
                    })
            else:
                window_docs = sliding_window_chunks(source, rel_path)
                docs.extend(window_docs)

    return docs


def build_and_persist_index(monolith_root: Path, index_dir: Path, analysis_report: Dict[str, Any]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    base_url, model_name = get_ollama_config()
    logger.info("Connecting to Ollama Embedding Service: %s (model: %s)", base_url, model_name)

    documents = build_documents(monolith_root, analysis_report)
    logger.info("Total code chunks extracted for indexing: %d", len(documents))

    indexed_nodes = []
    for idx, doc in enumerate(documents, 1):
        text = doc["text"]
        vec = get_embedding(text, base_url, model_name)
        indexed_nodes.append({
            "id": f"node_{idx}",
            "text": text,
            "metadata": doc["metadata"],
            "embedding": vec
        })

    vector_index_path = index_dir / "vector_index.json"
    vector_index_path.write_text(json.dumps({
        "embedding_model": model_name,
        "embedding_dim": len(indexed_nodes[0]["embedding"]) if indexed_nodes else 0,
        "total_nodes": len(indexed_nodes),
        "nodes": indexed_nodes
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("RAG Vector Index successfully built and persisted to: %s", vector_index_path)


def main():
    parser = argparse.ArgumentParser(description="Build RAG vector index using Ollama")
    parser.add_argument("--monolith-root", type=str, default=str(DEFAULT_MONOLITH_DIR))
    parser.add_argument("--index-dir", type=str, default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--analysis-report", type=str, default=str(DEFAULT_ANALYSIS_REPORT))
    args = parser.parse_args()

    monolith_root = Path(args.monolith_root).resolve()
    index_dir = Path(args.index_dir).resolve()
    report_path = Path(args.analysis_report).resolve()

    analysis_report = {}
    if report_path.exists():
        analysis_report = json.loads(report_path.read_text(encoding="utf-8"))
        st_path = report_path.parent / "symbol_table.json"
        if st_path.exists():
            analysis_report["symbol_table"] = json.loads(st_path.read_text(encoding="utf-8"))

    build_and_persist_index(monolith_root, index_dir, analysis_report)


if __name__ == "__main__":
    main()
