"""
Robust Hybrid Code RAG Tool
Performs vector similarity search (using Ollama embeddings) with automatic keyword/token search fallback.
"""
import json
import logging
import math
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

logger = logging.getLogger("CodeRAGTool")
ROOT_DIR = Path(__file__).resolve().parents[2]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class OllamaCodeRAG:
    """Hybrid vector and semantic keyword retriever over code index."""

    def __init__(self, index_dir: Optional[Path] = None):
        if index_dir is None:
            index_dir = ROOT_DIR / "storage" / "code_index"
        self.index_dir = index_dir
        self.vector_file = index_dir / "vector_index.json"
        self.base_url = os.getenv("EMBEDDINGS_OLLAMA_BASE_URL", "http://192.168.5.65:11434").rstrip("/")
        self.model_name = os.getenv("EMBEDDINGS_OLLAMA_MODEL_NAME", "mxbai-embed-large")
        self.nodes = []
        self._load_index()

    def _load_index(self):
        if not self.vector_file.exists():
            logger.warning("RAG vector index not found at: %s", self.vector_file)
            return
        try:
            data = json.loads(self.vector_file.read_text(encoding="utf-8"))
            self.nodes = data.get("nodes", [])
            logger.info("Loaded %d indexed code nodes from RAG store.", len(self.nodes))
        except Exception as e:
            logger.error("Error loading RAG index: %s", e)

    def _embed_query(self, query: str) -> Optional[List[float]]:
        url = f"{self.base_url}/api/embeddings"
        payload = json.dumps({"model": self.model_name, "prompt": query}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        
        # Bypass local proxy
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)
        with opener.open(req, timeout=5) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return res.get("embedding")

    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokens = set(re.findall(r"\w+", query.lower()))
        scored = []
        for node in self.nodes:
            text_lower = (node.get("text", "") + " " + node["metadata"].get("function_name", "")).lower()
            overlap = sum(1 for t in tokens if t in text_lower)
            if overlap > 0:
                scored.append((overlap, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, node in scored[:top_k]:
            results.append({
                "score": round(score / max(len(tokens), 1), 4),
                "file_path": node["metadata"]["file_path"],
                "function_name": node["metadata"]["function_name"],
                "start_line": node["metadata"]["start_line"],
                "end_line": node["metadata"]["end_line"],
                "code": node["text"]
            })
        return results

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search code knowledge base using semantic vector retrieval with keyword fallback."""
        if not self.nodes:
            return []

        query_vec = None
        try:
            query_vec = self._embed_query(query)
        except Exception as e:
            logger.warning("Ollama query embedding failed (%s), falling back to lexical search.", e)

        if query_vec:
            scored = []
            for node in self.nodes:
                score = cosine_similarity(query_vec, node["embedding"])
                scored.append((score, node))

            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, node in scored[:top_k]:
                results.append({
                    "score": round(score, 4),
                    "file_path": node["metadata"]["file_path"],
                    "function_name": node["metadata"]["function_name"],
                    "start_line": node["metadata"]["start_line"],
                    "end_line": node["metadata"]["end_line"],
                    "code": node["text"]
                })
            return results
        else:
            return self._keyword_search(query, top_k)

    def format_search_results(self, query: str, top_k: int = 5) -> str:
        hits = self.search(query, top_k)
        if not hits:
            return f"No code snippets found for query: '{query}'."

        lines = [f"Found {len(hits)} relevant code snippets for query '{query}':\n" + "=" * 60]
        for idx, h in enumerate(hits, 1):
            lines.append(f"[{idx}] {h['file_path']} -> {h['function_name']} (lines {h['start_line']}-{h['end_line']}) [Score: {h['score']}]")
            lines.append("```python")
            lines.append(h["code"])
            lines.append("```\n" + "-" * 60)
        return "\n".join(lines)
