#!/usr/bin/env python3
import os
import re
import json

hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

found = set()
for root, _, files in os.walk(hf_home):
    if any(f in files for f in ("config.json", "tokenizer.json", "model_index.json")):
        found.add(root)

# regex to match both "repos--OWNER--REPO" and "models--OWNER--REPO"
_pattern = re.compile(r'(?:repos|models)--([^/]+)--(.+?)(?:/|$)')

def infer_owner_model(path):
    # 1) direct match in the full path
    m = _pattern.search(path)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    # 2) climb up parent directories (useful when path points into snapshots/<sha>/...)
    p = os.path.abspath(path)
    for _ in range(8):
        p = os.path.dirname(p)
        if not p or p == '/':
            break
        m = _pattern.search(p)
        if m:
            return f"{m.group(1)}/{m.group(2)}"

    # 3) try to read common metadata files and look for repo/id fields
    for root, _, files in os.walk(path):
        for fname in ("model_index.json", "config.json", "tokenizer.json"):
            if fname in files:
                fp = os.path.join(root, fname)
                try:
                    with open(fp, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    # common fields that sometimes contain "owner/model" or model id
                    for key in ("modelId", "id", "_name_or_path", "name_or_path", "repo_id", "model_name"):
                        v = data.get(key)
                        if isinstance(v, str) and v:
                            return v
                except Exception:
                    # ignore parse errors and continue
                    pass

    # 4) try to extract from a path segment like "models--OWNER--REPO" more defensively
    parts = [p for p in path.split('/') if p]
    for part in parts[::-1]:
        if part.startswith("models--") or part.startswith("repos--"):
            sub = part
            if part.startswith("models--"):
                sub = part[len("models--"):]
            elif part.startswith("repos--"):
                sub = part[len("repos--"):]
            if "--" in sub:
                owner, repo = sub.split("--", 1)
                return f"{owner}/{repo}"

    # 5) fallback: return last two path components as a heuristic
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"

    # 6) ultimate fallback: return the original path
    return path

for p in sorted(found):
    print(infer_owner_model(p))

