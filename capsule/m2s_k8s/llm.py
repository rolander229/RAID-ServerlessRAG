"""LLM client abstraction for the M2' pipeline.

Contract (runner-facing): ``chat(messages) -> str`` where messages is a list
of {"role": "system"|"user"|"assistant", "content": str}.  Swapping this for
a CrewAI LLM later only means adapting the call convention, not the pipeline.

Two implementations:

  * FakeLLM  -- canned responses keyed by the task marker the runner embeds
                in every user message (``# m2s-k8s task: <id> ...``).  Lets
                the whole pipeline run offline in tests / CI.
  * ArkLLM   -- OpenAI-compatible chat-completions client for the Volc Ark
                endpoint.  Configuration comes EXCLUSIVELY from environment
                variables -- no key is ever hardcoded:

                    ARK_API_KEY     bearer token (parent session provisions
                                    it via arkcli; never commit it)
                    ARK_BASE_URL    default https://ark.cn-beijing.volces.com/api/v3
                    ARK_MODEL       model id or endpoint id (ep-...)
                    ARK_TIMEOUT_S   optional, default 120

ArkLLM uses only urllib (stdlib): the venv has no `openai`/`requests` need
and installing into the torch-pinned venv is a risk we deliberately avoid.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

ARK_BASE_URL_DEFAULT = "https://ark.cn-beijing.volces.com/api/v3"
TASK_MARKER_RE = re.compile(r"#\s*m2s-k8s task:\s*([A-Za-z0-9_]+)")

Msg = Dict[str, str]


class LLMError(RuntimeError):
    """Raised when an LLM call fails after retries."""


def extract_task_id(messages: List[Msg]) -> Optional[str]:
    """Find the task marker in the last user message (FakeLLM dispatch key)."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            m = TASK_MARKER_RE.search(msg.get("content", ""))
            return m.group(1) if m else None
    return None


class LLMClient:
    """Minimal interface -- structurally compatible with crewai's LLM.call."""

    def chat(self, messages: List[Msg], *, temperature: float = 0.2,
             max_tokens: Optional[int] = None) -> str:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"kind": type(self).__name__}


class FakeLLM(LLMClient):
    """Canned responses for offline pipeline execution (resnet50 golden path)."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = dict(responses or {})
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages: List[Msg], *, temperature: float = 0.2,
             max_tokens: Optional[int] = None) -> str:
        task_id = extract_task_id(messages)
        self.calls.append({"task_id": task_id, "n_messages": len(messages)})
        if task_id not in self.responses:
            raise LLMError(
                f"FakeLLM has no canned response for task {task_id!r}; "
                f"known tasks: {sorted(self.responses)}"
            )
        return self.responses[task_id]

    def describe(self) -> Dict[str, Any]:
        return {"kind": "FakeLLM", "model": None,
                "canned_tasks": sorted(self.responses)}


class ArkLLM(LLMClient):
    """OpenAI-compatible client for the Ark endpoint (env-configured)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, timeout_s: Optional[float] = None,
                 max_retries: int = 3):
        self.api_key = (api_key or os.getenv("ARK_API_KEY") or "").strip()
        self.base_url = (base_url or os.getenv("ARK_BASE_URL") or ARK_BASE_URL_DEFAULT).rstrip("/")
        self.model = (model or os.getenv("ARK_MODEL") or "").strip()
        self.timeout_s = float(timeout_s if timeout_s is not None
                               else os.getenv("ARK_TIMEOUT_S", "120"))
        self.max_retries = max_retries
        self.calls: List[Dict[str, Any]] = []

    # -- introspection -----------------------------------------------------
    @staticmethod
    def env_status() -> Dict[str, Any]:
        key = (os.getenv("ARK_API_KEY") or "").strip()
        return {
            "ARK_API_KEY": {"set": bool(key), "hint": _key_hint(key)},
            "ARK_BASE_URL": {"set": bool(os.getenv("ARK_BASE_URL")),
                             "default": ARK_BASE_URL_DEFAULT},
            "ARK_MODEL": {"set": bool((os.getenv("ARK_MODEL") or "").strip())},
        }

    def missing_config(self) -> List[str]:
        missing = []
        if not self.api_key:
            missing.append("ARK_API_KEY")
        if not self.model:
            missing.append("ARK_MODEL")
        return missing

    def describe(self) -> Dict[str, Any]:
        return {"kind": "ArkLLM", "model": self.model or None,
                "base_url": self.base_url,
                "api_key_set": bool(self.api_key),
                "timeout_s": self.timeout_s}

    # -- the call ----------------------------------------------------------
    def chat(self, messages: List[Msg], *, temperature: float = 0.2,
             max_tokens: Optional[int] = None) -> str:
        if self.missing_config():
            raise LLMError(
                "ArkLLM is not configured: missing "
                f"{', '.join(self.missing_config())}. {CONFIG_GUIDANCE}"
            )
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        url = f"{self.base_url}/chat/completions"
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self.calls.append({"attempt": attempt, "url": url, "model": self.model,
                               "n_messages": len(messages)})
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                content = (((body or {}).get("choices") or [{}])[0]
                           .get("message", {}).get("content"))
                if content is None:
                    raise LLMError(f"ArkLLM: no message content in response: "
                                   f"{json.dumps(body)[:400]}")
                return content
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:  # noqa: BLE001 -- best-effort error detail
                    pass
                last_error = LLMError(
                    f"ArkLLM HTTP {exc.code} from {url}: {detail or exc.reason}"
                )
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = LLMError(f"ArkLLM cannot reach {url}: {exc.reason}")
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise last_error from exc
        raise last_error or LLMError("ArkLLM failed for an unknown reason")


def _key_hint(key: str) -> str:
    if not key:
        return "not set"
    return f"set ({len(key)} chars, starts {key[:4]}...)"


CONFIG_GUIDANCE = """How to configure the Ark LLM base (no key is hardcoded anywhere):
  1. Generate a DEDICATED API key with arkcli (do not reuse a personal key):
       arkcli auth apikey create --name mopar-m2s   (or your arkcli's equivalent)
  2. Export the three variables before running the pipeline:
       export ARK_API_KEY="<the generated key>"
       export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
       export ARK_MODEL="<model id or ep-... endpoint id>"
     dell-65 can reach the public endpoint; keep the key out of git
     (env-only, or a git-ignored .env sourced by your shell).
  3. Verify connectivity without consuming tokens:
       python -m capsule.m2s_k8s.runner --check-llm
  4. Run the pipeline for real:
       python -m capsule.m2s_k8s.runner --manifest <slice_manifest.json> \
           --out runs/m2s_ark_run --generated-at 2026-11-03T00:00:00Z
     Model guidance: thinking-class model for Architect/Validator, code-class
     (e.g. a DeepSeek coder endpoint) for Developer; keep FakeLLM for CI."""
