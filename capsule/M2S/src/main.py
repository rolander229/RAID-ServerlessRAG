from pathlib import Path
import json
import os
import time
import sys
import functools
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
TASK_ID_BY_OBJECT_ID: dict[int, str] = {}

load_dotenv(ROOT_DIR / ".env", override=True)

# Reduce CrewAI's "remember format" frequency from every 3 tool calls to every 15,
# saving ~80% of redundant token overhead in message history.
from crewai.tools.tool_usage import ToolUsage  # noqa: E402
ToolUsage._should_remember_format = lambda self: (
    self.task is not None and self.task.used_tools % 8 == 0
)


# ---------------------------------------------------------------------------
# Retry wrapper for LLM throttling errors (400/429 ThrottlingException)
# CrewAI does not retry on 400 BadRequestError, but some providers (e.g.
# CodeWhisperer/Bedrock) wrap throttling as 400 instead of 429.
# ---------------------------------------------------------------------------
def _patch_llm_with_throttle_retry():
    """Monkey-patch CrewAI's OpenAI completion to retry on throttling errors."""
    try:
        from crewai.llms.providers.openai.completion import OpenAICompletion

        completion_cls = OpenAICompletion
    except ImportError:
        # Backward compatibility for older CrewAI releases.
        from crewai.llms.providers.openai.completion import OpenAICompletionProvider

        completion_cls = OpenAICompletionProvider

    original_handle = completion_cls._handle_completion

    @functools.wraps(original_handle)
    def _handle_with_retry(self, *args, **kwargs):
        max_retries = 5
        base_delay = 30  # seconds
        for attempt in range(max_retries):
            try:
                return original_handle(self, *args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                is_throttle = (
                    "throttling" in err_str
                    or "too many requests" in err_str
                    or "rate limit" in err_str
                )
                if is_throttle and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(
                        f"\nLLM throttled (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {delay}s before retry...\n"
                    )
                    time.sleep(delay)
                    continue
                raise

    completion_cls._handle_completion = _handle_with_retry
    print("LLM throttle-retry patch applied (max 5 retries, exponential backoff)")


_patch_llm_with_throttle_retry()


# ---------------------------------------------------------------------------
# Inter-task cooldown: pause between sequential tasks to avoid API throttling
# ---------------------------------------------------------------------------
INTER_TASK_COOLDOWN_SECONDS = int(os.getenv("INTER_TASK_COOLDOWN", "15"))


def _load_expected_lambda_names() -> list[str]:
    """Load expected Lambda names from the generated blueprint."""
    blueprint_path = ROOT_DIR / "storage" / "blueprint.json"
    if not blueprint_path.exists():
        raise RuntimeError(
            "Code generation validation failed: storage/blueprint.json does not exist."
        )

    try:
        blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Code generation validation failed: blueprint.json is invalid JSON: {e}"
        ) from e

    lambda_defs = blueprint.get("lambda_functions")
    if not isinstance(lambda_defs, list) or not lambda_defs:
        raise RuntimeError(
            "Code generation validation failed: blueprint.json has no lambda_functions."
        )

    lambda_names: list[str] = []
    for idx, item in enumerate(lambda_defs, start=1):
        if not isinstance(item, dict) or not item.get("name"):
            raise RuntimeError(
                "Code generation validation failed: "
                f"lambda_functions[{idx}] is missing a valid name."
            )
        lambda_names.append(str(item["name"]))

    return lambda_names


def validate_code_generation_artifacts() -> None:
    """Ensure the code-generation task really wrote every expected handler to disk."""
    expected_lambda_names = _load_expected_lambda_names()
    lambdas_root = ROOT_DIR / "output" / "lambdas"
    if not lambdas_root.exists():
        raise RuntimeError(
            "Code generation task reported success, but output/lambdas does not exist."
        )

    handler_paths = sorted(lambdas_root.rglob("handler.py")) + sorted(
        lambdas_root.rglob("handler.js")
    )
    if not handler_paths:
        raise RuntimeError(
            "Code generation task reported success, but no handler.py or handler.js "
            "files were written under output/lambdas."
        )

    generated_lambda_names = {path.parent.name for path in handler_paths}
    expected_lambda_name_set = set(expected_lambda_names)

    missing = sorted(expected_lambda_name_set - generated_lambda_names)
    unexpected = sorted(generated_lambda_names - expected_lambda_name_set)

    if missing or unexpected or len(handler_paths) != len(expected_lambda_names):
        details = [
            "Code generation validation failed after generate_complete_application.",
            f"Expected {len(expected_lambda_names)} Lambda handlers from blueprint.json, "
            f"but found {len(handler_paths)} handler files on disk.",
        ]
        if missing:
            details.append(f"Missing handlers for Lambda names: {', '.join(missing[:12])}")
        if len(missing) > 12:
            details.append(f"...and {len(missing) - 12} more missing handlers.")
        if unexpected:
            details.append(
                f"Unexpected handler directories found: {', '.join(unexpected[:12])}"
            )
        if len(unexpected) > 12:
            details.append(f"...and {len(unexpected) - 12} more unexpected handlers.")
        raise RuntimeError(" ".join(details))

    print(
        f" code generation validation passed "
        f"({len(handler_paths)}/{len(expected_lambda_names)} handlers present on disk)"
    )


def _validate_post_task_artifacts(task_obj) -> None:
    """Run targeted file-system validations after critical tasks."""
    task_id = TASK_ID_BY_OBJECT_ID.get(id(task_obj))
    if task_id == "generate_complete_application":
        validate_code_generation_artifacts()


def _patch_inter_task_cooldown():
    """Monkey-patch CrewAI's _execute_tasks to add cooldown between tasks."""
    from crewai.crew import Crew

    original_execute = Crew._execute_tasks

    @functools.wraps(original_execute)
    def _execute_with_cooldown(self, tasks, start_index=0, was_replayed=False):
        from crewai.task import Task

        original_sync = Task.execute_sync

        task_counter = {"count": 0}

        @functools.wraps(original_sync)
        def _sync_with_cooldown(task_self, *args, **kwargs):
            task_counter["count"] += 1
            if task_counter["count"] > 1 and INTER_TASK_COOLDOWN_SECONDS > 0:
                print(
                    f"\n⏳ Inter-task cooldown: waiting {INTER_TASK_COOLDOWN_SECONDS}s "
                    f"before task #{task_counter['count']}...\n"
                )
                time.sleep(INTER_TASK_COOLDOWN_SECONDS)
            result = original_sync(task_self, *args, **kwargs)
            _validate_post_task_artifacts(task_self)
            return result

        Task.execute_sync = _sync_with_cooldown
        try:
            return original_execute(self, tasks, start_index, was_replayed)
        finally:
            Task.execute_sync = original_sync

    Crew._execute_tasks = _execute_with_cooldown
    print(f"Inter-task cooldown patch applied ({INTER_TASK_COOLDOWN_SECONDS}s between tasks)")


_patch_inter_task_cooldown()


def configure_console_encoding() -> None:
    """Force UTF-8 console output to avoid Windows GBK UnicodeEncodeError."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


configure_console_encoding()


def _get_env(key: str) -> str:
    val = os.getenv(key)
    return (val or "").strip()


def validate_llm_env() -> None:
    """
    Quick sanity-check before the first LLM call to surface configuration errors
    with readable messages rather than raw provider stack traces.

    Supports:
    - Gemini (Google Generative Language API)
    - OpenAI-compatible endpoints (e.g. DeepSeek)
    """
    model = _get_env("OPENAI_MODEL_NAME")

    if model and "gemini" in model.lower():
        google_key = _get_env("GOOGLE_API_KEY")
        if not google_key:
            raise ValueError(
                "Model is configured as Gemini, but GOOGLE_API_KEY is not set.\n"
                "Set a valid Google AI Studio (Gemini API) key in .env, e.g.:\n"
                "  GOOGLE_API_KEY=AIza...  (no quotes)\n"
                "Then re-run: python src/main.py"
            )
        if google_key.lower().startswith("sk-"):
            raise ValueError(
                "Model is configured as Gemini, but GOOGLE_API_KEY looks like an "
                "OpenAI/DeepSeek key (starts with 'sk-').\n"
                "Gemini API keys typically start with 'AIza' and must be obtained "
                "from Google AI Studio. Please update GOOGLE_API_KEY and retry."
            )

    if model and "deepseek" in model.lower():
        if not _get_env("OPENAI_API_KEY"):
            raise ValueError(
                "Model appears to use a DeepSeek/OpenAI-compatible endpoint, "
                "but OPENAI_API_KEY is empty.\n"
                "Set OPENAI_API_KEY in .env (and ensure OPENAI_API_BASE points "
                "to the correct service)."
            )


validate_llm_env()

from crewai import Agent, Task, Crew
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

from src.tools.file_tools import ReadFileTool, WriteFileTool, FileListTool
from src.tools.rag_tools import CodeRAGTool
from src.tools.sam_validate_tool import SAMValidateTool


def load_yaml(path: Path):
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


OLLAMA_EMBEDDER_CONFIG = {
    "provider": "ollama",
    "config": {
        "model": "mxbai-embed-large",
        "base_url": "http://127.0.0.1:11434"
    }
}


def build_agents() -> dict:
    """Build the four core agents: Architect, Code Developer, SAM Engineer, Consistency Validator."""
    config = load_yaml(ROOT_DIR / "src" / "config" / "agents.yaml")

    # Knowledge Sources
    sam_knowledge = TextFileKnowledgeSource(
        file_paths=["sam_reference.md"],
        chunk_size=1200,
        chunk_overlap=150
    )

    lambda_coding_knowledge = TextFileKnowledgeSource(
        file_paths=["lambda_coding_reference.md"],
        chunk_size=1200,
        chunk_overlap=150
    )

    basic_serverless_knowledge = TextFileKnowledgeSource(
        file_paths=["basic_serverless_architecture.md"],
        chunk_size=1200,
        chunk_overlap=150
    )

    async_patterns_knowledge = TextFileKnowledgeSource(
        file_paths=["async_serverless_patterns.md"],
        chunk_size=1200,
        chunk_overlap=150
    )

    agents = {}

    # 1. Architect
    arch_cfg = config["architect"]
    agents["architect"] = Agent(
        role=arch_cfg["role"],
        goal=arch_cfg["goal"],
        backstory=arch_cfg["backstory"],
        tools=[ReadFileTool(ROOT_DIR), WriteFileTool(ROOT_DIR)],
        knowledge_sources=[basic_serverless_knowledge, async_patterns_knowledge],
        embedder=OLLAMA_EMBEDDER_CONFIG,
        verbose=True,
        allow_delegation=False,
        max_iter=arch_cfg.get("max_iter", 100),
        max_rpm=3,  # Very conservative limit to avoid throttling
    )

    # 2. Code Developer
    code_cfg = config["code_developer"]
    agents["code_developer"] = Agent(
        role=code_cfg["role"],
        goal=code_cfg["goal"],
        backstory=code_cfg["backstory"],
        tools=[
            ReadFileTool(ROOT_DIR),
            CodeRAGTool(ROOT_DIR / "storage" / "code_index"),
            WriteFileTool(ROOT_DIR),
        ],
        knowledge_sources=[lambda_coding_knowledge, async_patterns_knowledge],
        embedder=OLLAMA_EMBEDDER_CONFIG,
        verbose=True,
        allow_delegation=False,
        max_iter=code_cfg.get("max_iter", 200),
        max_rpm=3,  # Very conservative limit to avoid throttling
    )

    # 3. SAM Engineer
    sam_cfg = config["sam_engineer"]
    agents["sam_engineer"] = Agent(
        role=sam_cfg["role"],
        goal=sam_cfg["goal"],
        backstory=sam_cfg["backstory"],
        tools=[
            ReadFileTool(ROOT_DIR),
            WriteFileTool(ROOT_DIR),
            FileListTool(ROOT_DIR),
            SAMValidateTool(ROOT_DIR),
        ],
        knowledge_sources=[sam_knowledge],
        embedder=OLLAMA_EMBEDDER_CONFIG,
        verbose=True,
        allow_delegation=False,
        max_iter=sam_cfg.get("max_iter", 200),
        max_rpm=3,  # Very conservative limit to avoid throttling
    )

    # 4. Consistency Validator
    cv_cfg = config["consistency_validator"]
    agents["consistency_validator"] = Agent(
        role=cv_cfg["role"],
        goal=cv_cfg["goal"],
        backstory=cv_cfg["backstory"],
        tools=[
            ReadFileTool(ROOT_DIR),
            WriteFileTool(ROOT_DIR),
            FileListTool(ROOT_DIR),
            SAMValidateTool(ROOT_DIR),
        ],
        knowledge_sources=[sam_knowledge, lambda_coding_knowledge, async_patterns_knowledge],
        embedder=OLLAMA_EMBEDDER_CONFIG,
        verbose=True,
        allow_delegation=False,
        max_iter=cv_cfg.get("max_iter", 200),
        max_rpm=3,  # Very conservative limit to avoid throttling
    )

    return agents


def build_tasks(agents: dict) -> list:
    """Build the task chain: Architecture → Code Generation → SAM Template → Consistency Validation."""
    config = load_yaml(ROOT_DIR / "src" / "config" / "tasks.yaml")
    tasks_cfg = config["tasks"]

    tasks: list[Task] = []

    # These tasks clear upstream context to avoid context-window overflow.
    # They rely entirely on files written to disk rather than CrewAI's task-output passing.
    context_clearing_tasks = {
        "generate_complete_application",
        "generate_shared_resources_template",
        "generate_functions_template",
        "assemble_final_template",
        "validate_consistency",
    }

    for task_cfg in tasks_cfg:
        agent_id = task_cfg["agent"]
        agent = agents.get(agent_id)

        if not agent:
            print(f"Warning: Agent '{agent_id}' not found, skipping task {task_cfg.get('id')}")
            continue

        task_kwargs = {
            "description": task_cfg["description"],
            "expected_output": task_cfg.get("expected_output", "Task completed successfully"),
            "agent": agent,
            "verbose": True,
        }

        if task_cfg.get("id") in context_clearing_tasks:
            task_kwargs["context"] = []

        task = Task(**task_kwargs)
        TASK_ID_BY_OBJECT_ID[id(task)] = task_cfg.get("id", "")

        if "output_file" in task_cfg:
            task.output_file = task_cfg["output_file"]

        tasks.append(task)

    return tasks


def run_crew() -> None:
    """Run the multi-agent crew to migrate a monolithic application to serverless."""
    try:
        print("=" * 60)
        print("Capsule - Monolith to AWS Serverless Migration")
        print("=" * 60)

        workflow_start = time.time()

        agents = build_agents()
        tasks = build_tasks(agents)

        print(f"\nInitialized {len(agents)} agents and {len(tasks)} tasks.")
        print("\nAgent roles:")
        for name, agent in agents.items():
            print(f"  - {name}: {agent.role}")

        print("\nStarting migration workflow...\n")

        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            verbose=True,
            memory=False,
            embedder=OLLAMA_EMBEDDER_CONFIG,
        )

        print("\n" + "=" * 60)
        print(f"Executing {len(tasks)} tasks sequentially:")
        print("=" * 60)
        for i, task in enumerate(tasks):
            agent_role = task.agent.role if task.agent else "unknown"
            print(f"  Task {i+1}: {agent_role}")
        print("=" * 60 + "\n")

        crew_start = time.time()
        result = crew.kickoff()
        crew_end = time.time()

        crew_duration = crew_end - crew_start
        print(f"\nCrew total time: {crew_duration:.2f}s ({crew_duration/60:.2f}min)")

        if hasattr(result, 'tasks_output') and result.tasks_output:
            print("\nTask execution summary:")
            for i, task_output in enumerate(result.tasks_output):
                if i < len(tasks):
                    agent_name = tasks[i].agent.role if tasks[i].agent else "unknown"
                    print(f"  Task {i+1} ({agent_name}) - completed")

        workflow_end = time.time()
        workflow_duration = workflow_end - workflow_start

        print("\n" + "=" * 60)
        print("Migration workflow completed!")
        print("=" * 60)
        print(f"Total workflow time: {workflow_duration:.2f}s ({workflow_duration/60:.2f}min)")
        print("\nResult:")
        print(json.dumps({"result": str(result)}, indent=2, ensure_ascii=False))

        print("\nGenerated artifacts:")
        print("  - storage/blueprint.json (Architecture blueprint)")
        print("  - output/lambdas/ (Lambda functions and shared code)")
        print("  - output/layers/ (Shared Lambda layers, if applicable)")
        print("  - output/template.yaml (AWS SAM template)")
        print("  - output/consistency_report.json (Cross-artifact validation report)")
        print("\nNext steps:")
        print("  1. Review the generated code in output/")
        print("  2. Configure deployment parameters")
        print("  3. Run: sam build && sam deploy --guided")

    except BaseException as e:
        import traceback
        error_msg = f"\nCRITICAL ERROR: {str(e)}\n\n{traceback.format_exc()}"
        try:
            print(error_msg)
        except UnicodeEncodeError:
            print(error_msg.encode("utf-8", errors="replace").decode("utf-8"))

        crash_log = ROOT_DIR / "crash_report.log"
        with open(crash_log, "w", encoding="utf-8") as f:
            f.write(error_msg)
        print(f"\nCrash report saved to: {crash_log}")
        raise


if __name__ == "__main__":
    run_crew()
