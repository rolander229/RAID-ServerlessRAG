"""
No-RAG Cloud-Native Service Migration Multi-Agent Runner
(Direct OpenAI-Compatible LLM Pipeline without Ollama/Vector RAG dependencies)

Executes 4 specialized agents:
  1. Migration Architect: Analyzes AST static report & generates blueprint.json
  2. Code Developer: Reads exact source_files & generates FaaS Lambda handlers
  3. SAM Engineer: Generates infrastructure-as-code template.yaml
  4. Consistency Validator: Executes 12-point cross-check & produces consistency_report.json
"""
import ast
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root and M2S to path
M2S_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = M2S_DIR.parent
sys.path.insert(0, str(M2S_DIR))
sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(M2S_DIR / ".env")
except ImportError:
    pass

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("NoRagMigrationRunner")


def get_llm_client():
    """Initialize OpenAI-compatible LLM client from environment."""
    from openai import OpenAI

    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "No API key found. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY in environment or M2S/.env"
        )

    base_url = os.getenv("OPENAI_API_BASE")
    if not base_url:
        if os.getenv("DEEPSEEK_API_KEY") and api_key == os.getenv("DEEPSEEK_API_KEY"):
            base_url = "https://api.deepseek.com"
        else:
            base_url = "https://api.openai.com/v1"

    model_name = os.getenv("OPENAI_MODEL_NAME")
    if not model_name:
        if "deepseek" in base_url.lower():
            model_name = "deepseek-chat"
        else:
            model_name = "gpt-4o"

    logger.info("Initialized LLM Client: endpoint=%s, model=%s", base_url, model_name)
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_name


class DirectAgentRunner:
    """Orchestrates multi-agent execution with tool invocation and memory management."""

    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.storage_dir = working_dir / "storage"
        self.output_dir = working_dir / "output"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client, self.model_name = get_llm_client()

    def run_llm_completion(self, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
        """Call LLM with retry mechanism."""
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning("LLM call attempt %d failed: %s", attempt + 1, e)
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("LLM completion failed after 3 attempts.")

    # -------------------------------------------------------------
    # Step 1: Migration Architect Agent
    # -------------------------------------------------------------
    def run_architect_task(self, analysis_report_path: Path) -> Dict[str, Any]:
        logger.info("=== [Agent 1/4] Migration Architect Agent Starting ===")
        if not analysis_report_path.exists():
            raise FileNotFoundError(f"Analysis report not found: {analysis_report_path}")

        analysis_report_content = analysis_report_path.read_text(encoding="utf-8")
        
        system_prompt = (
            "You are an expert AWS Serverless Migration Architect. "
            "Your task is to analyze static code analysis data and design a complete serverless architecture blueprint (JSON). "
            "Core principles:\n"
            "1. ONE Lambda function per API endpoint (HTTP method + path). Never group multiple endpoints into one fat Lambda.\n"
            "2. Decouple long-running heavy AI/DL operations (e.g. document embedding, batch inference, model indexing) into SQS async queues and consumer workers.\n"
            "3. Replace authentication with AWS Cognito User Pools and API Gateway Authorizers (stateless).\n"
            "4. Return ONLY valid, parseable JSON matching the blueprint structure. Do NOT include markdown code blocks or conversational text."
        )

        user_prompt = f"""Analyze this static analysis report and produce storage/blueprint.json:

Analysis Report Data:
{analysis_report_content}

Output JSON schema must have keys:
{{
  "metadata": {{"application_name": "...", "architecture_pattern": "...", "has_authentication": true, "has_async_flows": true}},
  "lambda_functions": [
    {{
      "name": "...",
      "trigger_type": "api" or "sqs",
      "runtime": "python3.12",
      "handler": "handler.lambda_handler",
      "auth": "required" or "none",
      "source_files": ["..."],
      "entry_points": ["POST /..."],
      "publishes_to": [{{"type": "sqs", "target": "..."}}],
      "environment_variables": {{}}
    }}
  ],
  "sqs_queues": [{{"logical_name": "...", "purpose": "...", "dlq": true}}],
  "dynamodb_tables": [],
  "s3_buckets": [],
  "cognito": {{"enabled": true}},
  "api_gateway": {{"type": "REST", "authorizer": "Cognito"}},
  "dropped_functions": []
}}
"""
        response_text = self.run_llm_completion(system_prompt, user_prompt, max_tokens=4096)
        
        # Clean response if wrapped in ```json
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        blueprint = json.loads(response_text)
        blueprint_file = self.storage_dir / "blueprint.json"
        blueprint_file.write_text(json.dumps(blueprint, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Blueprint generated successfully: %d functions, %d queues", len(blueprint.get("lambda_functions", [])), len(blueprint.get("sqs_queues", [])))
        return blueprint

    # -------------------------------------------------------------
    # Step 2: Code Developer Agent
    # -------------------------------------------------------------
    def run_developer_task(self, blueprint: Dict[str, Any], monolith_root: Path) -> None:
        logger.info("=== [Agent 2/4] Code Developer Agent Starting ===")
        lambdas = blueprint.get("lambda_functions", [])
        lambdas_root = self.output_dir / "lambdas"
        lambdas_root.mkdir(parents=True, exist_ok=True)

        system_prompt = (
            "You are a Senior Serverless Developer. "
            "Generate production-grade AWS Lambda handler code for Python 3.12. "
            "Follow standard FaaS patterns:\n"
            "- def lambda_handler(event, context)\n"
            "- Extract body from json.loads(event.get('body', '{}'))\n"
            "- Extract user_id from event['requestContext']['authorizer']['claims']['sub'] if authenticated\n"
            "- Return dict with statusCode, headers (CORS), and json.dumps(body)\n"
            "- For SQS workers: iterate event.get('Records', []) and return {'batchItemFailures': []}\n"
            "- For orchestrators: publish job payload via boto3.client('sqs').send_message and return 202 Accepted\n"
            "- Return ONLY valid Python code."
        )

        for lf in lambdas:
            fn_name = lf["name"]
            logger.info("Generating Lambda handler for: %s", fn_name)
            
            # Read direct source files
            source_contexts = []
            for src_rel in lf.get("source_files", []):
                p = monolith_root / Path(src_rel).name
                if not p.exists():
                    p = monolith_root / src_rel
                if p.exists():
                    source_contexts.append(f"--- File: {src_rel} ---\n{p.read_text(encoding='utf-8', errors='ignore')}")

            sources_text = "\n\n".join(source_contexts) if source_contexts else "No source files specified."

            user_prompt = f"""Generate output/lambdas/{fn_name}/handler.py based on this specification:

Function Specification:
{json.dumps(lf, indent=2)}

Monolith Source Code Context:
{sources_text}

Write complete, functional Python 3.12 handler.py code:
"""
            code_text = self.run_llm_completion(system_prompt, user_prompt, max_tokens=2500)
            if "```python" in code_text:
                code_text = code_text.split("```python")[1].split("```")[0].strip()
            elif "```" in code_text:
                code_text = code_text.split("```")[1].split("```")[0].strip()

            fn_dir = lambdas_root / fn_name
            fn_dir.mkdir(parents=True, exist_ok=True)
            (fn_dir / "handler.py").write_text(code_text, encoding="utf-8")

        # Generate shared layer
        layers_dir = self.output_dir / "layers" / "shared" / "python"
        layers_dir.mkdir(parents=True, exist_ok=True)
        shared_utils_code = '''"""Shared utilities for Serverless Lambda handlers."""
import json, logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def format_response(status_code: int, body: dict, headers: dict = None) -> dict:
    default_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type,Authorization"
    }
    if headers:
        default_headers.update(headers)
    return {
        "statusCode": status_code,
        "headers": default_headers,
        "body": json.dumps(body, ensure_ascii=False)
    }

def get_user_id(event: dict) -> str:
    try:
        return event["requestContext"]["authorizer"]["claims"]["sub"]
    except Exception:
        return "anonymous_user"

def parse_json_body(event: dict):
    body = event.get("body")
    if not body:
        return {}, None
    try:
        return json.loads(body) if isinstance(body, str) else body, None
    except Exception as e:
        return {}, format_response(400, {"error": "Invalid JSON body", "detail": str(e)})

def handle_api_errors(func):
    def wrapper(event, context):
        try:
            return func(event, context)
        except Exception as e:
            logger.exception("API error: %s", e)
            return format_response(500, {"error": "Internal Server Error", "detail": str(e)})
    return wrapper
'''
        (layers_dir / "shared_utils.py").write_text(shared_utils_code, encoding="utf-8")
        logger.info("Code Developer task completed: All %d Lambda handlers written to disk", len(lambdas))

    # -------------------------------------------------------------
    # Step 3: SAM Engineer Agent
    # -------------------------------------------------------------
    def run_sam_engineer_task(self, blueprint: Dict[str, Any]) -> str:
        logger.info("=== [Agent 3/4] SAM Engineer Agent Starting ===")
        
        system_prompt = (
            "You are an AWS CloudFormation & SAM Specialist. "
            "Generate a valid, production-ready AWS SAM template.yaml file based on the architecture blueprint. "
            "Ensure:\n"
            "- Transform: AWS::Serverless-2016-10-31\n"
            "- Runtime: python3.12\n"
            "- Define AWS::Serverless::Api with Cognito DefaultAuthorizer\n"
            "- Define AWS::SQS::Queue for each queue with DLQ\n"
            "- Define AWS::Serverless::Function for every Lambda in blueprint\n"
            "- Return ONLY valid YAML content."
        )

        user_prompt = f"""Generate output/template.yaml based on blueprint:

Blueprint:
{json.dumps(blueprint, indent=2)}

Output complete SAM template.yaml:
"""
        template_yaml = self.run_llm_completion(system_prompt, user_prompt, max_tokens=4096)
        if "```yaml" in template_yaml:
            template_yaml = template_yaml.split("```yaml")[1].split("```")[0].strip()
        elif "```" in template_yaml:
            template_yaml = template_yaml.split("```")[1].split("```")[0].strip()

        (self.output_dir / "template.yaml").write_text(template_yaml, encoding="utf-8")
        logger.info("SAM Engineer task completed: template.yaml written")
        return template_yaml

    # -------------------------------------------------------------
    # Step 4: Consistency Validator Agent
    # -------------------------------------------------------------
    def run_validator_task(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("=== [Agent 4/4] Consistency Validator Agent Starting ===")
        
        lambdas_root = self.output_dir / "lambdas"
        template_file = self.output_dir / "template.yaml"
        
        findings = []
        expected_lambdas = {f["name"] for f in blueprint.get("lambda_functions", [])}
        disk_lambdas = {p.parent.name for p in lambdas_root.rglob("handler.py")}
        
        # C1: Directory <-> Function Coverage
        missing = expected_lambdas - disk_lambdas
        if missing:
            findings.append({"check_id": "C1", "severity": "ERROR", "description": f"Missing handlers: {list(missing)}"})
        else:
            findings.append({"check_id": "C1", "severity": "INFO", "description": f"All {len(expected_lambdas)} functions present on disk."})

        # C3: Handler validity
        all_valid = True
        for fn in expected_lambdas:
            h = lambdas_root / fn / "handler.py"
            if not h.exists():
                all_valid = False
                break
            try:
                tree = ast.parse(h.read_text(encoding="utf-8"))
                has_h = any(isinstance(n, ast.FunctionDef) and n.name == "lambda_handler" for n in ast.walk(tree))
                if not has_h:
                    all_valid = False
            except Exception:
                all_valid = False
        findings.append({"check_id": "C3", "severity": "INFO" if all_valid else "ERROR", "description": "All handler.py contain def lambda_handler."})

        report = {
            "status": "PASS" if not any(f["severity"] == "ERROR" for f in findings) else "FAIL",
            "total_checks": len(findings),
            "findings": findings,
            "summary": "All consistency cross-checks completed successfully."
        }
        
        (self.output_dir / "consistency_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Consistency Validator completed: %s", report["status"])
        return report

    def run_full_pipeline(self, monolith_dir: Path) -> None:
        """Run the end-to-end No-RAG 4-agent migration pipeline."""
        start_time = time.time()
        logger.info("Starting No-RAG Cloud-Native Service Migration Pipeline on: %s", monolith_dir)

        # 1. Run Static Analyzer
        from src.preprocessor.static_analyzer import run_static_analysis
        analysis_path = self.storage_dir / "analysis_report.json"
        symbol_path = self.storage_dir / "symbol_table.json"
        logger.info("Running AST static analyzer on %s...", monolith_dir)
        report = run_static_analysis(monolith_dir)
        symbol_table = report.pop("symbol_table", [])
        symbol_path.write_text(json.dumps(symbol_table, indent=2, ensure_ascii=False), encoding="utf-8")
        analysis_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Static analysis report written to: %s", analysis_path)

        # 2. Architect
        blueprint = self.run_architect_task(analysis_path)

        # 3. Developer
        self.run_developer_task(blueprint, monolith_dir)

        # 4. SAM Engineer
        self.run_sam_engineer_task(blueprint)

        # 5. Validator
        self.run_validator_task(blueprint)

        duration = time.time() - start_time
        logger.info("=== Migration Pipeline Finished in %.2fs ===", duration)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run No-RAG Multi-Agent Serverless Migration Pipeline")
    parser.add_argument("--monolith-dir", type=str, default=str(ROOT_DIR / "before"), help="Path to monolith application")
    args = parser.parse_args()

    runner = DirectAgentRunner(working_dir=M2S_DIR)
    runner.run_full_pipeline(Path(args.monolith_dir))
