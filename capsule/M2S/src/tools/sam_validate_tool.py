import subprocess
from pathlib import Path
from typing import Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class SAMValidateInput(BaseModel):
    """Input schema for SAMValidateTool."""
    template_path: str = Field(
        ...,
        description="Relative path to the SAM template.yaml file to validate (e.g., 'output/template.yaml')."
    )


class SAMValidateTool(BaseTool):
    name: str = "SAMValidateTool"
    description: str = (
        "Validate an AWS SAM template file using cfn-lint. "
        "Pass the file path (not file content). Returns success status and validation output."
    )
    args_schema: Type[BaseModel] = SAMValidateInput
    root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])

    def __init__(self, root: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        if root:
            self.root = root

    def _run(self, template_path: str) -> str:
        target = (self.root / template_path).resolve()

        if not target.exists():
            return f"Error: File not found: {template_path}"

        try:
            proc = subprocess.run(
                ["cfn-lint", str(target)],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return "Error: cfn-lint command not found. Please ensure it is installed."

        output = (proc.stdout + proc.stderr).strip()
        if proc.returncode == 0:
            return "Validation Successful! No errors found."

        lines = output.splitlines()
        errors = [l for l in lines if any(l.strip().startswith(p) for p in ("E", "I"))]
        warnings = [l for l in lines if l.strip().startswith("W")]

        if errors:
            return f"Validation Failed (errors must be fixed):\n{output}"
        if warnings:
            return (
                f"Validation Passed with warnings (no errors, safe to deploy):\n"
                f"{output}"
            )
        return f"Validation Failed:\n{output}"
