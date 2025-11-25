import asyncio
import os
import argparse

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.model import DashScopeChatModel
from agentscope.plan import PlanNotebook, Plan, SubTask
from config.config import Config

from agentscope.agent import ReActAgent, UserAgent
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import (Toolkit, execute_python_code,
                             execute_shell_command, view_text_file,
                             write_text_file, insert_text_file)
from tools.ToolRuffFix import py_ruff_fix
from tools.ToolFileList import load_file_tree
from agentscope.message import Msg


from prompts.msg_prompt import refactor_msg, devops_msg

def build_toolkit() -> Toolkit:
    toolkit = Toolkit()
    toolkit.register_tool_function(load_file_tree)
    toolkit.register_tool_function(view_text_file)
    toolkit.register_tool_function(write_text_file)
    toolkit.register_tool_function(execute_shell_command)
    toolkit.register_tool_function(py_ruff_fix)
    return toolkit

async def list_tools(plan_notebook: PlanNotebook) -> None:
    """列出PlanNotebook提供的工具函数。"""
    print("Provide Tools from PlanNotebook：")
    for tool in plan_notebook.list_tools():
        print(tool.__name__)
async def manual_plan_specification(plan_notebook: PlanNotebook, configs: Config) -> None:
    """手动计划规范示例。"""
    await plan_notebook.create_plan(
        name="AI Model refactor & deployment on k8s",
        description="Toolchain for refactoring AI models and deploying them on Kubernetes",
        expected_outcome="Refactored AI models with multiple python files, deployment dockerfile, yaml files for Kubernetes, and deployment scripts.",
        subtasks=[
            SubTask(
                name=f"Refactor the Model Code in path: '{configs.input_path}'",
                description=refactor_msg(configs),
                expected_outcome=f"Refactored AI model code with multiple python files, they should be saved at path '{configs.output_path}' using 'write_text_file' tool as needed.'",
            ),
            SubTask(
                name="Based on the refactored code, create deployment files for Kubernetes",
                description=devops_msg(configs),
                expected_outcome=f"The deployment dockerfile, yaml files for Kubernetes, and deployment scripts should be created at path '{configs.deploy_path}'.",
            ),
        ],
    )

    print("Current Prompt Msg：\n")
    msg = await plan_notebook.get_current_hint()
    print(f"{msg.name}: {msg.content}")

async def main(configs: Config, plan_notebook: PlanNotebook) -> None:
    await plan_notebook.create_plan(
        name="AI Model refactor & deployment on k8s",
        description="Toolchain for refactoring AI models and deploying them on Kubernetes",
        expected_outcome="Refactored AI models with multiple python files, deployment dockerfile, yaml files for Kubernetes, and deployment scripts.",
        subtasks=[
            SubTask(
                name=f"Refactor the Model Code in path: '{configs.input_path}'",
                description=refactor_msg(configs),
                expected_outcome=f"Refactored AI model code with multiple python files, they should be saved at path '{configs.output_path}' using 'write_text_file' tool as needed.'",
            ),
            SubTask(
                name="Based on the refactored code, create deployment files for Kubernetes",
                description=devops_msg(configs),
                expected_outcome=f"The deployment dockerfile, yaml files for Kubernetes, and deployment scripts should be created at path '{configs.deploy_path}'.",
            ),
        ],
    )
    toolkit = build_toolkit()
    RefactorAgent = ReActAgent(
        name="CodeRefactorExpert",
        sys_prompt=
        """
        You are an expert software engineer specialized in refactoring AI model code for deployment as function-services on Kubernetes.
        """,
        model=DashScopeChatModel(
            model_name=configs.backbone,
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=True,
            enable_thinking=False,
        ),
        memory=InMemoryMemory(),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
        plan_notebook=plan_notebook,
    )

    user = UserAgent(name="user")

    msg = Msg(
        "user",
        "Now start to finish the task by the given plan for refactoring and deployment.",
        "user",
    )
    while True:
        msg = await RefactorAgent(msg)
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break

# asyncio.run(manual_plan_specification())
if __name__ == "__main__":
    # init argparse
    parser = argparse.ArgumentParser(description="AI Model Code Refactoring Agent")
    parser.add_argument("--input_path", type=str, required=False, help="Father path of the AI model code to be refactored")
    parser.add_argument("--model_name", type=str, required=False, help="Name of the AI model to be refactored", default="saliency")
    parser.add_argument("--backbone", type=str, required=False, help="Backbone of Agent Model", default="qwen-max")
    args = parser.parse_args()

    workdir = "/Users/dinghaoxue/Projects/hello-agentscope/.tmp_cache/.pipes"
    model_name = "saliency"
    args.input_path = workdir
    args.model_name = model_name

    configs = Config(args.input_path, args.model_name, args.backbone)

    plan_notebook = PlanNotebook()
    asyncio.run(main(configs, plan_notebook))
    # asyncio.run(list_tools(plan_notebook))