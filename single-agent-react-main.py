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

from config.config import Config

import os
import asyncio
from logger.logger import get_logger
import argparse

def build_toolkit() -> Toolkit:
    toolkit = Toolkit()
    toolkit.register_tool_function(load_file_tree)
    toolkit.register_tool_function(view_text_file)
    toolkit.register_tool_function(write_text_file)
    toolkit.register_tool_function(execute_shell_command)
    toolkit.register_tool_function(py_ruff_fix)
    return toolkit


async def main(configs: Config):
    toolkit = build_toolkit()
    logger = get_logger(__name__, level=os.getenv("LOG_LEVEL", "INFO"))
    logger.info("Agent refactor run starting")
    # Agent for refactoring the AI model code for deploying as a function-service on kubernetes.
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
    )

    # K8SAgent = ReActAgent(
    #     name="DevOpsSpecialist",
    #     sys_prompt=
    #     f"""
    #     You are a DevOps specialist with expertise in turning the refactored AI model code into a deployable function-service on Kubernetes. Given the refactored code at path "{configs.output_path}", your task is as follows:
    #     1. Create the necessary Dockerfile with name "{configs.dockerfile_name}" at path "{configs.output_path}"
    #     2. Create "{configs.k8s_yaml_name}" at path "{configs.output_path}" for deploying this service on Kubernetes.
    #     3. Create "{configs.deploy_script_name}" at path "{configs.output_path}"for building the Docker image and deploying it to a Kubernetes cluster. Also create "{configs.undeploy_script_name}" for undeploying the service and delete the built docker images.
    #     """,
    #     model=DashScopeChatModel(
    #         model_name="qwen-max",
    #         api_key=os.environ["DASHSCOPE_API_KEY"],
    #         stream=True,
    #         enable_thinking=False,
    #     ),
    #     memory=InMemoryMemory(),
    #     formatter=DashScopeChatFormatter(),
    #     toolkit=toolkit,
    # )

    refactor_msg = Msg(
        name="user",
        content=f"""Hi! Your task is to:
        1. Use "load_file_tree" tool to get the file tree of the AI model code at `{configs.input_path}`. Args of the tool: root_path=`{configs.input_path}`.
        2. View all files obtained in step 1 using "view_text_file" tool as needed to understand the code structure and functionality.
        3. Analyze the the code information you obtained and refactor the code into multiple functional python files as needed for k8s deployment. An entry `app.py` file should be created to run the model. A dependency file `requirements.txt` should be generated. The refactored code should be saved at `{configs.output_path}` using "write_text_file" tool as needed.
        4. Use "py_ruff_fix" tool to fix any linting or dependency issues for the obtained refactored codes in `{configs.output_path}`. Args of the tool: root_path=`{configs.output_path}`.
        5. Fix the issues found in step 4 using "write_text_file" tool as needed.
        """,
        role="user",
    )

    await RefactorAgent(refactor_msg)
    logger.info("Refactoring task completed.")

    return

async def devops_main(configs: Config):
    from agents.DevopsAgent import create_devops_agent, example_msg
    logger = get_logger(__name__, level=os.getenv("LOG_LEVEL", "INFO"))
    logger.info("Devops Agent Starting")
    devops_agent = create_devops_agent(configs)
    devops_msg = example_msg(configs)
    await devops_agent(devops_msg)
    logger.info("Devops Agent Complete")
    return

if __name__ == "__main__":
    # init argparse
    parser = argparse.ArgumentParser(description="AI Model Code Refactoring Agent")
    parser.add_argument("--input_path", type=str, required=False, help="Father path of the AI model code to be refactored")
    parser.add_argument("--model_name", type=str, required=False, help="Name of the AI model to be refactored, also the folder name of the refactored code", default="saliency")
    parser.add_argument("--backbone", type=str, required=False, help="Backbone of Agent Model", default="qwen-max")
    parser.add_argument("--username", type=str, required=False, help="Username of the docker(harbor) registry", default="user-xuedh25")
    args = parser.parse_args()

    # workdir = "/Users/dinghaoxue/Projects/hello-agentscope/examples/"
    # model_name = "saliency"
    # args.input_path = workdir
    # args.model_name = model_name

    configs = Config(args.input_path, args.model_name, args.backbone, args.username)

    # configure top-level logger for the run
    app_logger = get_logger("singlemodel_main", level=os.getenv("LOG_LEVEL", "INFO"))
    app_logger.info("Starting singlemodel_main with input_path=%s", configs.input_path)
    app_logger.info("Model name: %s", args.model_name)
    app_logger.info("Output path: %s", configs.output_path)
    try:
        asyncio.run(main(configs))
        # asyncio.run(devops_main(configs))
    except Exception:
        app_logger.exception("Unhandled exception in main run")
        raise