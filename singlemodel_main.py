from agentscope.agent import ReActAgent, UserAgent
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit, execute_python_code, execute_shell_command, ToolResponse
from agentscope.message import Msg

from config.config import Config

import os
import asyncio
from logger.logger import get_logger
import argparse

def build_toolkit() -> Toolkit:
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)
    toolkit.register_tool_function(execute_shell_command)
    return toolkit


async def main(configs):
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
            model_name="qwen3-max",
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
        1. Analyze the provided AI model code at `{configs.input_path}`, refactoring the code into multiple functional python files as needed. An entry `app.py` file should be created to run the model.
        2. You do not need to execute the code to validate, just refactor based on your analysis.
        3. You need generate the requirements.txt file listing all dependencies, dockfile and k8s yaml files.
        4. The refactored code should be saved at `{configs.output_path}`. ".
        """,
        role="user",
    )


    await RefactorAgent(refactor_msg)
    logger.info("Refactoring task completed.")

    # await K8SAgent(msg2)
    # print("Deployment files creation task completed.")
    return


if __name__ == "__main__":
    # init argparse
    parser = argparse.ArgumentParser(description="AI Model Code Refactoring Agent")
    parser.add_argument("--input_path", type=str, required=False, help="Father path of the AI model code to be refactored")
    parser.add_argument("--model_name", type=str, required=False, help="Name of the AI model to be refactored", default="saliency")
    args = parser.parse_args()

    workdir = "/Users/dinghaoxue/Projects/hello-agentscope/.tmp_cache/.pipes"
    model_name = "saliency"
    args.input_path = workdir
    args.model_name = model_name

    configs = Config(args.input_path, args.model_name)

    # configure top-level logger for the run
    app_logger = get_logger("singlemodel_main", level=os.getenv("LOG_LEVEL", "INFO"))
    app_logger.info("Starting singlemodel_main with input_path=%s", configs.input_path)
    app_logger.info("Model name: %s", args.model_name)
    app_logger.info("Output path: %s", configs.output_path)
    try:
        asyncio.run(main(configs))
    except Exception:
        app_logger.exception("Unhandled exception in main run")
        raise