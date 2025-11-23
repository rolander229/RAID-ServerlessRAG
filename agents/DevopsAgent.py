from agentscope.agent import ReActAgent, UserAgent
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter
from agentscope.tool import (Toolkit, execute_python_code,
                             execute_shell_command, view_text_file,
                             write_text_file, insert_text_file)
from agentscope.message import Msg
from tools.ToolFileList import load_file_tree
from config.config import Config
import os

"""
TODO: dockerfile lint tool: https://github.com/hadolint/hadolint
"""

def build_toolkit() -> Toolkit:
    toolkit = Toolkit()
    toolkit.register_tool_function(load_file_tree)
    toolkit.register_tool_function(view_text_file)
    toolkit.register_tool_function(write_text_file)
    toolkit.register_tool_function(execute_shell_command)
    return toolkit

def create_devops_agent(configs: Config) -> ReActAgent:
    return ReActAgent(
        name="DevOpsSpecialist",
        sys_prompt=
        """
        You are a DevOps specialist with expertise in creating deployment files for AI model function-services on Kubernetes.
        """,
        model=DashScopeChatModel(
            model_name=configs.backbone,
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=True,
            enable_thinking=False,
        ),
        memory=InMemoryMemory(),
        formatter=DashScopeChatFormatter(),
        toolkit=build_toolkit()
    )

def example_msg(configs: Config) -> Msg:
    devops_msg = Msg(
        name="user",
        content=f"""
        Hi DevOps specialist! I have refactored the AI model code and it's located at path "{configs.output_path}". Then please help me with the following tasks:
        1.1 Check the "requirements.txt" file in the "{configs.output_path}" folder. Create a Dockerfile named "{configs.dockerfile_name}" at path "{configs.deploy_path}" for containerizing the AI model function-service, copy "{configs.output_path}" into the docker image.
        1.2 For the created Dockerfile, you can refer to the Dockerfile template at path "{configs.template_dockerfile_path}" using the existing image as base image.
        2.1 Create a Kubernetes YAML file named "{configs.k8s_yaml_name}" at path "{configs.deploy_path}" for deploying this service on Kubernetes. 
        2.2 For the created Kubernetes YAML file, you can refer to the Kubernetes YAML template at path "{configs.template_yaml_path}" but not directly copy it, you need to modify the configuration according to the AI model function-service.
        3.1 Create bash script "{configs.deploy_script_name}" at path "{configs.deploy_path}" for building the Docker image, the target platform is "linux/amd64", tag it and push it to the remote docker registry "{configs.remote_docker_registry}", and use the yaml to deploy on Kubernetes cluster.
        3.2 Also create bash script "{configs.undeploy_script_name}" at path "{configs.deploy_path}"  for undeploying the service and delete the built docker images.
        """,
        role="user",
    )
    return devops_msg

# 1. Use `view_text_file` tool to check the `requirements.txt` file in the folder and use `write_text_file` tool to create a Dockerfile named "{configs.dockerfile_name}" at path "{configs.output_path}" for containerizing the AI model function-service. You can refer to the Dockerfile template at path "{configs.template_dockerfile_path}" using the existing image.
# 2. Create a Kubernetes YAML file named "{configs.k8s_yaml_name}" at path "{configs.output_path}" for deploying this service on Kubernetes. You can refer to the Kubernetes YAML template at path "{configs.template_yaml_path}" using the existing node.
# 3. Create bash script "{configs.deploy_script_name}" at path "{configs.output_path}"for building the Docker image, tag it and push it to the remote docker registry "{configs.remote_docker_registry}", and use the yaml to deploy on Kubernetes cluster. Also create bash script "{configs.undeploy_script_name}" for undeploying the service and delete the built docker images.