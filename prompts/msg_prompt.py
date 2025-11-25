from config.config import Config

def refactor_msg(configs: Config) -> str:
    return f"""
            Hi! Your task is to:
            1. Use "load_file_tree" tool to get the file tree of the AI model code at `{configs.input_path}`. Args of the tool: root_path=`{configs.input_path}`.
            2. View all files obtained in step 1 using "view_text_file" tool as needed to understand the code structure and functionality.
            3. Analyze the the code information you obtained and refactor the code into multiple functional python files as needed for k8s deployment. An entry `app.py` file should be created to run the model. A dependency file `requirements.txt` should be generated. The refactored code should be saved at `{configs.output_path}` using "write_text_file" tool as needed.
            4. Use "py_ruff_fix" tool to fix any linting or dependency issues for the obtained refactored codes in `{configs.output_path}`. Args of the tool: root_path=`{configs.output_path}`.
            5. Fix the issues found in step 4 using "write_text_file" tool as needed.
            """

def devops_msg(configs: Config) -> str:
    return f"""
        Hi DevOps specialist! I have refactored the AI model code and it's located at path "{configs.output_path}". Then please help me with the following tasks:
        1.1 Check the "requirements.txt" file in the "{configs.output_path}" folder. Create a Dockerfile named "{configs.dockerfile_name}" at path "{configs.deploy_path}" for containerizing the AI model function-service, copy "{configs.output_path}" into the docker image.
        1.2 For the created Dockerfile, you can refer to the Dockerfile template at path "{configs.template_dockerfile_path}" using the existing image as base image.
        2.1 Create a Kubernetes YAML file named "{configs.k8s_yaml_name}" at path "{configs.deploy_path}" for deploying this service on Kubernetes. 
        2.2 For the created Kubernetes YAML file, you can refer to the Kubernetes YAML template at path "{configs.template_yaml_path}" but not directly copy it, you need to modify the configuration according to the AI model function-service.
        3.1 Create bash script "{configs.deploy_script_name}" at path "{configs.deploy_path}" for building the Docker image, the target platform is "linux/amd64", tag it and push it to the remote docker registry "{configs.remote_docker_registry}", and use the yaml to deploy on Kubernetes cluster.
        3.2 Also create bash script "{configs.undeploy_script_name}" at path "{configs.deploy_path}"  for undeploying the service and delete the built docker images.
        """