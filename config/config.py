import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class Config:
    def __init__(self, workdir: str = PROJECT_ROOT, model_name: str = 'model', backbone: str = 'qwen-max'):
        assert backbone in ['qwen-max', 'qwen3-max', 'qwen-plus']
        self.backbone = backbone
        self.input_path = os.path.join(workdir, model_name)
        self.output_dir_name = f"{model_name}_service/src"
        self.output_path = os.path.join(workdir, self.output_dir_name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.deploy_path = os.path.join(workdir, f"{model_name}_service")
        if not os.path.exists(self.deploy_path):
            os.makedirs(self.deploy_path)
        """
        Config for single model deployment
        """
        self.remote_docker_registry = "gpu-harbor.act.buaa.edu.cn/user-xuedh25/"
        self.dockerfile_name = "Dockerfile"
        self.requirements_name = "requirements.txt"
        self.k8s_yaml_name = "deployment.yaml"
        self.deploy_script_name = "deploy.sh"
        self.undeploy_script_name = "undeploy.sh"
        """
        Template Path
        """
        self.template_dir = os.path.join(workdir, "template")
        self.template_dockerfile_path = os.path.join(self.template_dir, "template.dockerfile")
        self.template_yaml_path = os.path.join(self.template_dir, "template.yaml")



__all__ = ['config']
