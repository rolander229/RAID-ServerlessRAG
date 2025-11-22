import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class Config:
    def __init__(self, workdir: str = PROJECT_ROOT, model_name: str = 'model', backbone: str = 'qwen-max'):
        assert backbone in ['qwen-max', 'qwen3-max']
        self.backbone = backbone
        self.input_path = os.path.join(workdir, model_name)
        self.output_dir_name = f"{model_name}_service"
        self.output_path = os.path.join(workdir, self.output_dir_name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        """
        Config for single model deployment
        """
        self.dockerfile_name = "Dockerfile"
        self.requirements_name = "requirements.txt"
        self.k8s_yaml_name = "deployment.yaml"
        self.deploy_script_name = "deploy.sh"
        self.undeploy_script_name = "undeploy.sh"

__all__ = ['config']
