import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

QWEN_API_CODE = [
    'qwen-plus', 'qwen-max', 'qwen3-max', 'qwen-flash', 'qwen-turbo',
    'qwen3-30b-a3b-instruct-2507', 
    'kimi-k2-thinking', 'Moonshot-Kimi-K2-Instruct',  
    'deepseek-v3.2', 'deepseek-v3.2-exp', 'deepseek-v3.1', 'deepseek-r1-0528',
    'abab6.5g-chat', 'glm-4.6'
]


class BackBoneConfig:
    def __init__(self, backbone: str = 'qwen-max'):
        assert backbone in QWEN_API_CODE
        self.backbone = backbone
    
    def get_backbone(self) -> str:
        # list available backbones (each row has 5 backbones)
        print("Available backbones:")
        for i in range(0, len(QWEN_API_CODE), 5):
            print(", ".join(QWEN_API_CODE[i:i+5]))
        # prompt user to select backbone
        backbone_name = input("Please enter the backbone model name (default: qwen-max): ") or "qwen-max"

        if backbone_name not in QWEN_API_CODE:  
            print(f"Backbone model '{backbone_name}' is not supported. Using default '{self.backbone}'.")
            backbone_name = self.backbone
        return backbone_name

class Config:
    def __init__(self, workdir: str = PROJECT_ROOT, model_name: str = 'model', backbone: str = 'qwen-max', harbor_username: str ='user-xuedh25'):
        assert backbone in QWEN_API_CODE
        self.backbone = backbone
        self.input_path = os.path.join(workdir, model_name)
        
        self.deploy_path = os.path.join(workdir, f"{model_name}_service")
        if not os.path.exists(self.deploy_path):
            os.makedirs(self.deploy_path)

        self.output_path = os.path.join(self.deploy_path, "src")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)


        """
        Config for single model deployment
        """
        self.remote_docker_registry = f"gpu-harbor.act.buaa.edu.cn/{harbor_username}/"
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
        """
        DAG
        """
        self.dag_output_path = os.path.join(self.output_path, f"DAG_{model_name}.py")



__all__ = ['config']
