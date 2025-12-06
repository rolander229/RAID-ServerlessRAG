# ModelDeployAgent
## QuickStart
- Add a new model for refactoring
```bash
cd ./examples/
mkdir <new_model_name>
# copy model files to the new folder
```
- Run a simple `ReAct` agent with tool use capability
```bash
python single-agent-react-main.py --input_path <model_folder_path> --model_name <model_folder_name> --backbone <agent_backbone_name>
```
- (Or) Run a simple `Plan` agent with tool use capability
```bash
python single-agent-plan-main.py --input_path <model_folder_path> --model_name <model_folder_name> --backbone <agent_backbone_name>
```
- Check the refactored model, deploy the service
```bash
cd ./examples/<model_folder_name>_service/
# deploy
bash deploy.sh
# undeploy
bash undeploy.sh
```
- See More Docs
  - [Design Philosophy](./docs/design.md)
  - [Others](./docs/others.md)
