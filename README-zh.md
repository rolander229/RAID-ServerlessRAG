# ModelDeployAgent
## 快速开始
- 添加新模型进行重构
```bash
cd ./examples/
mkdir <新模型名称>
# 将模型文件复制到新文件夹
```
- 运行简单的具备工具使用能力的`ReAct`智能体
```bash
python single-agent-react-main.py --input_path <模型文件夹路径> --model_name <模型文件夹名称> --backbone <智能体骨干模型名称>
```
- （或者）运行简单的具备工具使用能力的`Plan`智能体
```bash
python single-agent-plan-main.py --input_path <模型文件夹路径> --model_name <模型文件夹名称> --backbone <智能体骨干模型名称>
```
- 检查重构后的模型，部署服务
```bash
cd ./examples/<模型文件夹名称>_service/
# 部署
bash deploy.sh
# 取消部署
bash undeploy.sh
```
## 其他
### 选择模型
- 单体简单模型
  - [基于CLIP的图形显著性]()
- 单体高级模型
  - [音乐生成](https://github.com/ace-step/ACE-Step)
### 工具
- 使用`uv`管理依赖
- 使用`ruff`检查代码风格并修复简单问题
- 工具构建的初步构想
  - 使用`ToolFileList.py`获取当前代码层次结构
  - 使用`ToolGraphAnalysis.py`分析代码函数调用图结构
  - 构建`ToolDependencyAnalysis.py`分析代码模块依赖结构
  - 最好能获取`.json`格式的图数据供智能体分析

### 智能体角色
- React 智能体
  - “思考-行动”模式的核心
  - 负责分解任务、规划行动和执行工具
- 用户智能体
  - 用户与React智能体之间的接口
  - 接收用户输入并将其传达给React智能体

### 格式化器
- `DashScopeChatFormatter`：用户-智能体对话格式化器
- `DashScopeMultiAgentFormatter`：（收集）多智能体对话格式化器

### 工作流程
- 对话
  - 用户-助手交互
  - 多智能体对话
- 辩论
  - 多用户类智能体 + 协调员智能体
  - 用户就同一主题提出不同观点，协调员使用`structured_model`（如结构化输出、`bool`类型）总结和评估这些观点
  - [链接](https://doc.agentscope.io/zh_CN/tutorial/workflow_multiagent_debate.html)
- 并发
  - 并发运行多个智能体处理不同任务
- 路由
  - 显式路由
  - 隐式路由
- 交接
  - ![交接-示意图](res/hands-off-img.png)

### 智能体组件
- 记忆
  - 可以为智能体添加/删除`Msg`记忆
- 记忆模式
  - `agent_control`：智能体自主控制
  - `static_control`：开发者控制
- 长期记忆
  - `record`：永久记录（写入）某些记忆
  - `retrieve`：在需要时检索某些记忆
- 中断
  - 中断智能体的回复
  - `handle_interrupt`：处理中断逻辑
- 流水线
- 规划
- RAG（检索增强生成）
- 快照
  - 在特定时间点保存智能体的状态快照
- 会话保存/加载
- **带有MCP服务的服务器**
  - 具体使用方法？
  - 从服务器调用工具
- 令牌计数
## MLOPs 实践
- [Google Cloud](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?hl=zh-cn)