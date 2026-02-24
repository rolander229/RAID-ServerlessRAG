# TODO List
- [x] 实现配置预估工具。预估每个Stage的资源需求（内存、计算量）和执行时间，以便优化器做出更好的决策。（先实现静态方案，静态的作用于engine config的配置）
  - [x] 基本完成一个基本版本（输入配置，模型信息，估算每个模型的ray config），需要多重测试修复bug
- [ ] 实现更多优化 Pass，例如：
   - [x] Reranker Decomposition：将 reranking 阶段拆成多个子节点以提升并行度。 
   - [ ] PD分离的实现 
- [ ] DAG拆分工具的可视化（动画化）
- [ ] 增加原始未优化拆分的情况
- [ ] 推理引擎已经进行了一定程度的前后端分离，前端由graph scheduler负责DAG的调度和执行，后端由engine scheduler负责具体的引擎执行。后续考虑将后端由单机迁移到多pod的场景。或者进一步对前后端都进行重写