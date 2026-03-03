# vLLM 显存解析
> [vLLM显存管理](https://zhuanlan.zhihu.com/p/1916529253169734444)
# 分类
- torch类显存：torch 类显存是由 torch 相关 API 触发的（如Pytorch Allocator），其内容包括：
  - 权重（weights），与模型大小正相关，固定的 tensor 值；
  - 激活值（activation），运算过程中 tensor 值；
  - KV Cache（KV cache），KV 的 blocks，由 torch 创建 tensor 组成；
  - 其他显存（Others），用 torch 模块或者 API 创建的显存，比如 Sampler() 属于 torch.nn 模块；
- 非torch类显存
  - 通信显存：如 NCCL/HCCL 等通信过程中产生的部分缓存等；
  - Attention 模块：vLLM 中的 Attention 模块比较关键，有些操作采用底层 API 实现，有单独分配/释放显存行为；
  - 客户定义操作（custom ops）：操作可能有单独分配/释放显存行为.

# TODO
- Our resource estimation should support more cases:
  - Different hardware combinations
    - Multiple different GPUs (e.g., A100 + RTX3090)
    - Multiple Same GPUs
    - Multiple Nodes
  - Different Comm Groups / Parallelism strategies
    - Data Parallel
    - Model Parallel (Tensor, Pipeline, etc.)
    - Hybrid Parallel
