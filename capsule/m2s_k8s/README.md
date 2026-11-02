# capsule.m2s_k8s — M2' 四 Agent K8s 迁移流水线

Mono2Sls 四角色迁移器的 K8s 化改造（M2'）。设计依据
`mopar-k8s/m2s-k8s-adaptation-plan.md`（角色重排、k8s_render_tool、k8s_migrated/ 契约）
与 `mopar-k8s/metrics-review-k8s.md`（验收器=评分器、D4/D5、配额三层分离）。

## 流水线（顺序六任务 × 四角色）

```
slice_manifest.json (bridge.py 产物)
  1 architect_blueprint    Architect   → blueprint.json
  2 developer_adapters     Developer   → adapters/slice_<id>_adapter.py × N
  3 engineer_render        K8s Engineer→ k8s_render_tool（资源数字 100% 确定性）
  4 engineer_overlay       K8s Engineer→ 白名单 overlay（资源数字禁改，机械强制）
  5 validator_checks       Validator   → kube_validate_tool（D4/D5 门禁 = 评分器规则）
  6 validator_signoff      Validator   → 一致性核对 + 签核
输出契约 k8s_migrated/{blueprint.json, manifests/*.yaml, adapters/*.py, report.json}
```

## 运行

```bash
source .venv/bin/activate
# 离线验证（FakeLLM 罐头响应，CI 可跑）
python -m capsule.m2s_k8s.runner \
    --manifest tests/fixtures/resnet50_manifest.json \
    --out runs/m2s_fake_resnet50 --fake-llm \
    --generated-at 2026-11-03T00:00:00Z
# 连通性自测（不发 chat 请求，不消耗 token）
python -m capsule.m2s_k8s.runner --check-llm
```

退出码：0 通过；1 跑通但校验/签核未过；2 被拦截（坏 manifest、角色违规、工具拒绝）。

## Ark LLM 底座配置（M2'-4）

**不 hardcode 任何 key**——全部走环境变量：

| 变量 | 说明 |
|---|---|
| `ARK_API_KEY` | Bearer token。由父会话用 `arkcli` 生成**专用** key（勿复用个人主 key），导出到 dell-65 shell 或 git-ignore 的 .env |
| `ARK_BASE_URL` | 默认 `https://ark.cn-beijing.volces.com/api/v3`（dell-65 可出公网） |
| `ARK_MODEL` | 模型 id 或接入点 id（`ep-...`） |
| `ARK_TIMEOUT_S` | 可选，默认 120 |

建模建议（总纲 §4）：Architect/Validator 用思考型；Developer 用代码型（DeepSeek-V3，
后续自建 Qwen2.5-Coder 做 RQ2 消融）。

真实运行：

```bash
export ARK_API_KEY=...   # arkcli 生成，勿入库
python -m capsule.m2s_k8s.runner \
    --manifest <slice_manifest.json> --out runs/m2s_ark_run \
    --generated-at $(date -u +%Y-%m-%dT%H:%M:%SZ)
```

实现只依赖标准库 `urllib`（不引入 openai SDK，避免污染 torch 钉住的 venv）；
429/5xx 指数退避重试 ≤3 次。

## 工具层（capsule/agent_tools/，无 LLM）

- `K8sRenderTool`：包 `capsule.k8s_render.render()`；blueprint 结构校验（缺字段报清晰路径错误）；
  返回 `{file_name: yaml_text}` + 从渲染产物中提取的 per-slice 数字摘要。
- `KubeValidateTool`：两级校验。①语法（PyYAML 流解析）+ 结构 + D4/D5 规则（与评分器同源，
  常量出自 `mapping_rules.py`：limits ≈ (alloc+600Mi)×1.3 ±16Mi、cpu=ceil(alloc/1769)、
  shm=256Mi、挂载点齐备、仅 slice-0 有 Ingress、MOPAR_OUT_PEERS=Service DNS）；②
  kubeconform schema 校验（尽力而为：找 `bin/kubeconform` → $PATH；缺失/离线时如实报
  `skipped`，①级仍是权威门禁）。

## 映射常量与专家基准

`agent_tools/mapping_rules.py` 是唯一常量源（1769/600/1.3/±16Mi/256Mi/DNS 正则/label 约定），
三处共用：渲染工具、校验器、专家基准。`m2s_k8s/expert_baseline.py` 从 manifest 独立推导期望
产物（不 import 渲染器，防"自己抄自己"），resnet50 推导结果人工核对后锁在
`tests/golden/resnet50_k3_expert_baseline.json`。

## 换 CrewAI 的接口预留

- 工具：普通类 `name/description/run(**kwargs)->dict`，包一层 `crewai.tools.BaseTool`
  + pydantic args_schema 即可；
- LLM：`LLMClient.chat(messages)->str`，与 crewai LLM 的 call 约定同构；
- 编排：`tasks.TASKS` 六任务定义（role/kind/instruction）可平移为 Crew 的 tasks.yaml，
  runner 的机械门禁（overlay 白名单、拦截、report）在 Crew 模式下作为 after-task 钩子保留。
